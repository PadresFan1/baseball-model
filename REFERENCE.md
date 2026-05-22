# Baseball Model — Reference Guide

---

## Quick Start

```bash
# Daily predictions (run at 9am, 12pm, 4pm)
python model.py

# Backtest parameter search
python backtest.py

# Rebuild player→team mapping (only when adding new RP data)
python build_player_map.py

# Fetch remaining 2025 odds (Aug 17–Oct 1) from SportsGameOdds API
python fetch_2025_odds.py --test   # verify one event first
python fetch_2025_odds.py          # full pull (~608 events)

# Evaluate all backtest runs without running new trials
# Set N_RUNS = 0 in backtest.py, then:
python backtest.py
```

---

## File Structure

```
baseball-model/
├── model.py                    # Live daily prediction model
├── backtest.py                 # Parameter optimization via random search
├── build_player_map.py         # Builds player ID → team mapping from MLB API
├── fetch_2025_odds.py          # Pulls 2025 odds from SportsGameOdds API
│
├── statcast/
│   ├── batters_YYYY.csv        # Team-level batter statcast by game (2021-2025)
│   └── pitchers_YYYY.csv       # Team-level pitcher statcast by game (2021-2025)
│   # Columns: player_name(team), game_date, babip, xwoba, barrels_per_pa_percent, pa
│
├── rp/
│   └── *.csv                   # Relief pitcher statcast by division/year (80,312 rows)
│   # Columns: player_id, game_date, game_pk, total_pitches, babip, woba, xwoba,
│   #          k_percent, bb_percent, barrels_per_pa_percent, pa
│
├── constants/
│   ├── woba_fip_constants.csv  # Season wOBA weights + FIP constant (2015-2026)
│   ├── park_factors.csv
│   └── park_factors_handedness.csv
│
├── historical_data/
│   ├── historical_game_logs.json    # MLB API game logs 2021-2025 (cached, incremental)
│   ├── player_team_map.json         # {season: {player_id: fg_abbrev}} for 2021-2025
│   ├── statcast_cache.json          # Preprocessed statcast data (delete to rebuild w/ 2025)
│   ├── bullpen_cache.json           # Preprocessed bullpen data (aggregated to team-game)
│   ├── mlb_odds_dataset.json        # Historical odds 2021–Aug 16, 2025
│   ├── odds_2025_supplement.csv     # Aug 17–Oct 1, 2025 odds (from SportsGameOdds)
│   ├── search_*_r{N}.csv            # Backtest run results (one per run, round tagged)
│   └── evaluation_*.csv             # Combined evaluation files
│
├── offense_YYYY.csv            # FanGraphs team offense stats
├── pitching_YYYY.csv           # FanGraphs team pitching stats
├── predictions_log.csv         # Daily predictions log (all runs)
├── predictions_log.xlsx        # Color-coded Excel
├── splits_cache.json           # Platoon split data cache (6hr TTL)
├── email_last_sent.txt         # Timestamp of last email (2hr throttle — delete to force)
└── .env                        # Credentials — NEVER commit
```

---

## Environment Variables (.env)

```
API_KEY=                        # The Odds API key
GOOGLE_CREDENTIALS_PATH=        # Path to google_credentials.json
EMAIL_SENDER=                   # Gmail address that sends reports
EMAIL_APP_PASSWORD=             # Gmail app password (16 chars, NO spaces)
EMAIL_RECIPIENT=                # Address to receive reports
BANKROLL=                       # Betting bankroll in dollars (default: 1000)
SPORTSGAMEODDS_KEY=             # SportsGameOdds API key (for fetch_2025_odds.py)
```

---

## Team Abbreviation Maps

### Statcast CSV → FanGraphs (STATCAST_TEAM_MAP in backtest.py)
```
AZ→ARI, CWS→CHW, KC→KCR, SD→SDP, SF→SFG, TB→TBR, WSH→WSN
```

### Historical Odds → FanGraphs (ODDS_TO_FG in backtest.py)
```
KC→KCR, SD→SDP, SF→SFG, TB→TBR, WAS→WSN, OAK→ATH
```

### MLB API → FanGraphs (TEAM_MAP in model.py + backtest.py)
30-team dict mapping FG abbreviation to MLB team ID (e.g. NYY→147).

---

## model.py — Architecture

### Data Flow Per Run
1. Load odds from The Odds API (cached 2.5 hrs) → filter to today's upcoming games
2. Load team stats (cached 4 hrs):
   - Rolling runs scored/allowed (last 7 and 15 games) via MLB Stats API gamelog
   - wOBA, BABIP, FIP, xFIP, K%, BB% cumulative season via MLB Stats API
   - xwOBA from Baseball Savant (live API, player-level aggregated to team)
3. Get today's probable starters → fetch season RA9 and innings pitched
4. Fetch confirmed lineups (MLB API hydrate=lineups) → load splits_cache.json
5. Fetch injury adjustments (MLB transactions endpoint, one call)
6. For each upcoming game:
   - Compute platoon factors if lineup confirmed
   - Run raw matchup (no platoon) + platoon-adjusted matchup
   - Blend rolling + season stats + starter quality + injury adj + platoon → ratings
   - Run 5,000 Poisson simulations → win probability + projected total
   - Compare to market odds → calculate edge
   - If edge > 7%: compute Kelly bet size (half-Kelly, 15% bankroll cap)
7. Separate games: edge found / no edge / awaiting starters
8. Log predictions to CSV/Excel/Google Sheets (two-pass result system)
9. Print accuracy report (included in email)
10. Send email if > 2 hours since last send

### Key Constants
```python
OFFENSE_WEIGHTS = {             # From round 5 backtest top-10 averages
    'rolling': 0.090,
    'woba':    0.378,           # Dominant offensive signal
    'xwoba':   0.264,
    'babip':   0.268,
}
PITCHING_WEIGHTS = {
    'rolling': 0.078,
    'fip':     0.240,
    'xfip':    0.220,
    'k_pct':   0.192,
    'bb_pct':  0.270,           # Consistently important
}
NUM_SIMULATIONS  = 5000
BANKROLL         = float(os.getenv('BANKROLL', '1000'))
KELLY_FRACTION   = 0.5          # Half-Kelly
MODEL_V2_START   = '2026-05-21' # Accuracy report split date

# Edge thresholds
ml_edge_min = 0.07    # Minimum edge to bet (hardcoded in game loop)
ml_edge_max = 0.08    # Maximum edge to bet (beyond this = model outlier, skip)
ou_threshold = 1.5    # Run differential to bet O/U (not yet backtested — round 6)

# Rating caps
rating_cap_low  = 0.65
rating_cap_high = 1.50

# Rolling blend
rolling_weight_7  = 0.70   # 7-day window weight
rolling_weight_15 = 0.30   # 15-day window weight
```

### Kelly Criterion
```python
kelly_bet_size(model_prob, american_odds):
    b = odds/100 if odds > 0 else 100/abs(odds)
    kelly = (b*p - q) / b
    kelly_adj = min(kelly * 0.5, 0.15)   # half-Kelly, 15% cap
    bet_amount = BANKROLL * kelly_adj
```

### Injury Adjustment System
```python
get_injury_adjustments(team_ids, all_woba, all_fip, lg_avgs):
    # One call to MLB Stats API transactions endpoint
    # Finds: placed-on-IL but not yet activated = currently injured
    # Filters to < 14 days (taper: (14 - days) / 14, full→zero)
    # Hitter impact: (player_woba/lg_woba - 1) * pa_share * taper
    #   NO cap on pa_share (naturally bounded ~12-15% for regulars)
    # Pitcher impact: (lg_fip/player_fip - 1) * ip_share * taper
    #   Capped at 25% IP share
    # Output: {team_fg: {off_adj, pit_adj, notes}}
```

### Platoon Split System
```python
get_confirmed_lineups(game_date)        → {(home_fg, away_fg): lineup_dict}
get_players_hand(player_ids)            → batch handedness, 1 API call
get_player_split_woba(player_id, ...)   → {vs_lhp, vs_rhp}, min 30 PA
get_pitcher_split_fip(pitcher_id, ...)  → {vs_lhb, vs_rhb}, min 5 IP
get_all_platoon_splits(lineup_data)     → splits_cache.json (6hr TTL)
compute_platoon_factors(...)            → (home_factor, away_factor, notes)
    # factor = lineup_avg_woba_vs_pitcher_hand / lg_woba
    # No cap — PA threshold and ≥3 batter minimum are the safeguards
    # Applied: home_lambda *= home_factor, away_lambda *= away_factor
```

Platoon output: **NOT shown in terminal/email** — tracked in log for 3-week evaluation.
Analysis appears in accuracy report after 20+ settled confirmed-lineup ML bets.

### Email System
- `_Tee` class captures all stdout to buffer simultaneously
- Buffer reset before "Upcoming games today" — excludes startup noise
- Email sent AFTER print_accuracy_report() — accuracy report IS included
- 2-hour throttle: `email_last_sent.txt` stores last send timestamp
- Subject: "Baseball Model — {9am/12pm/4pm} Run — {date}"
- From: "Betting Model <sender>"
- Delete `email_last_sent.txt` to force immediate send

### Prediction Log Schema
```
date, game_time, home_team, away_team, bet_type,
run1_bet_team, run1_model_pct, run1_book_line, run1_edge,
run2_bet_team, run2_model_pct, run2_book_line, run2_edge, run2_change,
run3_bet_team, run3_model_pct, run3_book_line, run3_edge, run3_change,
final_run, final_bet_team, final_model_pct, final_book_line, final_edge,
raw_model_pct, home_platoon_factor, away_platoon_factor, platoon_confirmed,
actual_home_score, actual_away_score, winner, result
```

Key fields:
- `game_time`: "HH:MM" 24-hour MST — determines which run is "final"
- `run2_change / run3_change`: EDGE GAINED / EDGE LOST / PICK CHANGED / —
- `final_run`: last run BEFORE game start (not last logged run)
- `result`: evaluated against final_bet_team only (last pre-game prediction)
- `actual_home/away_score`: only populated for yesterday's rows (never today)
- `book_line` for O/U: "8.5@-110" format (total@odds for ROI calculation)
- Sort: newest date first, chronological by game_time within each day

### Result System (Two-Pass)
- Pass 1: Log today's predictions as PENDING (no score, no result)
- Pass 2: Update yesterday's rows with actual results from get_yesterdays_results()
- This prevents back-to-back games (same teams two days) from getting wrong results

---

## backtest.py — Architecture

### Module Load Sequence
```
1. Load mlb_odds_dataset.json → build odds_df
   + Merge odds_2025_supplement.csv if present
2. Load game logs from historical_game_logs.json
   (pull_historical_game_logs() pulls only missing seasons — incremental)
3. Load statcast data [2021-2025] (statcast_cache.json or CSVs)
   DELETE statcast_cache.json to rebuild with 2025
4. Load bullpen data (bullpen_cache.json or rp/*.csv)
5. precompute_all_rolling([2021-2025], game_logs, odds_df)
   → rolling_team_cache[(season, team, date)] and rolling_lg_cache[(season, date)]
   → 8.3x speedup via prefix sums (O(1) lookups per game)
```

### Current Param Space (Round 6)
```python
PARAM_ROUND = 6
# R1: base params, no statcast
# R2: statcast added (xwoba_bat, babip_bat, barrel_bat/pit)
# R3: bullpen added — HURT model (removed)
# R4: bullpen removed, favorites_only locked False
# R5: locked edge/caps, barrel_bat dropped, weights rebalanced
# R6: added ou_run_threshold, O/U tracking, 2025 out-of-sample data

param_space = {
    # LOCKED
    'ml_edge_min':      [0.07],
    'ml_edge_max':      [0.08],
    'favorites_only':   [False],
    'rating_cap_low':   [0.65],
    'rating_cap_high':  [1.50],

    # NEW in R6 — O/U threshold optimization
    'ou_run_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],

    # SEARCHING
    'rolling_weight_7': [0.50, 0.60, 0.70, 0.80],

    # Offense (barrel_bat dropped — -0.298 in r4)
    'w_rolling_off': [0.05, 0.08, 0.10],
    'w_woba':        [0.20, 0.30, 0.40],
    'w_xwoba_bat':   [0.10, 0.20, 0.30],
    'w_babip_bat':   [0.15, 0.25, 0.35],

    # Pitching
    'w_rolling_pit': [0.05, 0.08, 0.10],
    'w_fip':         [0.10, 0.20, 0.30],
    'w_xfip':        [0.10, 0.20, 0.30],
    'w_k_pit':       [0.10, 0.20, 0.30],
    'w_bb_pit':      [0.15, 0.25, 0.35],
    'w_xwoba_pit':   [0.10, 0.20, 0.30],
    'w_babip_pit':   [0.05, 0.08, 0.12],
    'w_barrel_pit':  [0.15, 0.25, 0.35],
}
```

### O/U Tracking (new in R6)
```python
# Per trial, tracked alongside ML:
n_ou_bets    — bets placed (requires fd_total only, no juice needed)
ou_win_rate  — WIN/(WIN+LOSS)
ou_roi       — only calculated for bets WITH pre-game juice data
ou_profit    — dollar profit (juice-confirmed bets only)
combined_roi — (ml_profit + ou_profit) / (total_wagered) * 100
               ← primary sort metric for R6 evaluate_runs()
```

### Rating Formulas
```python
# Offense (higher = better)
home_off = (off_roll_r * w_rolling_off + woba_r * w_woba +
            xwoba_bat_r * w_xwoba_bat + babip_bat_r * w_babip_bat)

# Pitching (higher = better, inverted where lower stat = better)
home_pit = (pit_roll_r * w_rolling_pit + fip_r * w_fip + xfip_r * w_xfip +
            k_pit_r * w_k_pit + bb_pit_r * w_bb_pit +
            xwoba_pit_r * w_xwoba_pit + babip_pit_r * w_babip_pit +
            barrel_pit_r * w_barrel_pit)

# Batting ratios: team/league (higher = better)
# Pitching ratios: league/team (inverted, lower FIP = better pitching = higher ratio)

home_lambda = (home_off / away_pit) * lg_avg_runs
```

### evaluate_runs() Output
- Round summary: top ML ROI, top combined ROI, median, % profitable
- Per-run summary
- Top 10 results sorted by combined_roi (all sample sizes)
- Top 10 results filtered to n_bets >= 300
- Parameter breakdown: top 25% vs bottom 75% for discrete params
  including ou_run_threshold
- Weight correlations with ROI

### File Naming
```
search_YYYY-MM-DD_seed{N}_100trials_r6.csv
evaluation_YYYY-MM-DD_HH-MM_{N}runs.csv
```

---

## Backtest Results Summary

| Round | Trials | Avg ROI (n≥300) | Max ROI | Key Change |
|-------|--------|-----------------|---------|------------|
| 1-3   | 3,669  | -1.51%          | +14.67% | Base → statcast → bullpen |
| 4     | 1,000  | +0.11%          | +13.20% | Bullpen removed, fav locked |
| 5     | 1,000  | +0.53%          | +16.33% | Barrel_bat removed, rebalanced |
| 6     | running| —               | —       | O/U added, 2025 data |

Round 5 top trial: ROI=16.33%, n=403, win_rate=51.86%

### Key Findings Across All Rounds
- favorites_only=True: small-sample artifact (n~90-135 bets). At n≥300 all flip to False.
- w_bullpen_fat: -0.246 correlation (actively harmful). Removed R4.
- w_bullpen_qual: +0.007 (noise). Removed R4.
- w_barrel_bat: -0.298 in R4. Removed R5.
- w_woba: consistently +0.25 to +0.38. Most important offensive stat.
- w_bb_pit: consistently elevated (+0.19 avg in top-10 R5). Walks matter.
- Rolling weights (7-day/15-day): weaker signal, top R5 avg 0.68 (settled on 0.70).

---

## 2025 Data Coverage

| Source | Date Range | ML Odds | O/U Odds |
|--------|-----------|---------|----------|
| mlb_odds_dataset.json | 2021–Aug 16, 2025 | ✅ Full | ✅ Full |
| odds_2025_supplement.csv | Aug 17–Oct 1, 2025 | ⚠️ 28/608 games | ✅ 608 games |

**Why only 28 pre-game ML games**: SportsGameOdds API stores last-known odds only.
For finalized games, all bookmakers have post-game timestamps. Only bookmakers that
freeze their line before first pitch (rare) provide pre-game data. The total LINE
(overUnder field) is reliable for all games since it's set pre-game and doesn't move.

**2026 data**: building organically via model.py's daily The Odds API cache.
Complete 2026 dataset available at end of season.

---

## Pending Items

1. **Platoon split retrospective** (3 weeks): check if confirmed-lineup bets where
   platoon boosted the model outperform those where it lowered the model.
   Accuracy report section activates automatically at 20+ confirmed-lineup settled bets.

2. **Round 6 results**: evaluate ou_run_threshold findings, check if O/U adds
   positive combined ROI, refine if needed and run round 7.

3. **Full 2025 ML odds**: if The Odds API releases historical data or SportsGameOdds
   adds snapshot capability, fill Aug 17–Oct 1, 2025 ML gap.

4. **Platoon splits in backtest**: not yet implemented. Current backtest uses team-level
   stats only. Adding lineup-level platoon data would require confirmed lineups
   historically, which is not available.
