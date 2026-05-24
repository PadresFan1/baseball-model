# Baseball Model — Reference Guide

---

## Quick Start

```bash
# Daily predictions (run at 9am, 12pm, 4pm)
python model.py

# Backtest — set RUN_MODE in backtest.py before running
python backtest.py

# Round 9 player-level pipeline (first run only — pulls lineups ~2-3hrs)
# Set RUN_MODE = 'player_level_meta' then:
python backtest.py

# Download player-level Statcast CSVs (run once, ~15 min)
python fetch_player_data.py --test   # verify column names first
python fetch_player_data.py          # full 2021-2025 pull (15 files)

# Dual strategy sim on team-level features (fast, no data build)
# Set RUN_MODE = 'dual_strategy' then:
python backtest.py

# Fetch remaining 2025 odds (Aug 17–Oct 1) from SportsGameOdds API
python fetch_2025_odds.py --test   # verify one event first
python fetch_2025_odds.py          # full pull (~608 events)

# Rebuild player→team mapping (only when adding new RP data)
python build_player_map.py
```

---

## File Structure

```
baseball-model/
├── model.py                    # Live daily prediction model
├── backtest.py                 # Backtest + meta model + player-level pipeline
├── fetch_player_data.py        # Downloads individual player Statcast CSVs from Savant
├── build_player_map.py         # Builds player ID → team mapping from MLB API
├── fetch_2025_odds.py          # Pulls 2025 odds from SportsGameOdds API
│
├── player_data/                # Round 9 player-level Statcast (NEW)
│   ├── batters_vs_L_YYYY.csv   # Per-batter per-game stats vs LHP (2021-2025)
│   ├── batters_vs_R_YYYY.csv   # Per-batter per-game stats vs RHP (2021-2025)
│   └── starters_YYYY.csv       # Per-starter per-game stats, PA >= 12 (2021-2025)
│   # Columns: player_id, player_name, game_date, game_pk, pa, woba, xwoba,
│   #          so, bb, hrs, k_percent, bb_percent, obp, hardhit_percent,
│   #          swing_miss_percent, pitcher_run_value_per_100
│
├── Statcast/
│   ├── batters_YYYY.csv        # Team-level batter statcast by game (2021-2025)
│   └── pitchers_YYYY.csv       # Team-level pitcher statcast by game (2021-2025)
│   # Columns: player_name(team abbrev), game_date, babip, xwoba, pa, ...
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
│   ├── game_lineups.json            # Box score batting orders + starter IDs (NEW Round 9)
│   │                                # {season: {game_pk: {home_lineup, away_lineup,
│   │                                #   home/away_starter_id/hand/ip}}}
│   ├── pitcher_hand_cache.json      # {player_id: 'L'/'R'} from MLB People API (NEW)
│   ├── player_team_map.json         # {season: {player_id: fg_abbrev}} for 2021-2025
│   ├── statcast_cache.json          # Preprocessed team-level statcast (delete to rebuild)
│   ├── bullpen_cache.json           # Preprocessed bullpen data (aggregated to team-game)
│   ├── mlb_odds_dataset.json        # Historical odds 2021–Aug 16, 2025
│   ├── odds_2025_supplement.csv     # Aug 17–Oct 1, 2025 odds (from SportsGameOdds)
│   ├── search_*_r{N}.csv            # Backtest run results (one per run, round tagged)
│   └── evaluation_*.csv             # Combined evaluation files
│
├── predictions/
│   ├── predictions_log.csv     # Daily predictions log (all runs) — 35 cols incl. model_strategy
│   ├── predictions_log.xlsx    # Color-coded Excel
│   └── archive/                # One-time migration backups (predictions_log_archived_YYYYMMDD.*)
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
date, game_time, game_num, home_team, away_team, bet_type, model_strategy,
run1_bet_team, run1_model_pct, run1_book_line, run1_edge,
run2_bet_team, run2_model_pct, run2_book_line, run2_edge, run2_change,
run3_bet_team, run3_model_pct, run3_book_line, run3_edge, run3_change,
final_run, final_bet_team, final_model_pct, final_book_line, final_edge,
raw_model_pct, home_platoon_factor, away_platoon_factor, platoon_confirmed,
actual_home_score, actual_away_score, winner, result, ou_direction
```
35 columns total. `model_strategy` is the 7th column (after `bet_type`).

Key fields:
- `model_strategy`: 'SNIPER' for ML bets, '' for O/U. Will be 'SNIPER'/'ENFORCER' once dual-strategy is live.
- `game_time`: "HH:MM" 24-hour MST — determines which run is "final"
- `run2_change / run3_change`: EDGE GAINED / EDGE LOST / PICK CHANGED / —
- `final_run`: last run BEFORE game start (not last logged run)
- `result`: evaluated against final_bet_team only (last pre-game prediction)
- `actual_home/away_score`: only populated for yesterday's rows (never today)
- `book_line` for O/U: "8.5@-110" format (total@odds for ROI calculation)
- `ou_direction`: HIGH/LOW/CLOSE + signed error once scores available
- Sort: newest date first, chronological by game_time within each day

### Result System (Two-Pass)
- Pass 1: Log today's predictions as PENDING (no score, no result)
- Pass 2: Update yesterday's rows with actual results from get_yesterdays_results()
- This prevents back-to-back games (same teams two days) from getting wrong results

### Production Startup Sequence
```
Step 1  archive_existing_prediction_logs()
          Detects old schema (no model_strategy col) → moves CSV/XLSX to
          predictions/archive/predictions_log_archived_YYYYMMDD.*
          Creates fresh blank files with 35-col headers. No-op if already migrated.

Step 2  get_cached_odds() + full inference engine runs

Step 3  log_predictions() → generate_excel_log() → update_google_sheets()
          sheet.batch_clear(["A2:AJ2000"])   # values only, formatting intact
          sheet.update("A1", [headers] + rows)
          sheet.spreadsheet.batch_update(color_requests)
```

### archive_existing_prediction_logs()
```python
# Migration trigger: 'model_strategy' absent from CSV column headers
# Archive destination: predictions/archive/predictions_log_archived_YYYYMMDD.*
# Post-archive: creates blank CSV + XLSX with full 35-col header row
# Idempotent: skips on every run after first migration
```

### update_google_sheets() — Value-Only Clear
```python
# OLD: sheet.clear()                      ← wiped all values from row 1 down
# NEW: sheet.batch_clear(["A2:AJ2000"])   ← clears values rows 2-2000 only
#   Preserves: Row 1 headers, cell borders, data validation dropdowns,
#              conditional formatting rules (not touched by values endpoint)
#   Range A2:AJ2000: 36 cols covers 35-col schema + 1 buffer col
#   After clear: sheet.update("A1", [headers] + rows) rewrites everything
```

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

## Financial Simulation

### de_vig_probs(home_ml, away_ml)
```python
# Strips sportsbook overround so market probs sum to exactly 1.0.
# FanDuel runs ~104–105% overround; raw implied probs inflate market estimates
# by ~2–3%, which deflates calculated edge by the same amount.
raw_h, raw_a = american_to_prob(home_ml), american_to_prob(away_ml)
total = raw_h + raw_a
return raw_h / total, raw_a / total
# Used in: run_backtest_with_params() edge calc, walk_forward_financial_sim()
# NOT used for payout math — books pay raw American odds.
```

### walk_forward_financial_sim(C, flat_bet, edge_min, all_feat=None)
Same 4-fold expanding-window structure. Per fold:
- Trains logistic regression on training years only
- Generates out-of-sample `p_home` for test year
- Edge = `model_prob − de_vigged_market_prob`
- Bets any side where `edge > edge_min` (no upper cap)
- Flat bet per game, payout via raw American odds

Pass `all_feat` to reuse pre-collected features across threshold sweeps.

### RUN_MODE = 'financial_sim'
```python
# Dispatcher collects features once, then sweeps thresholds:
_feat_cache = collect_game_features_for_meta([2021, 2022, 2023, 2024, 2025])
for _thresh in [...]:
    walk_forward_financial_sim(C=1.0, flat_bet=20.0, edge_min=_thresh, all_feat=_feat_cache)
```

---

## Market-Residual Meta Model (current architecture)

### Feature Set — _META_FEATURES (5 features)
```
[0] logit_market_prob  — de-vigged FD home win prob, logit-transformed
                         Clipped [0.001, 0.999] before logit.
                         UNSCALED — col 0 passed raw through train and test.
[1] d_woba             — home_woba_r − away_woba_r
[2] d_xwoba_bat        — home_xwoba_bat_r − away_xwoba_bat_r
[3] d_xfip             — home_xfip_r − away_xfip_r  (inverted)
[4] d_xwoba_pit        — home_xwoba_pit_r − away_xwoba_pit_r  (inverted)
```

Standardization: **columns 1–4 only**, fit on training fold.
Column 0 is left unscaled in both `walk_forward_meta_model` and
`walk_forward_financial_sim`.

`collect_game_features_for_meta()` now requires both `fd_home_ml` and
`fd_away_ml` — games without odds are excluded (10,181 games, down from 10,846).

### Key Findings

Original meta model (no market logit, 4 features):
- Win rate locked at ~40% regardless of edge threshold or regularization (C)
- C=0.1 vs C=1.0: dropped only 29 bets across 4 years — feature gaps too
  wide for regularization to meaningfully reduce volume
- ROI improved at higher thresholds (>12% → +6.1%) but via underdog payout
  structure, NOT improved win rate. Structural problem, not a filtering problem.

Market-residual model (logit_market_prob added as feature 0):
- Win rate immediately stabilized at ~49.7% — market anchor eliminates
  underdog bias
- Volume collapses fold-over-fold as training data grows: the regression
  learns the market logit explains most variance and shrinks Statcast
  residuals toward zero by folds 3–4
- ~49.7% win rate is below the ~52.4% break-even for standard −110 juice
- Root cause: team-level season-to-date Statcast metrics carry no residual
  information beyond what the sportsbook has already priced in

### Threshold Sweep Results (market-residual, C=1.0, flat $20)

| Edge Filter | Bets | Win%  | Profit   | ROI    |
|-------------|------|-------|----------|--------|
| > 3%        | 1,575 | 49.8% | −$1,924 | −6.1%  |
| > 4%        |   891 | 49.7% |   −$899 | −5.0%  |
| > 5%        |   549 | 49.7% |   −$475 | −4.3%  |

---

## Pending Items

1. **Round 9 financial sim results**: `player_level_meta` pipeline is built and running.
   Evaluate Sniper vs Enforcer results once backtest completes. Key question: does
   player-level + handedness-split + box score lineup carry residual signal vs market?

2. **Platoon split retrospective** (3 weeks): check if confirmed-lineup bets where
   platoon boosted the model outperform those where it lowered the model.
   Accuracy report section activates automatically at 20+ confirmed-lineup settled bets.

3. **Full 2025 ML odds**: if The Odds API releases historical data or SportsGameOdds
   adds snapshot capability, fill Aug 17–Oct 1, 2025 ML gap.

4. **Weather / travel / rest factors**: not yet implemented. These are candidates
   for residual signal the market may not fully price at line-posting time.

5. **Exponential recency weighting**: within the rolling 100 PA window, weight
   recent games more heavily. Currently all games in window are equal-weighted.

---

## Round 9 — Player-Level Feature System

### Why team-level failed (root cause, Section 18)
Team-level season-to-date Statcast metrics are too coarse and too public.
Win rate locked at ~49.7% even with market logit anchor — below 52.4% break-even.
The 4 features (wOBA, xwOBA bat/pit, xFIP) carry no residual info beyond the line.

### Architecture change
```
BEFORE: team wOBA = cumulative season-to-date from MLB API game logs
AFTER:  team wOBA = PA-weighted avg of each batter's rolling 100 PA vs L or R

BEFORE: team xFIP = season-to-date across all team pitchers  
AFTER:  xFIP = actual starting pitcher's rolling 100 BF stats
```

### aggregate_lineup_metric(player_stats_dict, lineup_list, lg_avg)
```python
PA_WEIGHTS = np.array([0.123, 0.120, 0.117, 0.114, 0.111, 0.108, 0.105, 0.102, 0.100])
# Positional PA weights: leadoff ~123%, cleanup ~111%, 9-hole ~100%
# player_stats_dict: {player_id: metric_value}  — e.g. rolling xwOBA vs RHP
# lineup_list: [pid1..pid9] in batting order from box score
# lg_avg: league-average fallback for rookies / missing players
# Returns: np.dot(metrics, PA_WEIGHTS)
```

### Rolling cache construction
```python
# Batter cache: (player_id, hand, date) → {woba, xwoba, pa}
# Minimum: 30 PA. Window: most recent 100 PA vs that pitcher hand type.
# Built from player_data/batters_vs_L_YYYY.csv + batters_vs_R_YYYY.csv

# Pitcher cache: (pitcher_id, date) → {xfip, xwoba_pit, pa}
# Minimum: 20 BF. Window: most recent 100 BF.
# IP source: game_lineups.json inningsPitched (exact, not estimated)
# xFIP: (13*exp_HR + 3*(BB+HBP) - 2*K) / IP + cFIP
#   exp_HR = lg_hr_fb_rate * (IP * 3 * 0.40)   [fly balls estimated from IP]
# Built from player_data/starters_YYYY.csv

# Both use prefix sums + binary search: O(log n) per target date lookup
```

### IP data pipeline
```python
# inningsPitched stored in:
#   bs[side]['players']['ID{pid}']['stats']['pitching']['inningsPitched']
# Returns string e.g. '6.1' (6⅓ IP), '5.2' (5⅔ IP)
# ip_to_float('5.2') = 5.667  (exact — baseball notation where .1=1/3 not 0.1)
# 'outs' field is NOT in the MLB API boxscore pitching stats block
# _build_ip_lookup(): reads game_lineups.json → {(player_id, game_pk): ip_decimal}
# load_player_pitcher_data() merges true IP via (player_id, game_pk) join
```

### Game lineup cache (game_lineups.json)
```python
# One-time pull: ~2-3 hours for 2021-2025 (~12,150 boxscore API calls)
# Checkpoint saves every 200 games — resumes automatically on restart
# Per-game entry:
{
  'date': 'YYYY-MM-DD',
  'home_fg': 'NYY',  'away_fg': 'BOS',
  'home_lineup': [pid1..pid9],   # battingOrder '100'..'900' = starters
  'away_lineup': [pid1..pid9],
  'home_starter_id': int, 'away_starter_id': int,
  'home_starter_hand': 'R', 'away_starter_hand': 'L',
  'home_starter_ip': '6.1', 'away_starter_ip': '5.0'  # from stats block
}
# Retry: 5x for schedule call, 3x for boxscore calls (503 transient errors)
# game_type == 'R' filter applied to exclude spring training / playoffs
```

### fetch_player_data.py
```bash
python fetch_player_data.py --test     # prints all 78 column names, no files saved
python fetch_player_data.py            # full pull 2021-2025 (15 files, ~15 min)
python fetch_player_data.py --seasons 2024 2025   # specific seasons only
```
Key lessons:
- Do NOT include `type=details` in params — returns 0 rows with `group_by=name-date`
- Monthly date-range pagination required (Savant caps at 10,000 rows per request)
- `ip` is NOT in the Savant grouped export — use game_lineups.json instead
- `xwoba` column is present and correct (not the long-form estimated_ name)

---

## Dual Strategy Simulation

### walk_forward_dual_strategy(all_feat, flat_bet=20.0)
```python
from sklearn.linear_model import LogisticRegression

STRATEGIES = [
    ('The Sniper',   C=1.0, edge_min=0.050),   # high-conviction, low volume
    ('The Enforcer', C=5.0, edge_min=0.035),   # relaxed reg, higher volume
]
# Same 4-fold walk-forward structure (train [2021]→test 2022, etc.)
# Same standardization: col 0 (market logit) unscaled, cols 1-4 scaled on train only
# Both models trained per fold; output: per-fold rows + side-by-side summary table
# predict_proba(X_te_s)[:, 1] = P(home win)
# Edge = model_prob - de_vigged_market_prob
```

Replaces old threshold sweep `[0.03, 0.04, 0.05]` in `player_level_meta` mode.

### RUN_MODE options (current)
```
'player_level_meta'  — full Round 9 pipeline (lineups → CSVs → caches → dual sim)
'dual_strategy'      — dual sim only, uses team-level meta features (fast)
'financial_sim'      — old threshold sweep on team-level features (C=1.0)
'meta_model'         — Brier score calibration walk-forward (no ROI)
'test_ip_cache'      — verify inningsPitched extraction, 10 sample games
'train'              — ROI-optimized random search (round 8)
'individual_metrics' — test each metric in isolation
'walk_forward'       — 4-fold walk-forward with fresh param search per fold
'calibration'        — Brier-score-optimized parameter search
```
