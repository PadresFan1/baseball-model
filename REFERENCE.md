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
│
├── Statcast/
│   ├── batters_YYYY.csv        # Team-level batter statcast by game (2021-2025)
│   └── pitchers_YYYY.csv       # Team-level pitcher statcast by game (2021-2025)
│   # Columns: player_name(team), game_date, babip, xwoba, barrels_per_pa_percent, pa
│   # Date format: batters=YYYY-MM-DD, pitchers=M/D/YYYY (both normalized on load)
│
├── rp/
│   └── *.csv                   # Relief pitcher statcast by division/year (80,312 rows)
│   # Columns: player_id, game_date, game_pk, total_pitches, babip, woba, xwoba,
│   #          k_percent, bb_percent, barrels_per_pa_percent, pa
│   # NO team column — mapped via player_team_map.json
│
├── constants/
│   ├── woba_fip_constants.csv  # Season wOBA weights + FIP constant (col: Season, wBB,
│   │                           # wHBP, w1B, w2B, w3B, wHR, cFIP)
│   ├── park_factors.csv
│   └── park_factors_handedness.csv
│
├── historical_data/
│   ├── historical_game_logs.json    # MLB API game logs 2021-2024 (cached)
│   ├── player_team_map.json         # {season: {player_id: fg_abbrev}} for 2021-2025
│   ├── statcast_cache.json          # Preprocessed statcast data
│   ├── bullpen_cache.json           # Preprocessed bullpen data (aggregated to team-game)
│   ├── mlb_odds_dataset.json        # Historical odds 2021-2025 (proprietary)
│   ├── search_*_r{N}.csv            # Backtest run results (one per run, round tagged)
│   └── evaluation_*.csv             # Combined evaluation files
│
├── offense_YYYY.csv            # FanGraphs team offense stats
├── pitching_YYYY.csv           # FanGraphs team pitching stats
├── predictions_log.csv         # Daily predictions log (all runs)
├── predictions_log.xlsx        # Color-coded Excel (green=WIN, red=LOSS, yellow=flag)
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
```

**Gmail app password setup:** myaccount.google.com/apppasswords
Requires 2-step verification enabled on the sending account first.

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
1. Load odds from API (cached 2.5 hrs) → filter to today's upcoming games
2. Load team stats (cached 4 hrs):
   - Rolling runs scored/allowed (last 7 and 15 games) via MLB Stats API gamelog
   - wOBA, FIP, xFIP, K%, BB% cumulative season via MLB Stats API
   - xwOBA, BABIP, barrel% from Baseball Savant (live API call, player-level)
3. Get today's probable starters → fetch season RA9 and innings pitched
4. For each upcoming game:
   - Blend rolling + season stats + starter quality into ratings
   - Run 5,000 Poisson simulations → win probability + projected total
   - Compare to FanDuel/DraftKings market odds → calculate edge
5. Separate games into: edge found, no edge, awaiting starters
6. Log predictions to CSV/Excel/Google Sheets
7. Email full output (edges, scores, no startup/accuracy noise)

### Key Constants (UPDATE THESE after backtest params locked)
```python
OFFENSE_WEIGHTS = {
    'rolling': 1/5, 'woba': 1/5, 'xwoba': 1/5, 'k_pct': 1/5, 'bb_pct': 1/5
}
PITCHING_WEIGHTS = {
    'rolling': 1/5, 'fip': 1/5, 'xfip': 1/5, 'k_pct': 1/5, 'bb_pct': 1/5
}
NUM_SIMULATIONS = 5000
STARTER_INNINGS = 5       # Assumed starter IP for starter/bullpen blend
MIN_INNINGS = 15          # Min IP before using starter's RA9
MAX_RA9 = 7.00            # Cap on starter RA9 used in model
```

Edge threshold: hardcoded at `0.03` in game loop (~line 1219).
Update to `ml_edge_min` value from backtest when params finalized.

### Email System
- `_Tee` class at top of file captures all stdout to buffer simultaneously
- Buffer reset just before "Upcoming games today" — excludes startup noise
- Email body captured BEFORE print_accuracy_report() — accuracy not in email
- Google Sheets message bypasses buffer via `sys.stdout._real.write()`
- Subject: "Baseball Model — {9am/12pm/4pm} Run — {date}"
- From display: "Betting Model <sender>"

### Output Sections (terminal + email)
```
=== TODAY'S EDGES ===       ← games with model edge > threshold
=== NO EDGE FOUND ===       ← all other games with projections
=== AWAITING STARTERS ===   ← skipped (starter not announced)
=== IN PROGRESS ===         ← live scores from MLB API
=== COMPLETED ===           ← final scores
```

Terminal only (not in email):
```
=== ACCURACY REPORT ===     ← overall W-L, ROI, edge tier breakdown
```

---

## backtest.py — Architecture

### Module Load Sequence (happens once, before any trials)
```
1. Load game logs from JSON cache
2. Load statcast data (statcast_cache.json or CSVs)
3. Load bullpen data (bullpen_cache.json or rp/*.csv + player_team_map.json)
4. precompute_all_rolling() ← CRITICAL PERFORMANCE FUNCTION
   - Sweeps each team's game logs ONCE per season using prefix sums
   - O(games + target_dates) per team vs O(games × target_dates) previously
   - Stores rolling_team_cache[(season, team, date)] and rolling_lg_cache[(season, date)]
   - 8.3x speedup — reduced from ~102 min/100 trials to ~12 min/100 trials
5. Load odds dataset, build odds_df DataFrame
```

### Per-Trial Flow (random_search → run_backtest_with_params)
```
For each season (2021-2024):
  ├── precompute_statcast(season, dates) → sc_lookup, sc_lg
  ├── precompute_bullpen(season, dates) → bp_lookup, bp_lg
  └── For each game in season_odds:
        ├── O(1) lookup: rolling_team_cache[(season, team, date)]
        ├── O(1) lookup: rolling_lg_cache[(season, date)]
        ├── O(1) lookup: sc_lookup[(team, side, date)]
        ├── O(1) lookup: bp_lookup[(team, date)]
        ├── Compute offense/pitching ratings (weighted sum, normalized)
        ├── home_lambda = (home_off / away_pit) * lg_avg_runs
        ├── away_lambda = (away_off / home_pit) * lg_avg_runs
        ├── run_simulation() → win probabilities
        └── Check edge vs FanDuel line → record WIN/LOSS
```

### Current Param Space (Round 5)
```python
PARAM_ROUND = 5
# History:
# R1: base params, no statcast
# R2: statcast added (xwoba_bat, babip_bat, barrel_bat/pit)
# R3: bullpen added (bullpen_qual, bullpen_fat) — hurt model
# R4: bullpen removed, favorites_only locked False, 53% profitable
# R5: locked edge/cap params, barrel_bat dropped, weights rebalanced

param_space = {
    # LOCKED (don't change — confirmed across 6,000+ trials)
    'ml_edge_min':    [0.07],
    'ml_edge_max':    [0.08],
    'favorites_only': [False],
    'rating_cap_low': [0.65],
    'rating_cap_high':[1.50],

    # STILL SEARCHING
    'rolling_weight_7': [0.50, 0.60, 0.70, 0.80],

    # Offense (barrel_bat removed — -0.298 correlation in r4)
    'w_rolling_off': [0.05, 0.08, 0.10],
    'w_woba':        [0.20, 0.30, 0.40],
    'w_xwoba_bat':   [0.10, 0.20, 0.30],
    'w_babip_bat':   [0.15, 0.25, 0.35],  # raised r5 (+0.115 in r4)

    # Pitching
    'w_rolling_pit': [0.05, 0.08, 0.10],
    'w_fip':         [0.10, 0.20, 0.30],  # pulled back (flipped negative r4)
    'w_xfip':        [0.10, 0.20, 0.30],
    'w_k_pit':       [0.10, 0.20, 0.30],
    'w_bb_pit':      [0.15, 0.25, 0.35],  # pulled back (over-weighted r4)
    'w_xwoba_pit':   [0.10, 0.20, 0.30],
    'w_babip_pit':   [0.05, 0.08, 0.12],  # lowered (consistently negative)
    'w_barrel_pit':  [0.15, 0.25, 0.35],  # raised (+0.136 in r4)
}
```

### Normalization
```python
off_keys = ['w_rolling_off', 'w_woba', 'w_xwoba_bat', 'w_babip_bat']
pit_keys = ['w_rolling_pit', 'w_fip', 'w_xfip', 'w_k_pit', 'w_bb_pit',
            'w_xwoba_pit', 'w_babip_pit', 'w_barrel_pit']
# Each group summed then each weight divided by total → sums to 1.0
```

### Rating Formulas
```python
# Offense (higher = better team)
home_off = (rolling_off_r * w_rolling_off + woba_r * w_woba +
            xwoba_bat_r * w_xwoba_bat + babip_bat_r * w_babip_bat)

# Pitching (higher = better team, all ratios inverted where lower stat = better)
home_pit = (rolling_pit_r * w_rolling_pit + fip_r * w_fip + xfip_r * w_xfip +
            k_pit_r * w_k_pit + bb_pit_r * w_bb_pit +
            xwoba_pit_r * w_xwoba_pit + babip_pit_r * w_babip_pit +
            barrel_pit_r * w_barrel_pit)

# Batting ratios: team/league (higher = better)
# Pitching ratios: league/team (inverted — lower stat = better pitcher)
# Exception: k_pit = team/league (higher K% = better pitcher)

# Lambda calculation
home_lambda = (home_off / away_pit) * lg_avg_runs
away_lambda = (away_off / home_pit) * lg_avg_runs
```

### File Output
```
historical_data/search_YYYY-MM-DD_seed{N}_100trials_r{round}.csv
  Columns: all param values + roi, win_rate, n_bets, profit,
           run_date, seed, n_trials, run_time_mins, param_round

historical_data/evaluation_YYYY-MM-DD_HH-MM_{N}runs.csv
  All trials combined, sorted by ROI descending
```

### Running the Backtest
```python
# Bottom of backtest.py:
N_RUNS = 10   # ~2 hours at 12 min/run

# To only run evaluate_runs() without new trials:
N_RUNS = 0
```

### Performance Numbers
```
Module load (one-time precompute): ~5 seconds for all 4 seasons
Per trial:  ~7.4 seconds  (was ~62 seconds before precompute_all_rolling)
100 trials: ~12 minutes   (was ~102 minutes)
10 runs:    ~2 hours      (was ~17 hours)
```

---

## Backtest Findings Summary (as of Round 4 — 6,000 trials)

### Locked Parameters
| Parameter | Value | Confidence |
|-----------|-------|-----------|
| ml_edge_min | 0.07 | Very high — consistent across all rounds |
| ml_edge_max | 0.08 | Very high — +4.98% avg ROI in top 25% |
| favorites_only | False | High — True was small-sample artifact |
| rating_cap_low | 0.65 | Medium-high |
| rating_cap_high | 1.50 | Medium-high — in 8/10 top r4 trials |

### Weight Signals (across all rounds)
| Weight | Signal | Action |
|--------|--------|--------|
| w_bb_pit | +0.281 overall, -0.042 in r4 | Keep, narrower range |
| w_woba | +0.181 | Keep high |
| w_barrel_pit | +0.136 in r4 | Raise range |
| w_babip_bat | +0.115 in r4 | Raise range |
| w_barrel_bat | -0.298 in r4 | **Removed entirely** |
| w_bullpen_fat | -0.246 | **Removed entirely** |
| w_rolling_off | -0.050 | Keep low |
| w_babip_pit | -0.085 | Keep low |

### Sample Size Warning
```
At n=90 bets, 95% CI on win rate spans 21 percentage points.
Always filter to n_bets >= 300 before drawing conclusions.

n >= 0:   top ROI +21.8%, True,  win 67%, n=100  ← NOISE
n >= 300: top ROI +14.7%, False, win 50%, n=375  ← SIGNAL
n >= 500: top ROI +11.3%, False, win 50%, n=863  ← SIGNAL
```

### Round-by-Round Profitability
| Round | Trials | Median ROI | % Profitable |
|-------|--------|------------|-------------|
| R1 (base) | 1,000 | -1.36% | 38.6% |
| R2 (statcast) | 2,000 | -0.92% | 38.5% |
| R3 (bullpen) | 2,000 | -2.55% | 23.5% |
| R4 (cleaned) | 1,000 | **+0.24%** | **53.0%** |

---

## Data Sources Reference

| Data | Source | Format | Update |
|------|--------|--------|--------|
| Live odds | The Odds API | JSON API | Cached 2.5 hrs |
| Team rolling stats | MLB Stats API | gamelog endpoint | Cached 4 hrs |
| Statcast batting | Baseball Savant export | CSV by year | Manual/season |
| Statcast pitching | Baseball Savant export | CSV by year | Manual/season |
| Relief pitcher data | Baseball Savant export | CSV by division/year | Manual/season |
| FanGraphs offense | FanGraphs team stats | CSV by year | Manual/season |
| FanGraphs pitching | FanGraphs team stats | CSV by year | Manual/season |
| wOBA/FIP constants | FanGraphs leaderboards | CSV | Manual/season |
| Park factors | FanGraphs | CSV | Manual/season |
| Historical odds | mlb_odds_dataset.json | Local JSON | Static 2021-2025 |
| Historical game logs | MLB Stats API | Cached JSON | Static 2021-2024 |

---

## Backtest Optimization Workflow

```
1. Set N_RUNS (10 recommended), run python backtest.py
2. Set N_RUNS = 0, run python backtest.py → evaluate_runs() only
3. Review output — focus on:
   a. Round summary: is median ROI improving? % profitable?
   b. Top 10 at n_bets >= 300 (not the small-sample top 10)
   c. Weight correlations: which are positive/negative?
   d. Discrete param breakdown: which values dominate top 25%?
4. Narrow param_space:
   - Lock discrete params that are consistent across rounds
   - Remove weights with strong negative correlation
   - Raise ranges for positive correlators
   - Lower ranges for negative correlators
5. Increment PARAM_ROUND, add comment explaining changes
6. Repeat from step 1
7. When weight correlations stabilize and round-over-round changes
   are minimal → parameters are ready to plug into model.py
```

### When to Stop Backtesting
- Same parameter values win across 3+ consecutive rounds
- Weight correlations consistent across rounds (not flipping)
- 53%+ profitable trials (current best: r4 at 53%)
- Top n>=300 trials show ROI 10%+ consistently

---

## Planned Features

### Platoon Splits (next after params locked)
- Batter performance vs LHP/RHP, pitcher vs LHB/RHB
- Needs: separate Baseball Savant exports by handedness
- Live model: check pitcher handedness (from MLB API) → use opponent's
  relevant batting split → adjust lambda
- Backtest: needs historical lineup L/R composition — significant work

### Bullpen in Live Model
- Removed from backtest (fatigue metric harmful, quality neutral)
- May revisit after parameters locked
- Would use existing get_pitching_game_logs() in model.py
- Compute season-to-date xwOBA against for each team's bullpen

### Recurring Schedule
- Currently run manually or via one-time Task Scheduler entries
- Task Scheduler command:
  ```powershell
  $action = New-ScheduledTaskAction -Execute "python" `
    -Argument "c:\Users\super\baseball-model\backtest.py" `
    -WorkingDirectory "c:\Users\super\baseball-model"
  $trigger = New-ScheduledTaskTrigger -Once -At "05:00AM"
  $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 5)
  Register-ScheduledTask -TaskName "BaseballBacktest_R5" -Action $action `
    -Trigger $trigger -Settings $settings -RunLevel Highest
  ```

---

## Python Environment Note

After a restart, if you get "No module named pandas":
- `python` may have switched to a different interpreter
- Check with: `python -c "import sys; print(sys.executable)"`
- Fix by installing in whichever python is active:
  ```
  python -m pip install pandas numpy requests MLB-StatsAPI python-dotenv
                        openpyxl gspread google-auth
  ```
- Package name is `MLB-StatsAPI` not `python-statsapi`
- PATH warnings about Scripts directory are harmless
