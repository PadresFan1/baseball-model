# Baseball Model — Reference Guide

## Running the Model

```bash
# Daily predictions (run at 9am, 12pm, 4pm)
python model.py

# Backtest parameter search
python backtest.py

# Rebuild player→team mapping (only needed when adding new RP data)
python build_player_map.py
```

---

## File Structure

```
baseball-model/
├── model.py                    # Live daily prediction model
├── backtest.py                 # Parameter optimization via random search
├── build_player_map.py         # Builds player ID → team mapping from MLB API
├── scraper.py                  # Data scraper
│
├── Statcast/
│   ├── batters_YYYY.csv        # Team-level batter statcast (2021-2025)
│   └── pitchers_YYYY.csv       # Team-level pitcher statcast (2021-2025)
│
├── rp/
│   └── *.csv                   # Relief pitcher statcast by division/year
│
├── constants/
│   ├── woba_fip_constants.csv  # Season wOBA weights and FIP constant
│   ├── park_factors.csv        # Park factors by team
│   └── park_factors_handedness.csv
│
├── historical_data/
│   ├── historical_game_logs.json    # Cached MLB API game logs (2021-2024)
│   ├── player_team_map.json         # Player ID → team by season
│   ├── statcast_cache.json          # Preprocessed statcast data
│   ├── bullpen_cache.json           # Preprocessed bullpen data
│   └── search_*.csv                 # Backtest run results (one per run)
│
├── offense_YYYY.csv            # FanGraphs team offense stats
├── pitching_YYYY.csv           # FanGraphs team pitching stats
├── mlb_odds_dataset.json       # Historical odds (2021-2025)
├── predictions_log.csv         # Daily predictions log
├── predictions_log.xlsx        # Color-coded Excel version
└── .env                        # API keys and credentials (never commit)
```

---

## Environment Variables (.env)

```
API_KEY=                        # The Odds API key
GOOGLE_CREDENTIALS_PATH=        # Path to google_credentials.json
EMAIL_SENDER=                   # Gmail address used to send reports
EMAIL_APP_PASSWORD=             # Gmail app password (no spaces, 16 chars)
EMAIL_RECIPIENT=                # Address to receive reports
```

---

## model.py — How It Works

### Data Flow
1. Loads odds from API (cached 2.5 hrs) → filters to today's upcoming games
2. Loads team stats (cached 4 hrs): rolling runs, wOBA, FIP, xFIP, K%, BB%,
   xwOBA, BABIP, barrel% from statcast
3. Gets today's probable starters → fetches their season RA9/IP
4. For each game: blends starter + bullpen ratings → runs 5,000 Poisson sims
5. Compares win probability to market odds → flags edges > threshold
6. Logs predictions to CSV/Excel/Google Sheets
7. Emails full output

### Key Constants (adjust after backtesting)
```python
OFFENSE_WEIGHTS = {
    'rolling': 1/5, 'woba': 1/5, 'xwoba': 1/5, 'k_pct': 1/5, 'bb_pct': 1/5
}
PITCHING_WEIGHTS = {
    'rolling': 1/5, 'fip': 1/5, 'xfip': 1/5, 'k_pct': 1/5, 'bb_pct': 1/5
}
NUM_SIMULATIONS = 5000
STARTER_INNINGS = 5       # Assumed starter IP for blending
MIN_INNINGS = 15          # Min IP before using starter's RA9
```

Edge threshold (moneyline): hardcoded at `0.03` in the game loop — update
this to the optimal `ml_edge_min` from backtest results.

### Output Sections
- `=== TODAY'S EDGES ===` — games where model finds betting value
- `=== NO EDGE FOUND ===` — all other games with projections
- `=== AWAITING STARTERS ===` — games skipped (starter not announced)
- `=== IN PROGRESS ===` — live scores from MLB API
- `=== COMPLETED ===` — final scores
- `=== ACCURACY REPORT ===` — running W-L, ROI, edge tier breakdown

---

## backtest.py — How It Works

### Architecture
```
Module load:
  ├── Load historical game logs (JSON cache)
  ├── Load statcast data (JSON cache)
  ├── Load bullpen data (JSON cache)
  └── precompute_all_rolling() ← one-time, reused across all trials

For each trial (random_search):
  └── run_backtest_with_params(params)
        ├── For each season (2021-2024):
        │     ├── precompute_statcast()  ← per season
        │     ├── precompute_bullpen()   ← per season
        │     └── Game loop (O(1) lookups only)
        └── Calculate W-L, win rate, ROI, profit
```

### Param Space (current — round 3)
| Parameter | Options | Notes |
|-----------|---------|-------|
| ml_edge_min | 0.06, 0.07 | Minimum edge to bet |
| ml_edge_max | 0.08, 0.10 | Maximum edge to bet |
| favorites_only | True, False | Only bet on favorites |
| rolling_weight_7 | 0.5–0.8 | Weight for 7-day vs 15-day rolling |
| rating_cap_low | 0.65, 0.70 | Min rating (prevents extreme values) |
| rating_cap_high | 1.30, 1.50 | Max rating |
| w_rolling_off | 0.05–0.10 | Offense: recent rolling runs weight |
| w_woba | 0.20–0.40 | Offense: wOBA weight |
| w_xwoba_bat | 0.10–0.30 | Offense: statcast xwOBA weight |
| w_babip_bat | 0.10–0.30 | Offense: BABIP weight |
| w_barrel_bat | 0.10–0.30 | Offense: barrel% weight |
| w_rolling_pit | 0.05–0.10 | Pitching: recent rolling RA weight |
| w_fip | 0.10–0.30 | Pitching: FIP weight |
| w_xfip | 0.10–0.30 | Pitching: xFIP weight |
| w_k_pit | 0.10–0.30 | Pitching: K% weight |
| w_bb_pit | 0.20–0.40 | Pitching: BB% weight |
| w_xwoba_pit | 0.10–0.30 | Pitching: statcast xwOBA against |
| w_babip_pit | 0.10–0.30 | Pitching: BABIP against |
| w_barrel_pit | 0.10–0.30 | Pitching: barrel% against |
| w_bullpen_qual | 0.10–0.30 | Bullpen quality (xwOBA/K%/BB%) |
| w_bullpen_fat | 0.05–0.15 | Bullpen fatigue (pitches last 7 days) |

### Running the Backtest
```python
# In backtest.py — bottom of file
N_RUNS = 5   # Number of consecutive 100-trial runs

# To just run evaluate_runs() without new trials:
N_RUNS = 0
```

### Output Files
- `historical_data/search_YYYY-MM-DD_seed{N}_100trials.csv` — one per run
- `historical_data/evaluation_YYYY-MM-DD_HH-MM_{N}runs.csv` — combined

### Performance
- Precompute (one-time): ~5 seconds for all 4 seasons
- Per trial: ~7.4 seconds (was ~62 seconds before optimization)
- 100 trials: ~12 minutes
- 5 runs: ~60 minutes

---

## Team Abbreviation Maps

### MLB API → FanGraphs (TEAM_MAP in model.py)
| MLB API | FanGraphs |
|---------|-----------|
| AZ | ARI |
| CWS | CHW |
| KC | KCR |
| SD | SDP |
| SF | SFG |
| TB | TBR |
| WSH | WSN |

### Odds API → FanGraphs (ODDS_TO_FG in backtest.py)
| Odds API | FanGraphs |
|----------|-----------|
| KC | KCR |
| SD | SDP |
| SF | SFG |
| TB | TBR |
| WAS | WSN |
| OAK | ATH |

---

## Data Sources

| Data | Source | Update Frequency |
|------|--------|-----------------|
| Live odds | The Odds API | Cached 2.5 hrs |
| Team stats (live) | MLB Stats API | Cached 4 hrs |
| Statcast batting | Baseball Savant CSV export | Manually, per season |
| Statcast pitching | Baseball Savant CSV export | Manually, per season |
| Relief pitcher data | Baseball Savant CSV export | Manually, per season |
| FanGraphs offense/pitching | FanGraphs CSV export | Manually, per season |
| wOBA/FIP constants | FanGraphs | Manually, per season |
| Park factors | FanGraphs | Manually, per season |
| Historical odds | mlb_odds_dataset.json | Static (2021-2025) |
| Historical game logs | MLB Stats API (cached) | Static (2021-2025) |

---

## Key Findings from Backtesting (as of May 2026)

### Locked In
- `ml_edge_min = 0.07` — best single value across 2,700+ trials
- `ml_edge_max = 0.08` — tight window strongly outperforms wide
- `favorites_only = True` — slight edge, consistent signal

### Strong Positive Signals (keep weights high)
- `w_bb_pit` (+0.277 correlation with ROI) — bullpen/pitcher walk rate
- `w_woba` (+0.252) — batting wOBA

### Strong Negative Signals (keep weights low)
- `w_rolling_off` (-0.183) — recent rolling offense
- `w_rolling_pit` (-0.175) — recent rolling pitching RA

### Statcast Impact
Adding statcast raised median trial ROI from -2.7% to -0.7%, and best
single-trial ROI from ~6% to 21.8%.

### Bullpen
Being evaluated in current runs. Signal TBD.

---

## Planned Features (Not Yet Built)

### Platoon Splits
- Batter performance vs LHP/RHP, pitcher performance vs LHB/RHB
- Needs separate Baseball Savant exports filtered by handedness
- Live model: straightforward once starter announced + lineup posted
- Backtest: needs historical lineup composition — significant data work

### Bullpen in Live Model
- Currently only in backtest. Wire into model.py after backtest confirms signal.
- Data available same-day via MLB Stats API (already used for rolling averages)

---

## Backtest Workflow (Repeating Cycle)

```
1. Run N_RUNS = 5-10
2. Set N_RUNS = 0, run python backtest.py → evaluate_runs() only
3. Read parameter breakdown + weight correlations
4. Narrow param_space: remove underperforming values, shift weight ranges
5. Increment round comment in param_space
6. Repeat until correlations stabilize and top params are consistent
7. Take top parameter combination → update model.py constants
```

### How to Narrow Params
- **Keep**: values that appear most in top-25% of trials with highest avg ROI
- **Remove**: values with consistently negative avg ROI
- **Shift weight ranges up**: params with strong positive correlation
- **Shift weight ranges down**: params with strong negative correlation
- **Target**: 500–1000 trials per round for reliable univariate signal
