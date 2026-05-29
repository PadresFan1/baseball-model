# MLB Betting Model

A player-driven MLB moneyline betting model that identifies edges against opening lines using Log5 win probability, live player snapshots, and Kelly Criterion staking.

---

## How It Works

The model runs on a 5-window daily schedule (6 PM, 11 PM, 5 AM, 11 AM, 3 PM Mountain Time) and targets **opening line inefficiencies** — the window between when lines are posted and when confirmed lineups arrive.

### Win Probability Engine

- **Log5 logistic regression** trained on 9,346 games (2021–2025) using player-level features only — no market anchor
- Features: `d_log5_woba`, `d_log5_xwoba`, `d_xfip` (differential Log5 matchup stats between lineups)
- Each batter-pitcher matchup is computed individually via the Log5 formula, then aggregated per team
- Model predicts against **opening lines** (posted the night before); closing line movement is treated as confirmation of edge, not a reason to re-evaluate
- Platoon splits applied when confirmed lineups are available (wOBA vs LHP / vs RHP per batter)
- Run totals computed separately via Poisson simulation (display only; not used for betting decisions)

### Origination Logic

Bets lock at the first run that finds an edge and are **carried forward** on subsequent runs unless a fundamental event occurs:
- **Pitcher change** → recompute win probability; carry forward if same or better, re-evaluate odds if worse
- **Lineup confirmed** → same logic; show `📉` detail line if win probability drops and recommendation changes
- **Line movement alone** → never triggers re-evaluation (closing line value confirms the original signal)

### Staking

Half-Kelly sizing with a 15% bankroll cap per bet. Effective bankroll = total bankroll minus all open pending stakes.

### Edge Thresholds

| Tier | Min Edge | Label |
|------|----------|-------|
| Sniper | > 4.5% | `[ACTION: SNIPER BET]` |
| Enforcer | > 4.0% | `[ACTION: ENFORCER BET]` |

### 2026 Backtest Results (638 games, Mar 29 – May 24, vs SBR opening lines)

| Strategy | Bets | Win Rate | ROI |
|----------|------|----------|-----|
| Sniper (edge > 4.5%) | 296 | 48.0% | +9.70% |
| Enforcer (edge > 4.0%) | 329 | 47.1% | +8.81% |

Positive ROI in all three months (March, April, May).

---

## Repository Structure

```
model.py                  # Main production model — runs on scheduler
backtest.py               # Backtesting, parameter search, model training
fetch_player_data.py      # Pulls Statcast CSVs and builds game lineup cache

constants/                # Park factors, wOBA/FIP constants by season
historical_data/          # Scraped odds CSVs (2021–2026), cleaned game logs
season_stats/             # Team offense/pitching stats by season (MLB Stats API)
statcast/                 # Batter/pitcher Statcast CSVs by season
rp/                       # Relief pitcher data by division/year
SBR Odds/                 # Scraped SportsbookReview opening lines (2026)
models/                   # Trained model assets (not committed — see Setup)
cache/                    # Runtime caches: odds, stats, splits, live snapshot
predictions/              # Predictions log CSV/XLSX (not committed — personal)
misc_py/                  # Utility scripts: odds fetchers, scrapers, diagnostics
```

---

## Setup

### Prerequisites

- Python 3.10+
- A Google service account with Sheets API enabled (for Sheets logging — optional)
- A Gmail App Password (for email reports — optional)
- [The Odds API](https://the-odds-api.com/) key for live odds

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/baseball-model.git
cd baseball-model
pip install -r requirements.txt
```

> **Note:** `requirements.txt` is not yet committed. Install: `mlb-statsapi`, `pybaseball`, `gspread`, `oauth2client`, `pandas`, `scikit-learn`, `numpy`, `scipy`, `python-dotenv`, `openpyxl`

### 2. Configure `.env`

Create a `.env` file in the project root (never committed):

```
ODDS_API_KEY=your_the_odds_api_key
BANKROLL=1000

# Optional — email reports
EMAIL_SENDER=your@gmail.com
EMAIL_APP_PASSWORD=your_app_password
EMAIL_RECIPIENT=your@gmail.com

# Optional — Google Sheets
GOOGLE_SHEETS_ID=your_sheet_id
```

### 3. Build the data pipeline

```bash
# Pull Statcast CSVs and lineup history (2021–2026)
python fetch_player_data.py --seasons 2026

# Train the production model (saves models/log5_regression.pkl)
# In backtest.py, set RUN_MODE = 'build_production_model' and run:
python backtest.py
```

### 4. Run the model

```bash
python model.py
```

Output goes to terminal. Optionally configure `run_model.bat` and `configure_scheduler.ps1` for Windows Task Scheduler automation.

---

## Scheduler Setup (Windows)

Edit `run_model.bat` — set `PROJ_DIR` and `PYTHON_EXE` to match your installation, then run `configure_scheduler.ps1` as Administrator to register the 5 daily triggers.

---

## Key Design Decisions

**No market anchor** — earlier versions included a de-vigged market logit as a feature. The market coefficient dominated all player features ~59:1, making the model a market proxy. Removing it entirely and predicting against opening lines is what generates the edge.

**Opening lines only** — the model was trained on opening lines and that's where the alpha lives. Line movement after 6 PM is the market catching up; the origination bet was already placed.

**Projected lineups for overnight runs** — when official lineups aren't posted yet, the model uses each team's most common batting order across the last 10 confirmed games combined with the announced probable pitchers. This enables origination bets hours before lineups officially post.

---

## License

MIT
