# Baseball Model — Project Log

*Living document. Update after every meaningful session.*
*For technical specs (formulas, file structure, API details, full schema), see [REFERENCE.md](REFERENCE.md).*

---

## Current State — as of 2026-05-26

**Live model** (`model.py`) runs **hourly** via Windows Task Scheduler (updated from 9am/12pm/4pm).

- **Win probability**: Log5 logistic regression (C=3.5, trained on 9,346 games 2021–2025) — replaces Poisson for edge/betting decisions
  - Features: `logit_market_prob`, `d_log5_woba`, `d_log5_xwoba`, `d_xfip` (player-level, platoon-split rolling 100 PA)
  - Model assets: `models/log5_regression.pkl` + `historical_data/player_snapshot.json` (1,118 batters, 645 pitchers, as_of 2026-05-25)
  - Fallback to Poisson if assets missing or lineup not confirmed (pre-lineup games)
  - **Refresh process**: `fetch_player_data.py --seasons 2026` + `pull_historical_lineups([2026])` (incremental) + `backtest.py RUN_MODE='build_production_model'`
- **Run totals / display**: Poisson simulation (5,000 runs/game) still used for `avg_total`, `home_lambda`, `away_lambda`
- Features for Poisson: rolling 7/15-day runs scored/allowed, wOBA, xwOBA, BABIP (offense) + FIP, xFIP, K%, BB% (pitching) — season-to-date from MLB Stats API + Savant
- Park factors applied to lambda
- Injury adjustments (14-day tapered IL window, MLB transactions endpoint)
- Platoon splits when lineups confirmed (vs LHP / vs RHP per batter)
- **Pre-lineup estimation**: games without confirmed lineups now run the full model with platoon_data=None (factor=1.0), shown under `=== PRE-LINEUP ESTIMATES (no platoon) ===`. Logged to CSV with `platoon_confirmed='No'`. Replaced prior hard skip gate.
- Edge thresholds: **Sniper >4.5%** | **Enforcer >4.0%** (updated from flat 7% on 2026-05-25)
- Staking: **dynamic 1/2 Kelly** via `calculate_kelly_stake()` — upgraded 2026-05-23 from flat $20
  - 15% bankroll cap per bet (`MAX_BANKROLL_EXPOSURE`)
  - Effective bankroll = total bankroll minus sum of all open/PENDING bet stakes
  - `recommended_stake` logged as a column in predictions CSV/XLSX/Sheets
- **O/U betting OFF** (`OU_BETTING_ENABLED = False`) — confirmed anti-predictive at team level AND player level
- **O/U terminal/email output OFF (2026-05-24)**: O/U lines removed from game output and accuracy report. Model projected total (`avg_total`) still shown inline via `proj_text` (e.g., `Model total: 8.7`). O/U rows still logged to CSV/XLSX/Sheets for retrospective analysis.
- **Accuracy tracker reset to 2026-05-24**: `MODEL_V2_START = '2026-05-24'`. The pre-overhaul record is no longer shown. Single section: `── Record (since 2026-05-24)`.
- **Platoon analysis NOT printed**: platoon factors calculated and logged (CSV/Sheets), but platoon analysis section removed from accuracy report output entirely.
- **Run Total Accuracy section** replaces O/U direction analysis in accuracy report: shows games tracked, avg error, HIGH/LOW/CLOSE breakdown — no O/U betting framing.
- Predictions logged: 36-col CSV + color-coded XLSX + Google Sheets
- Email report sent only when a game starts within **3 hours** of the current run (`_email_window_hours = 3` at bottom of model.py). Model still runs hourly — logs/Sheets update every run regardless. Terminal prints `[Email skipped]` with hours-until-next-game when suppressed.
- **Inter-run change detection (2026-05-24)**: each email flags per-game changes vs the prior run's logged recommendation
  - 🔄 `PICK CHANGED: A → B` — same game, different team
  - 🆕 `NEW EDGE: A` — no prior edge, now has one
  - ⚠️ `EDGE LOST: A` — had an edge, no longer does
  - Summary line at top of email: `📋 Changes from prior run: X pick changes | X new edges | X edges lost`
  - Silent on first run of the day (no prior data to compare against)
  - Change detection runs after `ml_bet_team` is assigned — critical ordering; running it before caused stale prior-game value to bleed in
- **Game time display (2026-05-25)**: uses `ZoneInfo('America/Denver')` for automatic MDT/MST handling. Game headers show correct local time year-round (no hardcoded UTC-7).
- **Terminal icons (2026-05-25)**: edge found (pre-lineup or confirmed) now shows `✅`, money bag `💰` reserved for `[ACTION: SNIPER/ENFORCER BET]` lines.
- **Email formatting (2026-05-25)**: switched to responsive HTML. Desktop shows monospace `pre` with horizontal scroll. Mobile (≤600px) wraps cleanly with smaller font via `@media` query. Both plain-text and HTML versions sent for universal email client support.
- **Email body cleanup (2026-05-25)**: "Google Sheets updated successfully" removed from email (printed to terminal only).

**SBR odds scraper** (`C:\Users\super\scrape_mlb_odds.py`): Selenium + BeautifulSoup scraper targeting sportsbookreview.com. Collects closing ML odds from BetMGM, FanDuel, Caesars, bet365, DraftKings for every game. Output: `MLB_Odds_Mar29_May26_2026.xlsx` (786 games, Mar 29–May 26 2026). Rerunnable — update start/end dates in `main()`. Covers the 2026 OOS validation gap that SportsGameOdds (~14%) couldn't fill. (2026-05-26)

**Backtest** (`backtest.py`): Round 9 complete. Player-level dual-strategy sim run. O/U player-level test run and definitively ruled out. **2025 holdout fold run (2026-05-25)** — see Backtest History below.

**2026 OOS validation architecture (2026-05-25):**
- `misc_py/fetch_2026_odds.py` — pulls 2026 ML odds from SportsGameOdds with 5-min grace period + |odds|<=450 magnitude filter. Saves to `historical_data/odds_2026.csv`. **NOTE: SportsGameOdds rate-limits quickly and does not reliably store pre-game ML timestamps for finished 2026 games — tested 14% coverage on March-April sample. Run this after daily API quota resets.**
- `RUN_MODE='2026_validation'` — loads saved production model (log5_regression.pkl), builds 2021-2026 rolling caches, collects Log5 features for 2026 games only, evaluates Sniper/Enforcer thresholds. Requires `odds_2026.csv` to be populated.
- `RUN_MODE='2025_holdout'` — isolates 2025 terminal fold (train 2021-2024, test 2025, C=3.5) as closest OOS proxy while 2026 data accumulates.
- `backtest.py` startup now merges `historical_data/odds_2026.csv` if present (same pattern as 2025 supplement).

---

## What's Been Built

### Production model (model.py)
- [x] Poisson model with rolling 7/15-day windows and 5,000 simulations
- [x] Starter RA9 blended with team bullpen (55/45 weight, 15 IP minimum, 7.00 RA9 cap)
- [x] The Odds API integration with 1hr cache
- [x] **Log5 logistic regression** (2026-05-25): win probability from player-level Log5 matchup grid, replaces Poisson for edge/betting decisions. Poisson retained for run totals only.
- [x] **Dual-model thresholds** (2026-05-25): Sniper >4.5%, Enforcer >4.0% — terminal labels `[ACTION: SNIPER BET]` / `[ACTION: ENFORCER BET]`
- [x] Kelly Criterion staking (half-Kelly, 15% bankroll cap, BANKROLL from .env)
- [x] **Dynamic 1/2 Kelly engine** (2026-05-23): `calculate_kelly_stake()` with American→decimal odds conversion, effective bankroll, `recommended_stake` logged per prediction
- [x] Injury adjustment system (tapered 14-day IL window)
- [x] Platoon splits (MLB API confirmed lineups, splits cache 6hr TTL)
- [x] **Pre-lineup estimation** (2026-05-25): removed hard lineup gate; pre-lineup games run full model without platoon (factor=1.0), printed under `=== PRE-LINEUP ESTIMATES (no platoon) ===`
- [x] 3-run daily log with `final_run` logic (game time determines which run is "final")
- [x] Two-pass result system (prevents back-to-back game result mixups)
- [x] Google Sheets integration — value-only wipe preserving formatting
- [x] Excel output with WIN/LOSS color rows
- [x] Email reports with accuracy section — gated to within 3h of next game start
- [x] Accuracy report: single section from 2026-05-24 onward, edge tiers, Run Total Accuracy section
- [x] O/U rows still logged to CSV/XLSX/Sheets (direction + error vs book total) — not shown in terminal/email
- [x] Platoon factors logged silently — platoon analysis removed from all printed output
- [x] Schema migration system (`archive_existing_prediction_logs()`) — one-time on startup
- [x] Windows Task Scheduler automation — `run_model.bat` uses full path `C:\Python314\python.exe` and absolute log paths; confirmed working when PC is asleep / no user logged in (2026-05-26)
- [x] **Inter-run change detection (2026-05-24)**: per-game flags + email summary line for pick changes, new edges, and lost edges across hourly runs

### SBR odds scraper
- [x] **`scrape_mlb_odds.py`** (2026-05-26): Selenium Firefox + BeautifulSoup scraper for sportsbookreview.com
  - Loops all dates automatically; waits for JS render before parsing
  - CSS selectors: `GameRows_eventMarketGridContainer` (games), `OddsCells_numbersContainer` (odds — combined string e.g. `+140-170`, away first)
  - Columns: Date, Time, Away/Home Team, Away/Home Pitcher, Final Score, Away/Home Win%, Opening ML (away & home), Closing ML × 5 books
  - Output: `MLB_Odds_Mar29_May26_2026.xlsx` — 786 games
  - Rerunnable: update `start_date` / `end_date` in `main()` for additional date ranges

### Backtest (backtest.py)
- [x] Random search parameter optimization (7,000+ trials, rounds 1-8)
- [x] 8.3x speedup via prefix sums for rolling averages (O(1) per lookup)
- [x] Holdout validation — discovered R1-5 results were overfit
- [x] Individual metric testing — isolated each feature's genuine out-of-sample signal
- [x] Calibration meta model: logistic regression over 4 validated metrics (Brier 0.2449 on 2025 holdout)
- [x] Market-residual meta model: de-vigged market logit added as anchor feature
- [x] Walk-forward 4-fold expanding-window validation
- [x] **Dual-strategy framework updated** (2026-05-25): both strategies C=3.5 — Sniper edge>4.5%, Enforcer edge>4.0%
- [x] Player-level data pipeline: Statcast CSVs 2021–2026, game lineup cache 2021–2026 (incremental), batter + pitcher rolling caches
- [x] O/U tracking and analysis across all backtest modes
- [x] **O/U player-level test** (2026-05-24): `collect_ou_features_player_level()` + `walk_forward_ou_strategy()` — sum-based features, pushes excluded, de-vigged juice edge — definitively ruled out
- [x] **Log5 feature pipeline** (2026-05-25): `collect_game_features_log5()`, `log5_matchup()`, `_LOG5_FEATURES`
- [x] **Log5 parameter sweep** (2026-05-25): `walk_forward_log5_sweep()` — 35 combinations (C × edge), Kelly staking, Markdown table output. Found 4.0% edge cliff, C=3.5 optimal.
- [x] **Production model builder** (2026-05-25): `RUN_MODE='build_production_model'` — trains C=3.5 regression on all available data, saves `log5_regression.pkl` + `player_snapshot.json`
- [x] **game_lineups.json incremental update** (2026-05-25): `pull_historical_lineups()` detects current year, resumes from last cached date instead of skipping or full re-pull

---

## Resolved Issues

### model.py
| Bug | What Happened | Fix |
|-----|--------------|-----|
| UnicodeEncodeError | Emoji output crashed Windows terminal | `sys.stdout.reconfigure(encoding='utf-8')` at top |
| Google Sheets 429 | ~62 `append_row()` calls per run hit rate limit | Replaced with single `sheet.update()` |
| O/U evaluate_result() never matched | Checked `bet_type in ['Over','Under']` but stored value is `'Over/Under'` | Fixed check to `bet_type == 'Over/Under'`, `bet_team` for direction |
| Old O/U rows stuck at PENDING | `evaluate_result()` never matched, so rows with actual scores stayed PENDING | Added retroactive re-evaluation pass in `log_predictions()` |
| NaN game_num | `str(nan or '1')` = `'nan'` — NaN is truthy in Python | Added `_game_num_str()` helper using `pd.isna()` |
| `'N/A'` became `'nan'` in CSV | pandas reads `'N/A'` as NaN by default | Changed cancelled bet_team to `'CANCELLED'` |
| Finished games marked CANCELLED | Completed games past commence_time not in `upcoming` feed | Added `live_or_done` check using MLB API status field |
| Date sort wrong in Sheets | Old rows used M/D/YYYY, new rows YYYY-MM-DD — breaks lexicographic sort | `_norm_date()` normalizes all dates to ISO on CSV load |
| Sheets wipe erased formatting | `sheet.clear()` wiped everything including conditional formatting | Switched to `sheet.batch_clear(["A2:AK2000"])` — values only |
| Back-to-back games wrong results | Same teams playing two days in a row would inherit yesterday's result | Two-pass system: Pass 1 = today PENDING, Pass 2 = yesterday's rows only |
| Old schema incompatible | Adding `model_strategy` column broke existing log files | `archive_existing_prediction_logs()` one-time migration on startup |
| `print_accuracy_report()` KeyError 'bet_team' | CSV had only a header row (no data); `pd.DataFrame([])` produces no columns | Added `if not rows: return` guard before DataFrame creation |
| Migration missed `recommended_stake` | `archive_existing_prediction_logs()` returned early if `model_strategy` present, even when `recommended_stake` was absent | Migration now requires both columns before skipping |
| `Pandas4Warning` on boolean mask | `df['bet_type'] == 'Moneyline'` returns `BooleanDtype` (pandas 2.x StringDtype), `apply(_is_active_bet)` returns object dtype — `&` between them warns | Added `.astype(bool)` to `apply()` result; `has_scores` now uses `.fillna(False).astype(bool)` |
| `TypeError` on `recommended_stake` assignment | CSV loaded with `dtype=str` → every column is string-typed; assigning float `11.52` to `recommended_stake` raises `TypeError` in pandas 2.x | Wrapped assignment in `str()`: `df.at[idx, 'recommended_stake'] = str(pred['recommended_stake'])` |
| Change detection used stale `ml_bet_team` | Change detection block placed before `ml_bet_team` assignment — read leftover value from previous loop iteration, producing bogus `PICK CHANGED: ATL → No Bet` on a live ATL bet | Moved block to immediately after `ml_bet_team` is set |
| Rain-delayed games marked CANCELLED | `live_or_done` check didn't include Delayed/Suspended statuses — game dropped from odds feed during delay, triggering CANCELLED | `live_or_done` now uses `startswith('Delayed')` and `startswith('Suspended')` to catch all MLB API variants |
| Task Scheduler silent failures (PC asleep / no user logged in) | `run_model.bat` used bare `python` command — only resolves in the user account environment; Task Scheduler runs as system user with no PATH | Replaced with full path `C:\Python314\python.exe`; log paths changed from relative to absolute (`C:\Users\super\baseball-model\logs\`) (2026-05-26) |

### backtest.py
| Bug | Fix |
|-----|-----|
| `random_search()` never called | Fixed the call site |
| `get_league_hr_fb_rate()` called but not defined | Added function |
| `calc_xfip()` KeyError on missing dict keys | Fixed |
| `ip_to_float()` wrong — `"5.1"` ≠ 5.1 IP (baseball notation: .1=⅓, .2=⅔) | Added correct converter |
| `load_woba_fip_constants()` — `'season'` vs `'Season'` column name mismatch | Fixed |
| KeyError `'w_barrel_bat'` — removed from param space but still in formula | Fixed |
| `type=details` in Savant API params returning 0 rows | Removed — incompatible with `group_by=name-date` |
| Overfitting — all prior rounds trained and evaluated on same 2021-2024 data | Rebuilt with proper holdout split; then moved to player-level feature architecture |
| sklearn `FutureWarning` on `penalty='l2'` | `penalty` param deprecated in sklearn 1.8 (L2 is still default) | Removed `penalty='l2'` from both `LogisticRegression` calls |

---

## Key Decisions & Why

**Terminal/email output cleaned up (2026-05-24):** O/U betting lines removed from game output and accuracy report — O/U is anti-predictive so the betting framing adds noise. Model projected total still shown inline (`Model total: X.X` in `proj_text`). O/U rows kept in predictions log for retrospective total accuracy analysis. Accuracy tracker reset to 2026-05-24 (`MODEL_V2_START`); the pre-overhaul record is irrelevant to current model performance. Platoon analysis section removed from accuracy report output (factors still calculated and logged).

**O/U definitively closed (2026-05-24):** Tested at both team level and player level. Team-level: win rate decreases as threshold increases (48.4% → 42.2%). Player-level (rolling 100 PA vs pitcher hand, actual starter 100 BF xFIP, sum features): Precise -0.3% ROI, Aggressive -0.9% ROI across 2022-2025. 2023 collapsed both strategies (-11.9% / -8.6%) — same year-specific overfitting pattern as moneyline. The book's total is more efficiently priced than any of our feature sets can overcome. O/U direction still tracked in predictions log for retrospective analysis, but no path to profitability identified.

**No PA cap on injury adjustments (hitters):** Elite hitters naturally sit at ~12-15% of team PA. An artificial cap would understate players like Judge. Pitchers kept at 25% IP cap — starters only pitch every 5th day.

**Platoon factor uncapped:** 30 PA minimum + ≥3 batter requirement already guards against small-sample extremes. Adding a cap was solving a solved problem.

**Platoon results not displayed in terminal:** Tracked silently for 3-week retrospective. Accuracy report activates at 20+ confirmed-lineup settled bets.

**1/2 Kelly over flat $20 (2026-05-23):** Flat bets don't scale with bankroll or edge strength. Half-Kelly with 15% hard cap sizes proportionally — bigger stakes on high-edge bets, preserves capital on borderline ones. Effective bankroll prevents over-committing on simultaneous games.

**Hourly runs + lineup gate (2026-05-23):** Switched from 3 fixed daily runs to hourly Task Scheduler trigger. Safe because the lineup confirmation gate skips any game whose batting order isn't officially posted — no placeholder-lineup bets, no double-logging. The `run_num` 3-window map (`<11h` / `<15h` / else) naturally collapses multiple hourly writes to the same slot, so the schema stays intact.

**Email throttle removed (2026-05-23):** With hourly runs the 2hr throttle was redundant and blocked reports on confirmed-lineup games.

**Email gated to 3h pre-game window (2026-05-24):** Hourly runs were generating too many emails throughout the day. Email now only sends when at least one upcoming game starts within 3 hours. Model still runs every hour — CSV/XLSX/Sheets update on every run. Window is `_email_window_hours = 3` at bottom of model.py. Terminal prints `[Email skipped] No games within 3h. Next game in X.Xh.` on suppressed runs.

**Team-level stats failed (root cause):** Season-to-date Statcast metrics are too coarse and too public. The market has already priced them. Even with a de-vigged market logit as an anchor, win rate locks at ~49.7% — below the 52.4% break-even for -110 juice. Volume collapsed fold-over-fold as the regression learned the market explains all variance. Led to Round 9 player-level rebuild.

**Individual metric validation before composite:** After discovering R5's +16% ROI was overfit (p=1.0 on 2025 holdout), we stopped re-running random_search. Instead: validate each metric independently out-of-sample, then rebuild only from metrics with genuine signal.

**Bullpen removed entirely:** w_bullpen_fat had -0.246 correlation (actively harmful). w_bullpen_qual was +0.007 (noise). Removing improved results.

**favorites_only=False locked:** True showed 64-67% win rates but only at n=90-135. At n≥300 all top-10 trials flip to False. Confirmed small-sample artifact.

**2026 historical backtest not yet possible (2026-05-25):** SportsGameOdds API does not reliably store pre-game ML timestamps for finished 2026 games — all 9 bookmakers update their lines during gameplay, so the `lastUpdatedAt` filter returns nothing. A 5-minute grace period + |odds|<=450 magnitude cap yields only ~14% coverage (67/480 games in a March-April test sample, extrapolates to ~110/792 total). That coverage produces too few bets (Sniper generates ~1 bet per 100 games in-sample) for statistical significance. Primary evidence of out-of-sample validity remains the 4-fold walk-forward sweep (2022-2025): Sniper +8.86% ROI / 17 bets/yr, Enforcer +8.22% ROI / 35 bets/yr. The 2025 terminal fold with C=3.5 gives the closest per-year proxy. True 2026 validation builds organically from `predictions_log.csv` as the live model accumulates settled bets going forward.

**SBR odds scraper fills the 2026 coverage gap (2026-05-26):** `scrape_mlb_odds.py` collects closing ML odds from SportsBookReview (BetMGM, FanDuel, Caesars, bet365, DraftKings) for 786 games Mar 29–May 26 2026. Unlike SportsGameOdds, SBR stores closing lines reliably. This dataset can now feed `RUN_MODE='2026_validation'` once integrated into the backtest odds-loading logic.

**Log5 ported to production (2026-05-25):** The backtest's Log5 logistic regression is now the win-probability engine in model.py. Poisson still runs for run-total display only. The Log5 regression explicitly models each batter-pitcher matchup interaction (`Pr = (B×P/L) / [(B×P/L) + (1-B)(1-P)/(1-L)]`) before aggregating, capturing signal the market hasn't fully priced. Model trained on 9,346 games (2021–2025, C=3.5), player snapshot refreshed with 2026 data.

**Pre-lineup estimation (2026-05-25):** Replaced hard lineup gate with degraded-mode inference. Games without confirmed lineups now run the full model using probable starter stats but platoon_factor=1.0. Shown as `=== PRE-LINEUP ESTIMATES (no platoon) ===` with `[PRE-LINEUP ESTIMATE]` tag. Allows early signals hours before official lineups post.

**Dual-model thresholds calibrated via Log5 sweep (2026-05-25):** 35-combination grid search (5 C values × 7 edge thresholds) with Kelly staking found 4.0% as the edge signal cliff — win rates collapse to ~50% below it. C=3.5 optimal for both strategies. Sniper: C=3.5, edge>4.5% (+8.9% ROI, 17 bets/yr). Enforcer: C=3.5, edge>4.0% (+8.2% ROI, 35 bets/yr). Full results: `historical_data/log5_sweep_results.txt`.

**2026 data pipeline (2026-05-25):** `fetch_player_data.py` updated with `2026: (3,5)` season months. `pull_historical_lineups()` now supports incremental current-year updates — detects last cached date, resumes from there, uses yesterday as end_date. `build_batter_rolling_cache` and `build_pitcher_rolling_cache` accept `extra_target_dates` so today's date is always included as a target for the production snapshot. Pitcher xFIP gap closed: 1,461/1,673 2026 starter rows now have true IP denominators.

**Automatic timezone handling (2026-05-25):** Replaced hardcoded `MST = timezone(timedelta(hours=-7))` with `ZoneInfo('America/Denver')`. The model now auto-detects Mountain Daylight Time (UTC-6) during summer and Mountain Standard Time (UTC-7) during winter. Game headers display "MDT" or "MST" accordingly. Fixes the 1-hour display offset that occurred during DST months.

**Terminal icon clarity (2026-05-25):** Unified the edge-found icon across pre-lineup and confirmed-lineup games to `✅`. The money bag icon `💰` is now reserved exclusively for the `[ACTION: SNIPER BET]` and `[ACTION: ENFORCER BET]` action lines. This eliminates icon redundancy and clarifies intent: checkmark = we detected an edge, money bag = we recommend a bet.

**Email cleanup & responsive design (2026-05-25):** Removed "Google Sheets updated successfully" message from email body (still prints to terminal). Switched email from plain text to responsive HTML with fallback, using MIMEMultipart alternative. Desktop clients (>600px) get monospace `pre` with horizontal scroll capability and 14px font. Mobile clients (≤600px) activate a `@media` query for `white-space: pre-wrap` with 11px font so content wraps cleanly instead of forcing horizontal scrolling. Both plain-text and HTML versions included for universal email client support.

---

## Backtest History

| Round | Change | Result |
|-------|--------|--------|
| 1–3 | Base model → added statcast → added bullpen | Avg ROI -1.51% (n≥300). Bullpen actively hurt. |
| 4 | Bullpen removed, favorites_only locked False | Avg ROI +0.11% |
| 5 | barrel_bat dropped, weights rebalanced | Avg ROI +0.53%, best +16.33% at n=403 — but overfit |
| 6 | O/U tracking added, 2025 data merged | O/U anti-predictive at all thresholds |
| 7 | Park factors added to lambda | Structural improvement |
| 8 | Holdout split introduced (train=[2021,22] only) | 2025 holdout: **-1.9% ROI, p=1.0** — confirmed R5 +16% was overfitting |
| Meta model | Logistic regression over 4 validated metrics | Brier 0.2449 on 2025 holdout — directionally useful, not profitable |
| Market-residual | Added de-vigged market logit as anchor feature | Win rate 40% → 49.7%, but still below 52.4% break-even |
| 9 | Player-level: rolling 100 PA per batter split by pitcher hand, actual starter rolling 100 BF | **Sniper (C=1.0, edge>5%): +7.4% ROI, 56.6% win rate, 122 bets (30/yr). Enforcer (C=5.0, edge>3.5%): -3.3% ROI, 50.2% win rate, 582 bets. Sniper is the only profitable player-level strategy.** |
| 9 Log5 | Same player-level pipeline but Log5 matchup formula per batter vs pitcher instead of ratio diff. Features: logit_market_prob, d_log5_woba, d_log5_xwoba, d_xfip | **Sniper: +8.4% ROI, 57.1% win rate, 35 bets (9/yr) — volume too thin at >5% threshold. Enforcer: -0.4% ROI, 53.1% win rate, 292 bets (73/yr) — near breakeven.** Raw output: historical_data/round9_log5_results.txt |
| Log5 sweep | 35-combo grid search (C × edge threshold, Kelly staking). Found 4.0% cliff — win rate collapses below it. C=3.5 optimal for both strategies. | Sniper C=3.5/>4.5%: +8.9% ROI, 17 bets/yr. Enforcer C=3.5/>4.0%: +8.2% ROI, 35 bets/yr. Full table: historical_data/log5_sweep_results.txt. **These are the current production thresholds.** |
| O/U player-level | Same pipeline with sum features (home+away), P(over) logistic regression | Precise: -0.3% ROI (1,842 bets). Aggressive: -0.9% ROI (2,852 bets). 2023 catastrophic for both. **O/U closed.** |
| 2025 holdout fold | Train 2021-2024, test 2025, C=3.5. Kelly staking (½ Kelly, 15% cap, $1K start). Isolates 2025 terminal fold with production thresholds. | **Sniper (edge>4.5%): 8 bets, 62.5% win rate, +37.0% ROI, +$191.95 profit, bankroll $1,191.95. Enforcer (edge>4.0%): 15 bets, 60.0% win rate, +35.1% ROI, +$244.96 profit, bankroll $1,244.96.** Small sample (n=8/15) but directionally very strong — confirms 2025 is the model's strongest fold. Note: Kelly staking amplifies these vs flat-$20 baseline. |

**4 validated metrics with genuine out-of-sample signal:** wOBA, xwOBA_bat, xFIP, xwOBA_pit

---

## Round 9 Status — COMPLETE

**Goal:** Player-level features — individual batter rolling 100 PA (vs LHP vs RHP) + actual starter rolling 100 BF xFIP — to carry residual signal the market hasn't priced.

**All done:**
- [x] 15 Statcast CSVs downloaded to `player_data/`
- [x] Game lineup cache built for 2021-2025 (`game_lineups.json`, ~12,150 boxscore API calls)
- [x] Pitcher hand cache (`pitcher_hand_cache.json`)
- [x] Batter rolling cache: 1,457,524 entries across 1,812 (hand, date) pairs
- [x] Pitcher rolling cache built successfully
- [x] `walk_forward_dual_strategy()` run on player-level features (9,562 games with O/U line; ~8,200 with ML odds)
- [x] O/U player-level test run (`RUN_MODE = 'player_level_ou'`) — ruled out

**Dual-strategy ML results (2026-05-25):**
- **The Sniper** (C=1.0, edge>5%): 122 bets, 56.6% win rate, **+7.4% ROI**, $+181.57 profit — profitable
- **The Enforcer** (C=5.0, edge>3.5%): 582 bets, 50.2% win rate, **-3.3% ROI**, $-387.54 — not profitable
- Sniper volume is very low (~30 bets/yr). 2025 fold: only 6 bets.
- Raw output saved to `historical_data/round9_dual_strategy_results.txt`

**O/U player-level results:** Precise -0.3% ROI / Aggressive -0.9% ROI — not profitable. Closed.

---

## Pending Evaluations

**Platoon retrospective:** ~20 confirmed-lineup settled bets needed to activate the accuracy report section. Key question: do "platoon-boosted" bets outperform "platoon-lowered"?

**Kelly staking performance:** `recommended_stake` now logged per prediction. After enough settled bets accumulate, compare Kelly-sized results vs the old flat-$20 baseline.

**2025 ML odds gap:** Aug 17–Oct 1, 2025 has O/U data but only 28/608 games with pre-game ML odds (SportsGameOdds stores last-known odds, not snapshots). If The Odds API historical endpoint becomes available or budget allows, fill this gap.

---

## Next Steps (Priority Order)

1. ~~**Record Round 9 dual-strategy ML results**~~ — DONE (2026-05-25).
2. ~~**Port Log5 regression to model.py**~~ — DONE (2026-05-25). Live in production.
3. ~~**Close pitcher xFIP gap**~~ — DONE (2026-05-25). 2026 box scores in game_lineups.json.
4. ~~**Build 2026 OOS validation architecture**~~ — DONE (2026-05-25). `2026_validation` + `2025_holdout` RUN_MODEs added. `fetch_2026_odds.py` built. 2025 holdout fold shows Sniper +37% ROI / 8 bets, Enforcer +35% ROI / 15 bets. True 2026 test blocked by SportsGameOdds pre-game odds coverage; accumulating via live predictions_log going forward.
5. **Platoon retrospective** — let confirmed-lineup bets accumulate to 20+
6. **Kelly performance review** — once ~50 settled bets with `recommended_stake` logged
7. **Log5 snapshot refresh cadence** — decide how often to re-run `build_production_model` (weekly? monthly?) to keep player rolling stats current through 2026 season
8. **2026 OOS validation** — integrate `MLB_Odds_Mar29_May26_2026.xlsx` (SBR scrape, 786 games) into `RUN_MODE='2026_validation'` odds-loading logic; SportsGameOdds ~14% coverage no longer the bottleneck

---

## Future Ideas (Unscheduled)

- **Weather/wind** — not yet priced at line-posting time; would need an API
- **Travel/rest disadvantage** — back-to-back road games, cross-timezone series
- **Exponential recency weighting** — within rolling window, weight recent games more heavily (currently equal-weighted)
- **Preseason/April cold start** — use ZiPS/Steamer projections as starting lambda, blend to actuals through April
- **Playoff mode** — different bullpen weights, small sample, aces on short rest; needs separate model architecture
- **Remote server hosting** — discussed 2026-05-26; tabled until Task Scheduler proves unreliable long-term. Would allow model access from anywhere without keeping local PC awake. Revisit if Task Scheduler issues recur.
