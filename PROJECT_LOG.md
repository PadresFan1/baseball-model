# Baseball Model — Project Log

*Living document. Update after every meaningful session.*
*For technical specs (formulas, file structure, API details, full schema), see [REFERENCE.md](memory/REFERENCE.md).*

---

## Current State — as of 2026-05-23

**Live model** (`model.py`) runs at 9am, 12pm, 4pm MST via Windows Task Scheduler.

- Poisson simulation (5,000 runs/game) → win probability
- Features: rolling 7/15-day runs scored/allowed, wOBA, xwOBA, BABIP (offense) + FIP, xFIP, K%, BB% (pitching) — season-to-date from MLB Stats API + Savant
- Park factors applied to lambda
- Injury adjustments (14-day tapered IL window, MLB transactions endpoint)
- Platoon splits when lineups confirmed (vs LHP / vs RHP per batter)
- Edge threshold: **7%** probability vs de-vigged market line
- Staking: **dynamic 1/2 Kelly** via `calculate_kelly_stake()` — upgraded 2026-05-23 from flat $20
  - 15% bankroll cap per bet (`MAX_BANKROLL_EXPOSURE`)
  - Effective bankroll = total bankroll minus sum of all open/PENDING bet stakes
  - `recommended_stake` logged as a column in predictions CSV/XLSX/Sheets
- **O/U betting OFF** (`OU_BETTING_ENABLED = False`) — backtest proved anti-predictive at every threshold
- Predictions logged: 36-col CSV + color-coded XLSX + Google Sheets
- Email report after each run (2hr throttle), includes accuracy report

**Backtest** (`backtest.py`): Round 9 in progress — rebuilding features using individual player-level rolling 100 PA stats split by pitcher handedness.

---

## What's Been Built

### Production model (model.py)
- [x] Poisson model with rolling 7/15-day windows and 5,000 simulations
- [x] Starter RA9 blended with team bullpen (55/45 weight, 15 IP minimum, 7.00 RA9 cap)
- [x] The Odds API integration with 2.5hr cache
- [x] 7% edge threshold detection
- [x] Kelly Criterion staking (half-Kelly, 15% bankroll cap, BANKROLL from .env)
- [x] **Dynamic 1/2 Kelly engine** (2026-05-23): `calculate_kelly_stake()` with American→decimal odds conversion, effective bankroll, `recommended_stake` logged per prediction
- [x] Injury adjustment system (tapered 14-day IL window)
- [x] Platoon splits (MLB API confirmed lineups, splits cache 6hr TTL)
- [x] 3-run daily log with `final_run` logic (game time determines which run is "final")
- [x] Two-pass result system (prevents back-to-back game result mixups)
- [x] Google Sheets integration — value-only wipe preserving formatting
- [x] Excel output with WIN/LOSS color rows
- [x] Email reports with accuracy section (2hr throttle)
- [x] Accuracy report: pre/post v2 split, edge tiers, platoon analysis, O/U direction tracking
- [x] O/U tracking (direction + error vs book total) even with betting disabled
- [x] Schema migration system (`archive_existing_prediction_logs()`) — one-time on startup
- [x] Windows Task Scheduler automation

### Backtest (backtest.py)
- [x] Random search parameter optimization (7,000+ trials, rounds 1-8)
- [x] 8.3x speedup via prefix sums for rolling averages (O(1) per lookup)
- [x] Holdout validation — discovered R1-5 results were overfit
- [x] Individual metric testing — isolated each feature's genuine out-of-sample signal
- [x] Calibration meta model: logistic regression over 4 validated metrics (Brier 0.2449 on 2025 holdout)
- [x] Market-residual meta model: de-vigged market logit added as anchor feature
- [x] Walk-forward 4-fold expanding-window validation
- [x] Dual-strategy framework: Sniper (C=1.0, edge>5%) + Enforcer (C=5.0, edge>3.5%)
- [x] Player-level data pipeline: 15 Statcast CSVs, game lineup cache, batter + pitcher rolling caches
- [x] O/U tracking and analysis across all backtest modes

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

---

## Key Decisions & Why

**O/U disabled:** Win rate *decreases* as threshold increases (48.4% at 1.0-run → 42.2% at 3.0-run). The model's run estimates are less accurate than the book's total for O/U purposes. The book total is more efficiently priced than expected. O/U predictions still tracked for retrospective analysis.

**No PA cap on injury adjustments (hitters):** Elite hitters naturally sit at ~12-15% of team PA. An artificial cap would understate players like Judge. Pitchers kept at 25% IP cap — starters only pitch every 5th day.

**Platoon factor uncapped:** 30 PA minimum + ≥3 batter requirement already guards against small-sample extremes. Adding a cap was solving a solved problem.

**Platoon results not displayed in terminal:** Tracked silently for 3-week retrospective. Accuracy report activates at 20+ confirmed-lineup settled bets.

**1/2 Kelly over flat $20 (2026-05-23):** Flat bets don't scale with bankroll or edge strength. Half-Kelly with 15% hard cap sizes proportionally — bigger stakes on high-edge bets, preserves capital on borderline ones. Effective bankroll prevents over-committing on simultaneous games.

**Team-level stats failed (root cause):** Season-to-date Statcast metrics are too coarse and too public. The market has already priced them. Even with a de-vigged market logit as an anchor, win rate locks at ~49.7% — below the 52.4% break-even for -110 juice. Volume collapsed fold-over-fold as the regression learned the market explains all variance. Led to Round 9 player-level rebuild.

**Individual metric validation before composite:** After discovering R5's +16% ROI was overfit (p=1.0 on 2025 holdout), we stopped re-running random_search. Instead: validate each metric independently out-of-sample, then rebuild only from metrics with genuine signal.

**Bullpen removed entirely:** w_bullpen_fat had -0.246 correlation (actively harmful). w_bullpen_qual was +0.007 (noise). Removing improved results.

**favorites_only=False locked:** True showed 64-67% win rates but only at n=90-135. At n≥300 all top-10 trials flip to False. Confirmed small-sample artifact.

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
| 9 (active) | Player-level: rolling 100 PA per batter split by pitcher hand, actual starter rolling 100 BF | Pipeline built; dual-strategy sim pending |

**4 validated metrics with genuine out-of-sample signal:** wOBA, xwOBA_bat, xFIP, xwOBA_pit

---

## Round 9 Status

**Goal:** Player-level features — individual batter rolling 100 PA (vs LHP vs RHP) + actual starter rolling 100 BF xFIP — to carry residual signal the market hasn't priced.

**Done:**
- [x] 15 Statcast CSVs downloaded to `player_data/`
- [x] Game lineup cache built for 2021-2025 (`game_lineups.json`, ~12,150 boxscore API calls)
- [x] Pitcher hand cache (`pitcher_hand_cache.json`)
- [x] Batter rolling cache: 1,457,524 entries across 1,812 (hand, date) pairs
- [x] Pitcher rolling cache: NaN IP bug fixed; was building at session end

**Still needed:**
- [ ] Confirm pitcher cache built successfully
- [ ] Run `walk_forward_dual_strategy()` on player-level features
- [ ] Evaluate: does Sniper or Enforcer generate edge vs the market-residual team-level model?

```
# To run:
python backtest.py   # RUN_MODE = 'player_level_meta' in backtest.py
```

---

## Pending Evaluations

**Platoon retrospective:** ~20 confirmed-lineup settled bets needed to activate the accuracy report section. Key question: do "platoon-boosted" bets outperform "platoon-lowered"?

**Kelly staking performance:** `recommended_stake` now logged per prediction. After enough settled bets accumulate, compare Kelly-sized results vs the old flat-$20 baseline.

**2025 ML odds gap:** Aug 17–Oct 1, 2025 has O/U data but only 28/608 games with pre-game ML odds (SportsGameOdds stores last-known odds, not snapshots). If The Odds API historical endpoint becomes available or budget allows, fill this gap.

---

## Next Steps (Priority Order)

1. **Finish Round 9** — confirm pitcher cache, run dual-strategy sim, evaluate player-level results
2. **Platoon retrospective** — let confirmed-lineup bets accumulate to 20+
3. **Kelly performance review** — once ~50 settled bets with `recommended_stake` logged
4. **O/U re-examination** — only if Round 9 shows a genuine edge path (team totals, negative-binomial)

---

## Future Ideas (Unscheduled)

- **Weather/wind** — not yet priced at line-posting time; would need an API
- **Travel/rest disadvantage** — back-to-back road games, cross-timezone series
- **Exponential recency weighting** — within rolling window, weight recent games more heavily (currently equal-weighted)
- **Negative Binomial for O/U** — handles run-scoring variance better than Poisson
- **Team totals** — home vs away separately instead of game total; may be less efficient market
- **Preseason/April cold start** — use ZiPS/Steamer projections as starting lambda, blend to actuals through April
- **Playoff mode** — different bullpen weights, small sample, aces on short rest; needs separate model architecture
