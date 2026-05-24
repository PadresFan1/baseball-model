---
name: project-round9-player-level
description: Round 9 backtest overhaul — player-level features replacing team-level rolling averages
metadata:
  type: project
---

Team-level cumulative season-to-date stats carry no residual information beyond what the sportsbook prices in (market-residual model: ~49.7% win rate, below 52.4% break-even). Root cause identified in Section 18 of session summary.

Round 9 rebuilds features from the bottom up using individual player-level data.

**Architecture:**
- `aggregate_lineup_metric(player_stats_dict, lineup_list, lg_avg)` — dot product of per-slot metrics with `PA_WEIGHTS = [0.123, 0.120, ..., 0.100]` (positions 1-9)
- `pull_historical_lineups(seasons)` — fetches MLB API box scores to get actual batting order + starter IDs. Cached to `historical_data/game_lineups.json`
- `build_batter_rolling_cache(batter_data_by_hand)` — rolling 100 PA per player per pitcher-hand using prefix sums + binary search. Cache key: `(player_id, hand, date)`
- `build_pitcher_rolling_cache(pitcher_df, fip_const_by_season)` — rolling 100 BF per starter. xFIP computed with per-date league HR/FB rate. Cache key: `(player_id, date)`
- `collect_game_features_player_level(seasons, ...)` — same 5-feature schema as meta model: `logit_market_prob, d_woba, d_xwoba_bat, d_xfip, d_xwoba_pit`

**Data needed (download via fetch_player_data.py):**
- `player_data/batters_vs_L_YYYY.csv` and `batters_vs_R_YYYY.csv` — Baseball Savant individual batter per-game stats filtered by pitcher hand
- `player_data/starters_YYYY.csv` — Baseball Savant individual starter per-game stats (filtered to IP >= 3.0)

**Run sequence:**
1. `python fetch_player_data.py --test` — verify 2024 data format
2. `python fetch_player_data.py` — full 2021-2025 pull (~15 files, ~3-4 min)
3. Set `RUN_MODE = 'player_level_meta'` in backtest.py
4. Run backtest.py — pulls lineups first (~30 min/season one-time), then builds caches, then runs walk-forward financial sim

**Why this matters:** The key residual vs the sportsbook: handedness-split batting stats (vs LHP vs vs RHP) and actual starter performance (not team ERA) may not be fully priced into early lines. Rolling 100 PA captures recent form better than season-to-date cumulative.

**How to apply:** When discussing backtest results or next steps, this is the active architecture as of Round 9.
