"""
fetch_player_data.py
====================
Downloads individual player-level Statcast data from Baseball Savant for the
player-level backtest feature system (Round 9).

Produces three file types under player_data/:
  batters_vs_L_YYYY.csv   -- per-batter, per-game stats vs left-handed pitchers
  batters_vs_R_YYYY.csv   -- per-batter, per-game stats vs right-handed pitchers
  starters_YYYY.csv       -- per-starting-pitcher, per-game stats

Baseball Savant caps each request at 10,000 rows.  To stay under the cap,
requests are split by calendar month (12 requests per file per season).

Usage
-----
    python fetch_player_data.py              # all seasons 2021-2025
    python fetch_player_data.py --seasons 2024 2025
    python fetch_player_data.py --test       # 2024 only, verify columns then exit
"""

import argparse
import os
import sys
import time
from io import StringIO

import pandas as pd
import requests

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = 'player_data'
os.makedirs(OUT_DIR, exist_ok=True)

# Baseball Savant statcast search CSV endpoint
SAVANT_URL = 'https://baseballsavant.mlb.com/statcast_search/csv'

# Columns we keep from each export.  Extra columns are dropped to reduce size.
# Presence is checked at runtime; missing columns are silently skipped.
BATTER_COLS = [
    'player_id', 'player_name', 'game_date', 'game_pk',
    'pa', 'woba', 'xwoba', 'ba', 'xba', 'slg', 'xslg',
    'k_percent', 'bb_percent', 'hardhit_percent',
    'swing_miss_percent', 'batter_run_value_per_100',
]

PITCHER_COLS = [
    'player_id', 'player_name', 'game_date', 'game_pk',
    'pa',
    'woba',         # wOBA allowed (fallback when xwoba absent)
    'xwoba',        # xwOBA allowed
    'so',           # strikeouts
    'bb',           # walks
    'hrs',          # home runs (Savant uses 'hrs', not 'hr')
    'k_percent', 'bb_percent',
    'hardhit_percent', 'swing_miss_percent',
    'pitcher_run_value_per_100',
    # 'ip' is NOT in Savant grouped data; IP is estimated in backtest from pa+obp:
    #   ip_est = pa * (1 - obp) / 3
    #   fb_est = ip_est * 3 * 0.40   (league-avg ~40% fly-ball rate)
    'obp',          # OBP against -- used to estimate outs -> IP
]

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://baseballsavant.mlb.com/statcast_search',
}

# Season calendar: (start_month, end_month) inclusive
_SEASON_MONTHS = {
    2021: (4, 10), 2022: (4, 10), 2023: (3, 10),
    2024: (3, 9),  2025: (3, 9),  2026: (3, 5),
}

_REQUEST_DELAY = 2.5   # seconds between requests (polite rate-limiting)
_CAP_WARNING   = 9500  # warn if a monthly slice hits this row count


# ── Core fetch ────────────────────────────────────────────────────────────────

def _savant_params(player_type, season, pitcher_throws=None,
                   date_gt=None, date_lt=None):
    """
    Build query params for the Savant statcast_search CSV endpoint.

    NOTE: Do NOT add type=details — that requests pitch-level data and
    returns 0 rows when combined with group_by=name-date (aggregated view).
    """
    params = {
        'player_type': player_type,
        'season':      season,
        'hfSea':       f'{season}|',
        'hfGT':        'R|',           # regular season only
        'group_by':    'name-date',    # one row per player per game
        'sort_col':    'pa',
        'sort_order':  'desc',
        'min_pa':      1,
    }
    if pitcher_throws in ('L', 'R'):
        params['pitcherThrows'] = pitcher_throws
    if date_gt:
        params['game_date_gt'] = date_gt   # e.g. '2024-04-01'
    if date_lt:
        params['game_date_lt'] = date_lt   # e.g. '2024-04-30'
    return params


def _fetch_one(params, label, verbose=False):
    """
    Fetch one CSV slice from Baseball Savant.
    Returns DataFrame or None.  Retries once after 30s on failure.
    """
    if verbose:
        req = requests.Request('GET', SAVANT_URL, params=params, headers=_HEADERS)
        print(f"    URL: {req.prepare().url}")

    for attempt in range(2):
        try:
            resp = requests.get(SAVANT_URL, params=params,
                                headers=_HEADERS, timeout=120)
            if verbose:
                print(f"    HTTP {resp.status_code} | {len(resp.content):,} bytes")

            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            text = resp.text.strip()
            if not text:
                raise ValueError("empty body")
            if text.lstrip().startswith('<'):
                raise ValueError(f"HTML response (bot-block?): {text[:200]}")

            df = pd.read_csv(StringIO(text), low_memory=False)

            if len(df) == 0:
                # Headers-only response is not an error — some months have no games
                return pd.DataFrame(columns=df.columns)

            if len(df) >= _CAP_WARNING:
                print(f"    WARNING: {label} returned {len(df):,} rows"
                      f" -- may be hitting the 10k cap. Consider splitting further.")

            return df

        except Exception as e:
            print(f"    {label} attempt {attempt+1} failed: {e}")
            if attempt == 0:
                time.sleep(30)

    return None


def _fetch_season_by_month(player_type, season, pitcher_throws=None,
                            label_prefix='', verbose_first=False):
    """
    Fetch all months of a season and concatenate.  Each monthly slice stays
    well under Savant's 10,000-row cap.

    Returns a combined DataFrame (possibly empty if all months fail).
    """
    start_m, end_m = _SEASON_MONTHS.get(season, (4, 10))
    frames = []
    first = True

    for month in range(start_m, end_m + 1):
        import calendar
        last_day = calendar.monthrange(season, month)[1]
        date_gt = f'{season}-{month:02d}-01'
        date_lt = f'{season}-{month:02d}-{last_day}'
        label   = f'{label_prefix} {season}-{month:02d}'

        params = _savant_params(player_type, season, pitcher_throws,
                                date_gt=date_gt, date_lt=date_lt)

        # Print URL only on first slice (or if verbose_first forced)
        df = _fetch_one(params, label, verbose=(first and verbose_first))
        first = False

        if df is not None and not df.empty:
            frames.append(df)
            print(f"    {label}: {len(df):,} rows")
        elif df is not None:
            print(f"    {label}: 0 rows (off-season month, skipped)")

        time.sleep(_REQUEST_DELAY)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _keep_cols(df, wanted):
    """Return df restricted to columns in `wanted` that actually exist in df."""
    present = [c for c in wanted if c in df.columns]
    return df[present].copy()


def _rename_xwoba(df):
    """Normalise the xwOBA column name regardless of which long form Savant uses."""
    renames = {
        'estimated_woba_using_speedangle': 'xwoba',
        'estimated_ba_using_speedangle':   'xba',
        'estimated_slg_using_speedangle':  'xslg',
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


# ── Public fetch functions ────────────────────────────────────────────────────

def fetch_batters(season, hand, test_mode=False):
    """
    Download and save player_data/batters_vs_{hand}_{season}.csv.
    In test_mode: prints all column names then exits without saving.
    """
    out_path = os.path.join(OUT_DIR, f'batters_vs_{hand}_{season}.csv')
    if os.path.exists(out_path) and not test_mode:
        print(f"  Exists -- skipping {out_path}")
        return

    print(f"\n  Fetching batters vs {hand}HP -- {season} (by month)...")

    if test_mode:
        # Single-month probe just to verify columns
        import calendar
        start_m = _SEASON_MONTHS.get(season, (4, 10))[0]
        last_day = calendar.monthrange(season, start_m)[1]
        params = _savant_params('batter', season, pitcher_throws=hand,
                                date_gt=f'{season}-{start_m:02d}-01',
                                date_lt=f'{season}-{start_m:02d}-{last_day}')
        df = _fetch_one(params, f'batters vs {hand}HP {season}-{start_m:02d}',
                        verbose=True)
        if df is not None and not df.empty:
            df = _rename_xwoba(df)
            print(f"\n  All {len(df.columns)} columns returned by Savant:")
            for i, col in enumerate(df.columns):
                print(f"    [{i:02d}] {col}")
            wanted   = [c for c in BATTER_COLS if c in df.columns]
            missing  = [c for c in BATTER_COLS if c not in df.columns]
            print(f"\n  BATTER_COLS present  ({len(wanted)}): {wanted}")
            print(f"  BATTER_COLS missing  ({len(missing)}): {missing}")
        return

    df = _fetch_season_by_month('batter', season, pitcher_throws=hand,
                                label_prefix=f'batters vs {hand}HP',
                                verbose_first=True)
    if df is None or df.empty:
        print(f"  WARNING: no data for batters vs {hand}HP {season}")
        return

    df = _rename_xwoba(df)
    df = _keep_cols(df, BATTER_COLS)
    df = df[df['pa'].notna() & (df['pa'].astype(float) > 0)]
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates(subset=['player_id', 'game_date', 'game_pk'])
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  ({len(df):,} rows, {df['player_id'].nunique():,} players)")


def fetch_starters(season, test_mode=False):
    """
    Download and save player_data/starters_{season}.csv.
    Pulls all pitchers, then filters to starters (PA >= 12 as proxy for 3+ IP).
    In test_mode: prints all column names then exits without saving.
    """
    out_path = os.path.join(OUT_DIR, f'starters_{season}.csv')
    if os.path.exists(out_path) and not test_mode:
        print(f"  Exists -- skipping {out_path}")
        return

    print(f"\n  Fetching pitchers -- {season} (by month)...")

    if test_mode:
        import calendar
        start_m = _SEASON_MONTHS.get(season, (4, 10))[0]
        last_day = calendar.monthrange(season, start_m)[1]
        params = _savant_params('pitcher', season,
                                date_gt=f'{season}-{start_m:02d}-01',
                                date_lt=f'{season}-{start_m:02d}-{last_day}')
        df = _fetch_one(params, f'pitchers {season}-{start_m:02d}', verbose=True)
        if df is not None and not df.empty:
            df = _rename_xwoba(df)
            print(f"\n  All {len(df.columns)} columns returned by Savant:")
            for i, col in enumerate(df.columns):
                print(f"    [{i:02d}] {col}")
            wanted  = [c for c in PITCHER_COLS if c in df.columns]
            missing = [c for c in PITCHER_COLS if c not in df.columns]
            print(f"\n  PITCHER_COLS present  ({len(wanted)}): {wanted}")
            print(f"  PITCHER_COLS missing  ({len(missing)}): {missing}")
        return

    df = _fetch_season_by_month('pitcher', season,
                                label_prefix='pitchers',
                                verbose_first=True)
    if df is None or df.empty:
        print(f"  WARNING: no pitcher data for {season}")
        return

    df = _rename_xwoba(df)

    # Filter to starting pitchers: PA >= 12 is a reliable proxy for 3+ IP starter
    # (relievers typically face 3-6 batters; starters face 15-25+)
    if 'pa' in df.columns:
        starters = df[df['pa'].fillna(0).astype(float) >= 12].copy()
    else:
        starters = df.copy()

    print(f"  Filtered {len(df):,} pitcher-game rows -> {len(starters):,} starter-game rows (PA >= 12)")

    starters = _rename_xwoba(starters)
    starters = _keep_cols(starters, PITCHER_COLS)
    starters = starters[starters['pa'].notna() & (starters['pa'].astype(float) > 0)]
    starters['game_date'] = pd.to_datetime(starters['game_date']).dt.strftime('%Y-%m-%d')
    starters = starters.drop_duplicates(subset=['player_id', 'game_date', 'game_pk'])
    starters.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  ({len(starters):,} rows, {starters['player_id'].nunique():,} pitchers)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(seasons, test_mode=False):
    if test_mode:
        seasons = [2024]
        print("TEST MODE -- probing column names for 2024 (one month each, no files saved).\n")
        print("Batters vs LHP:")
        fetch_batters(2024, 'L', test_mode=True)
        print("\nPitchers:")
        fetch_starters(2024, test_mode=True)
        print("\nTest complete.  Review columns above, then run without --test for full pull.")
        return

    print(f"Fetching player-level Statcast data for seasons: {seasons}")
    print(f"Output directory: {os.path.abspath(OUT_DIR)}\n")

    for season in seasons:
        print(f"\n{'='*55}")
        print(f"  Season {season}")
        print(f"{'='*55}")

        fetch_batters(season, 'L')
        fetch_batters(season, 'R')
        fetch_starters(season)

    print("\n\nDone. Files written to player_data/")
    print("Next: set RUN_MODE = 'player_level_meta' in backtest.py and run.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch player-level Statcast data from Baseball Savant.'
    )
    parser.add_argument('--seasons', nargs='+', type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--test', action='store_true',
                        help='Print column names for 2024 (one month each) then exit')
    args = parser.parse_args()
    main(seasons=args.seasons, test_mode=args.test)
