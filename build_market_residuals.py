#!/usr/bin/env python3
"""
build_market_residuals.py
--------------------------
Builds a market-residual dataset for the residual-targeting model architecture.

Residual definition:
  residual = home_win_indicator (1/0) - market_prob_home

Reference odds selection per game (from odds_snapshots.csv):
  Primary  : snapshot closest to 3 PM ET, within ±3 hours
  Fallback : snapshot closest to 5 AM ET, within ±3 hours
  Last resort: earliest pre-game snapshot available

Usage:
  # Log current odds_cache.json as a snapshot (run at 3 PM ET and 5 AM ET):
  python build_market_residuals.py --log

  # Build residual CSV from snapshot log + historical results:
  python build_market_residuals.py --build

  # Backfill 2021-2025 from historical_odds_clean.csv (no time filtering):
  python build_market_residuals.py --historical

  # Backfill historical, then append 2026 snapshot-based records:
  python build_market_residuals.py --historical
  python build_market_residuals.py --build --append

  # Combined log + build (typical scheduled run):
  python build_market_residuals.py --log --build --append

Outputs:
  cache/odds_snapshots.csv         — running odds log (append-only)
  historical_data/market_residuals.csv — training labels (overwrite or append)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent

SNAPSHOT_LOG  = ROOT / 'cache' / 'odds_snapshots.csv'
HIST_ODDS     = ROOT / 'historical_data' / 'historical_odds_clean.csv'
RESIDUALS_OUT = ROOT / 'historical_data' / 'market_residuals.csv'
ODDS_CACHE    = ROOT / 'cache' / 'odds_cache.json'

SNAPSHOT_COLS = [
    'logged_at',      # ISO 8601 UTC when snapshot was recorded
    'game_id',        # The Odds API game ID
    'game_date',      # YYYY-MM-DD in ET
    'commence_time',  # ISO 8601 UTC game start
    'home_team',      # FanGraphs abbreviation
    'away_team',
    'fd_home_ml',     # FanDuel home moneyline (American, empty if unavailable)
    'fd_away_ml',
    'dk_home_ml',     # DraftKings home moneyline
    'dk_away_ml',
]

RESIDUAL_COLS = [
    'game_date',
    'home_team',
    'away_team',
    'snapshot_label',      # '3pm', '5am', 'earliest', or 'historical'
    'snapshot_logged_at',  # ISO 8601 UTC of the snapshot used (empty for historical)
    'fd_home_ml',
    'fd_away_ml',
    'market_prob_home',    # de-vigged implied probability (home)
    'market_prob_away',    # de-vigged implied probability (away); = 1 - market_prob_home
    'home_win',            # 1 or 0
    'residual',            # home_win - market_prob_home
]

# Snapshot target times (all times assume EDT = UTC-4, standard for baseball season)
# 3 PM EDT = 19:00 UTC  |  5 AM EDT = 09:00 UTC
_TARGET_3PM_UTC = 19
_TARGET_5AM_UTC = 9
_WINDOW_HOURS   = 3   # max distance from target hour to qualify for that window

# Full team name -> FanGraphs abbreviation (matches model.py's NAME_TO_FG)
_NAME_TO_FG: Dict[str, str] = {
    'Athletics': 'ATH',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Minnesota Twins': 'MIN',
    'Philadelphia Phillies': 'PHI',
    'Atlanta Braves': 'ATL',
    'Chicago White Sox': 'CHW',
    'Miami Marlins': 'MIA',
    'New York Yankees': 'NYY',
    'Milwaukee Brewers': 'MIL',
    'Los Angeles Angels': 'LAA',
    'Arizona Diamondbacks': 'ARI',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Dodgers': 'LAD',
    'Washington Nationals': 'WSN',
    'New York Mets': 'NYM',
}


# ── Odds math ─────────────────────────────────────────────────────────────────

def american_to_implied(odds: float) -> float:
    """American moneyline -> raw implied probability (includes vig)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def devige(raw_home: float, raw_away: float) -> Tuple[float, float]:
    """
    Multiplicative de-vig: divide each implied prob by their sum so they total 1.0.
    Returns (market_prob_home, market_prob_away).
    """
    total = raw_home + raw_away
    if total <= 0.0:
        raise ValueError(f"Cannot de-vig non-positive probabilities: {raw_home}, {raw_away}")
    return raw_home / total, raw_away / total


def odds_to_market_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    """American ML odds -> de-vigged market probabilities."""
    return devige(american_to_implied(home_ml), american_to_implied(away_ml))


# ── Snapshot logging ──────────────────────────────────────────────────────────

def log_odds_snapshot(
    odds_cache_path: Optional[str] = None,
    snapshot_log_path: Optional[str] = None,
) -> int:
    """
    Read odds_cache.json and append one row per game to odds_snapshots.csv.
    Rows are stamped with the current UTC time. Safe to call multiple times per day —
    each call adds a new snapshot row; the selector later picks the right one.

    Returns the number of game rows logged.
    """
    cache_path = Path(odds_cache_path) if odds_cache_path else ODDS_CACHE
    log_path   = Path(snapshot_log_path) if snapshot_log_path else SNAPSHOT_LOG

    with open(cache_path, encoding='utf-8') as f:
        raw = json.load(f)

    games: list = raw if isinstance(raw, list) else list(raw.values())

    now_utc   = datetime.now(timezone.utc).isoformat()
    new_file  = not log_path.exists()
    logged    = 0
    skipped   = 0

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS)
        if new_file:
            writer.writeheader()

        for game in games:
            commence_raw = game.get('commence_time', '')
            try:
                commence_dt = datetime.fromisoformat(commence_raw.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                skipped += 1
                continue

            # Game date in ET (EDT = UTC-4 during baseball season)
            game_date_et = (commence_dt - timedelta(hours=4)).strftime('%Y-%m-%d')

            home_full = game.get('home_team', '')
            away_full = game.get('away_team', '')
            home_abbr = _NAME_TO_FG.get(home_full)
            away_abbr = _NAME_TO_FG.get(away_full)

            if not home_abbr or not away_abbr:
                print(f"  WARNING: unknown team '{home_full}' / '{away_full}' — skipped")
                skipped += 1
                continue

            fd_home = fd_away = dk_home = dk_away = None
            for bm in game.get('bookmakers', []):
                bm_key = bm.get('key', '')
                for market in bm.get('markets', []):
                    if market.get('key') != 'h2h':
                        continue
                    for outcome in market.get('outcomes', []):
                        price    = outcome.get('price')
                        is_home  = (outcome.get('name') == home_full)
                        if bm_key == 'fanduel':
                            if is_home:
                                fd_home = price
                            else:
                                fd_away = price
                        elif bm_key == 'draftkings':
                            if is_home:
                                dk_home = price
                            else:
                                dk_away = price

            writer.writerow({
                'logged_at':     now_utc,
                'game_id':       game.get('id', ''),
                'game_date':     game_date_et,
                'commence_time': commence_raw,
                'home_team':     home_abbr,
                'away_team':     away_abbr,
                'fd_home_ml':    '' if fd_home is None else fd_home,
                'fd_away_ml':    '' if fd_away is None else fd_away,
                'dk_home_ml':    '' if dk_home is None else dk_home,
                'dk_away_ml':    '' if dk_away is None else dk_away,
            })
            logged += 1

    msg = f"  Logged {logged} game(s) -> {log_path}"
    if skipped:
        msg += f"  ({skipped} skipped)"
    print(msg)
    return logged


# ── Snapshot selection ────────────────────────────────────────────────────────

def _hours_from_utc_hour(ts: datetime, target_utc_hour: int) -> float:
    """
    Minutes-of-day distance between ts and target UTC hour, handling midnight wrap.
    Returns distance in hours (always non-negative).
    """
    snap_min   = ts.hour * 60 + ts.minute
    target_min = target_utc_hour * 60
    diff = abs(snap_min - target_min)
    return min(diff, 1440 - diff) / 60.0


def select_snapshot(
    rows: List[dict],
    commence_dt: datetime,
) -> Tuple[Optional[dict], str]:
    """
    Given all snapshot rows for one game, return (best_row, label).

    label values:
      '3pm'     — snapshot within ±3h of 3 PM ET (19:00 UTC)
      '5am'     — snapshot within ±3h of 5 AM ET (09:00 UTC)
      'earliest' — no window match; used the earliest pre-game snapshot
      ''        — no usable snapshot found

    Only considers snapshots recorded before game start (pre-game odds).
    """
    pre_game: List[Tuple[datetime, dict]] = []
    for r in rows:
        raw_ts = r.get('logged_at', '')
        try:
            ts = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
        except ValueError:
            continue
        if ts < commence_dt:
            pre_game.append((ts, r))

    if not pre_game:
        return None, ''

    # Priority 1: closest to 3 PM ET within window
    in_3pm = [
        (ts, r) for ts, r in pre_game
        if _hours_from_utc_hour(ts, _TARGET_3PM_UTC) <= _WINDOW_HOURS
    ]
    if in_3pm:
        best_ts, best_row = min(in_3pm, key=lambda x: _hours_from_utc_hour(x[0], _TARGET_3PM_UTC))
        return best_row, '3pm'

    # Priority 2: closest to 5 AM ET within window
    in_5am = [
        (ts, r) for ts, r in pre_game
        if _hours_from_utc_hour(ts, _TARGET_5AM_UTC) <= _WINDOW_HOURS
    ]
    if in_5am:
        best_ts, best_row = min(in_5am, key=lambda x: _hours_from_utc_hour(x[0], _TARGET_5AM_UTC))
        return best_row, '5am'

    # Fallback: earliest pre-game snapshot regardless of time
    _, best_row = min(pre_game, key=lambda x: x[0])
    return best_row, 'earliest'


# ── Result loading ────────────────────────────────────────────────────────────

def _load_results(hist_path: Optional[str] = None) -> Dict[Tuple[str, str, str], int]:
    """
    Load {(game_date, home_team, away_team): home_win} from historical_odds_clean.csv.
    home_win = 1 if home_score > away_score, else 0.
    """
    path = Path(hist_path) if hist_path else HIST_ODDS
    results: Dict[Tuple[str, str, str], int] = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                hw = 1 if int(row['home_score']) > int(row['away_score']) else 0
            except (ValueError, KeyError):
                continue
            key = (row['date'].strip(), row['home_team'].strip(), row['away_team'].strip())
            results[key] = hw
    return results


# ── Build from snapshots ──────────────────────────────────────────────────────

def build_residual_dataset(
    snapshot_log_path: Optional[str] = None,
    hist_results_path: Optional[str] = None,
    output_path: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    append: bool = False,
) -> int:
    """
    Build market_residuals.csv from odds_snapshots.csv + game results.

    For each game:
      1. Select the 3 PM ET snapshot (fallback: 5 AM ET, then earliest pre-game).
      2. Prefer FanDuel odds; fall back to DraftKings if FanDuel is missing.
      3. Convert to de-vigged market probabilities.
      4. Compute residual = home_win - market_prob_home.

    Returns number of games written.
    """
    log_path = Path(snapshot_log_path) if snapshot_log_path else SNAPSHOT_LOG
    out_path = Path(output_path) if output_path else RESIDUALS_OUT

    if not log_path.exists():
        print(f"  No snapshot log at {log_path} — run --log first.")
        return 0

    results = _load_results(hist_results_path)

    # Group snapshot rows by (game_date, home_team, away_team)
    game_snaps: Dict[Tuple[str, str, str], List[dict]] = {}
    game_commence: Dict[Tuple[str, str, str], str] = {}
    with open(log_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['game_date'], row['home_team'], row['away_team'])
            game_snaps.setdefault(key, []).append(row)
            game_commence.setdefault(key, row.get('commence_time', ''))

    if date_from:
        game_snaps = {k: v for k, v in game_snaps.items() if k[0] >= date_from}
    if date_to:
        game_snaps = {k: v for k, v in game_snaps.items() if k[0] <= date_to}

    written = skipped_no_result = skipped_bad_odds = skipped_no_snap = 0
    mode = 'a' if append else 'w'
    write_header = not append or not out_path.exists()

    with open(out_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESIDUAL_COLS)
        if write_header:
            writer.writeheader()

        for key in sorted(game_snaps):
            game_date, home_team, away_team = key

            # Parse commence time for pre-game filtering
            commence_raw = game_commence.get(key, '')
            try:
                commence_dt = datetime.fromisoformat(commence_raw.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Fall back to end-of-day if commence time is unavailable
                commence_dt = datetime.strptime(game_date + 'T23:59:00+00:00',
                                                 '%Y-%m-%dT%H:%M:%S%z')

            snap, label = select_snapshot(game_snaps[key], commence_dt)
            if snap is None:
                skipped_no_snap += 1
                continue

            # Prefer FanDuel; fall back to DraftKings
            raw_home = snap.get('fd_home_ml') or snap.get('dk_home_ml') or ''
            raw_away = snap.get('fd_away_ml') or snap.get('dk_away_ml') or ''
            src_home = snap.get('fd_home_ml') or snap.get('dk_home_ml') or ''
            src_away = snap.get('fd_away_ml') or snap.get('dk_away_ml') or ''

            try:
                home_ml = float(raw_home)
                away_ml = float(raw_away)
            except (ValueError, TypeError):
                skipped_bad_odds += 1
                continue

            try:
                market_prob_home, market_prob_away = odds_to_market_probs(home_ml, away_ml)
            except ValueError:
                skipped_bad_odds += 1
                continue

            result_key = (game_date, home_team, away_team)
            if result_key not in results:
                skipped_no_result += 1
                continue

            home_win = results[result_key]
            writer.writerow({
                'game_date':          game_date,
                'home_team':          home_team,
                'away_team':          away_team,
                'snapshot_label':     label,
                'snapshot_logged_at': snap.get('logged_at', ''),
                'fd_home_ml':         home_ml,
                'fd_away_ml':         away_ml,
                'market_prob_home':   round(market_prob_home, 6),
                'market_prob_away':   round(market_prob_away, 6),
                'home_win':           home_win,
                'residual':           round(home_win - market_prob_home, 6),
            })
            written += 1

    print(f"  Snapshot residuals: {written} games -> {out_path}")
    if skipped_no_result:
        print(f"    {skipped_no_result} skipped — no matching result in historical_odds_clean.csv")
    if skipped_bad_odds:
        print(f"    {skipped_bad_odds} skipped — missing or invalid odds")
    if skipped_no_snap:
        print(f"    {skipped_no_snap} skipped — no pre-game snapshot found")
    return written


# ── Historical backfill ───────────────────────────────────────────────────────

def build_historical_residuals(
    hist_path: Optional[str] = None,
    output_path: Optional[str] = None,
    append: bool = False,
) -> int:
    """
    Backfill market_residuals.csv from historical_odds_clean.csv (2021-2025).
    No snapshot filtering — uses whichever FanDuel odds are stored in that file
    directly (opening lines for most seasons).

    label = 'historical' for all rows produced here.
    """
    src_path = Path(hist_path) if hist_path else HIST_ODDS
    out_path = Path(output_path) if output_path else RESIDUALS_OUT

    mode = 'a' if append else 'w'
    write_header = not append or not out_path.exists()
    written = skipped = 0

    with open(src_path, encoding='utf-8') as fin, \
         open(out_path, mode, newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=RESIDUAL_COLS)
        if write_header:
            writer.writeheader()

        for row in reader:
            try:
                home_score = int(row['home_score'])
                away_score = int(row['away_score'])
            except (ValueError, KeyError):
                skipped += 1
                continue

            raw_home_ml = (row.get('fd_home_ml') or '').strip()
            raw_away_ml = (row.get('fd_away_ml') or '').strip()
            try:
                home_ml = float(raw_home_ml)
                away_ml = float(raw_away_ml)
            except ValueError:
                skipped += 1
                continue

            try:
                market_prob_home, market_prob_away = odds_to_market_probs(home_ml, away_ml)
            except ValueError:
                skipped += 1
                continue

            home_win = 1 if home_score > away_score else 0
            writer.writerow({
                'game_date':          row['date'].strip(),
                'home_team':          row['home_team'].strip(),
                'away_team':          row['away_team'].strip(),
                'snapshot_label':     'historical',
                'snapshot_logged_at': '',
                'fd_home_ml':         home_ml,
                'fd_away_ml':         away_ml,
                'market_prob_home':   round(market_prob_home, 6),
                'market_prob_away':   round(market_prob_away, 6),
                'home_win':           home_win,
                'residual':           round(home_win - market_prob_home, 6),
            })
            written += 1

    print(f"  Historical residuals: {written} games -> {out_path}")
    if skipped:
        print(f"    {skipped} skipped — missing scores or odds")
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='Market residual dataset builder — logs odds snapshots and '
                    'computes de-vigged residuals for model training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--log', action='store_true',
                   help='Log current odds_cache.json as a timestamped snapshot')
    p.add_argument('--build', action='store_true',
                   help='Build residuals from snapshot log + game results')
    p.add_argument('--historical', action='store_true',
                   help='Backfill 2021-2025 residuals from historical_odds_clean.csv')
    p.add_argument('--append', action='store_true',
                   help='Append to existing output file instead of overwriting')
    p.add_argument('--from', dest='date_from', metavar='YYYY-MM-DD',
                   help='Start date filter (--build only)')
    p.add_argument('--to', dest='date_to', metavar='YYYY-MM-DD',
                   help='End date filter (--build only)')
    p.add_argument('--odds-cache', metavar='PATH',
                   help='Override path to odds_cache.json')
    p.add_argument('--snapshot-log', metavar='PATH',
                   help='Override path to odds_snapshots.csv')
    p.add_argument('--output', metavar='PATH',
                   help='Override path to market_residuals.csv')

    args = p.parse_args()

    if not any([args.log, args.build, args.historical]):
        p.print_help()
        sys.exit(0)

    if args.log:
        print("Logging odds snapshot...")
        log_odds_snapshot(
            odds_cache_path=args.odds_cache,
            snapshot_log_path=args.snapshot_log,
        )

    if args.historical:
        print("Backfilling historical residuals (2021-2025)...")
        build_historical_residuals(
            output_path=args.output,
            append=args.append,
        )

    if args.build:
        print("Building residuals from snapshot log...")
        build_residual_dataset(
            snapshot_log_path=args.snapshot_log,
            output_path=args.output,
            date_from=args.date_from,
            date_to=args.date_to,
            append=args.append,
        )


if __name__ == '__main__':
    main()
