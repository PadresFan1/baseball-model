import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import csv
import json
import calendar
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
API_KEY = os.getenv('API_KEY')

MT = ZoneInfo('America/Denver')

SNAPSHOT_PATH = 'odds_snapshots_2026.csv'
CREDIT_LOG_PATH = 'logs/odds_credit_log.txt'
CAPTURED_CLOSERS_PATH = 'logs/captured_closers.json'
RUN_WINDOW = 'WAVE_LOGGER'
MIN_REMAINING_CREDITS = 40
CAPTURE_WINDOW_MINUTES = 16
CAPTURED_RETENTION_DAYS = 2
PACE_CREDITS_PER_DAY = 12

ODDS_SNAPSHOT_COLUMNS = [
    'snapshot_ts_utc', 'run_window_mst', 'event_id', 'commence_time_utc',
    'home_team', 'away_team', 'book_key', 'book_title', 'market',
    'outcome_name', 'american_odds', 'point'
]


def get_schedule_games(date_str):
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}'
    resp = requests.get(url, timeout=15).json()
    games = []
    for date_entry in resp.get('dates', []):
        games.extend(date_entry.get('games', []))
    return games


def load_captured_closers():
    if not os.path.exists(CAPTURED_CLOSERS_PATH):
        return {}
    with open(CAPTURED_CLOSERS_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_captured_closers(captured):
    os.makedirs(os.path.dirname(CAPTURED_CLOSERS_PATH), exist_ok=True)
    with open(CAPTURED_CLOSERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(captured, f, indent=2)


def prune_captured_closers(captured, today):
    cutoff = today - timedelta(days=CAPTURED_RETENTION_DAYS)
    pruned = {}
    for date_str, game_pks in captured.items():
        try:
            d = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            continue
        if d >= cutoff:
            pruned[date_str] = game_pks
    return pruned


def days_remaining_in_month(today):
    last_day = calendar.monthrange(today.year, today.month)[1]
    return last_day - today.day + 1


def get_remaining_credits():
    if not os.path.exists(CREDIT_LOG_PATH):
        return None
    last_remaining = None
    with open(CREDIT_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            for part in line.split('|'):
                part = part.strip()
                if part.startswith('remaining:'):
                    try:
                        last_remaining = int(part.split(':', 1)[1].strip())
                    except ValueError:
                        pass
    return last_remaining


def log_odds_snapshot(data, run_window_mst, out_path=SNAPSHOT_PATH):
    """Append one row per game+book+market+outcome from a fresh odds API response."""
    if not isinstance(data, list):
        return

    snapshot_ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rows = []
    for game in data:
        event_id = game.get('id')
        commence_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        for book in game.get('bookmakers', []):
            book_key = book.get('key')
            book_title = book.get('title')
            for market in book.get('markets', []):
                market_key = market.get('key')
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'snapshot_ts_utc': snapshot_ts,
                        'run_window_mst': run_window_mst,
                        'event_id': event_id,
                        'commence_time_utc': commence_time,
                        'home_team': home_team,
                        'away_team': away_team,
                        'book_key': book_key,
                        'book_title': book_title,
                        'market': market_key,
                        'outcome_name': outcome.get('name'),
                        'american_odds': outcome.get('price'),
                        'point': outcome.get('point', '')
                    })

    if not rows:
        return

    file_exists = os.path.exists(out_path)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ODDS_SNAPSHOT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def log_odds_credit(headers, out_path=CREDIT_LOG_PATH):
    """Append remaining/used request credits from an odds API response to the credit log."""
    remaining = headers.get('x-requests-remaining', '')
    used = headers.get('x-requests-used', '')
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(f"{ts} | remaining: {remaining} | used: {used}\n")


def main():
    now_utc = datetime.now(timezone.utc)
    today_mst = datetime.now(MT).date()
    today_str = today_mst.strftime('%Y-%m-%d')

    schedule_games = get_schedule_games(today_str)

    captured = prune_captured_closers(load_captured_closers(), today_mst)
    captured_today = set(captured.get(today_str, []))

    target_game_pks = []
    for game in schedule_games:
        game_pk = game.get('gamePk')
        game_date_str = game.get('gameDate')
        if game_pk is None or game_date_str is None or game_pk in captured_today:
            continue
        start_time_utc = datetime.strptime(game_date_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        minutes_until_start = (start_time_utc - now_utc).total_seconds() / 60
        if 0 <= minutes_until_start <= CAPTURE_WINDOW_MINUTES:
            target_game_pks.append(game_pk)

    if not target_game_pks:
        print("No games starting soon")
        return

    remaining = get_remaining_credits()
    if remaining is not None and remaining < MIN_REMAINING_CREDITS:
        print(f"WARNING: only {remaining} odds API credits remaining (< {MIN_REMAINING_CREDITS}). Skipping call.")
        return

    markets = 'h2h,totals'
    if remaining is not None and remaining < days_remaining_in_month(today_mst) * PACE_CREDITS_PER_DAY:
        markets = 'h2h'

    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={API_KEY}&regions=us&markets={markets}&oddsFormat=american'
    response = requests.get(url, timeout=15)
    data = response.json()

    log_odds_credit(response.headers)
    log_odds_snapshot(data, RUN_WINDOW)

    captured.setdefault(today_str, [])
    for game_pk in target_game_pks:
        if game_pk not in captured[today_str]:
            captured[today_str].append(game_pk)
    save_captured_closers(captured)

    n_games = len(data) if isinstance(data, list) else 0
    print(f"Captured closing line for {len(target_game_pks)} game(s); logged {n_games} games total (markets={markets}).")


if __name__ == '__main__':
    main()
