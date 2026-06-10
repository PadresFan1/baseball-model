import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import csv
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
API_KEY = os.getenv('API_KEY')

# ── Config ──────────────────────────────────────────────────────────────
DRY_RUN = True              # must default True — zero Odds API calls while True
START_DATE = '2022-04-07'
END_DATE = '2025-10-01'
REGIONS = 'us'
MARKETS = 'h2h,totals'
CREDITS_PER_SNAPSHOT = 20    # 10 per region per market on the historical endpoint
MAX_SNAPSHOTS_PER_DAY = 8

ET = ZoneInfo('America/New_York')
WAVE_GAP_MINUTES = 30
OPENER_HOUR_ET = 19
CLOSER_LEAD_MINUTES = 5

SCHEDULE_CACHE_PATH = 'historical_data/backfill_schedule_cache.json'
BACKFILL_CSV_PATH = 'historical_data/odds_snapshots_backfill.csv'
CHECKPOINT_PATH = 'historical_data/backfill_checkpoint.json'
GAPS_CSV_PATH = 'historical_data/backfill_gaps.csv'

ODDS_SNAPSHOT_COLUMNS = [
    'snapshot_ts_utc', 'run_window_mst', 'event_id', 'commence_time_utc',
    'home_team', 'away_team', 'book_key', 'book_title', 'market',
    'outcome_name', 'american_odds', 'point'
]
BACKFILL_COLUMNS = ODDS_SNAPSHOT_COLUMNS + ['snapshot_kind', 'requested_ts_utc']
GAPS_COLUMNS = ['date', 'home_team', 'away_team', 'missing_book']
REQUIRED_BOOKS = ('draftkings', 'fanduel')


# ── Schedule cache (free MLB Stats API) ────────────────────────────────
def fetch_schedule_for_date(date_str):
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}'
    resp = requests.get(url, timeout=15).json()
    games = []
    for date_entry in resp.get('dates', []):
        for g in date_entry.get('games', []):
            if g.get('gameType') != 'R':
                continue
            teams = g.get('teams', {})
            games.append({
                'gamePk': g.get('gamePk'),
                'gameDate': g.get('gameDate'),
                'gameType': g.get('gameType'),
                'home_team': teams.get('home', {}).get('team', {}).get('name'),
                'away_team': teams.get('away', {}).get('team', {}).get('name'),
            })
    return games


def _cache_is_old_format(cache):
    for games in cache.values():
        if games:
            return 'gameType' not in games[0]
    return False


def load_or_build_schedule_cache(start_date, end_date):
    cache = {}
    if os.path.exists(SCHEDULE_CACHE_PATH):
        with open(SCHEDULE_CACHE_PATH, 'r', encoding='utf-8') as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}

    if cache and _cache_is_old_format(cache):
        print("  Existing schedule cache predates gameType filtering — rebuilding.")
        cache = {}

    updated = False
    d = start_date
    n_fetched = 0
    while d <= end_date:
        date_str = d.strftime('%Y-%m-%d')
        if date_str not in cache:
            cache[date_str] = fetch_schedule_for_date(date_str)
            updated = True
            n_fetched += 1
            if n_fetched % 50 == 0:
                print(f"  ...fetched schedule for {n_fetched} new dates")
        d += timedelta(days=1)

    if updated:
        os.makedirs(os.path.dirname(SCHEDULE_CACHE_PATH), exist_ok=True)
        with open(SCHEDULE_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)

    return cache


# ── Snapshot planning ───────────────────────────────────────────────────
def cluster_waves(start_times_utc):
    sorted_times = sorted(start_times_utc)
    waves = []
    for t in sorted_times:
        if waves and (t - waves[-1][-1]) <= timedelta(minutes=WAVE_GAP_MINUTES):
            waves[-1].append(t)
        else:
            waves.append([t])
    return waves


def merge_waves(waves, max_waves):
    while len(waves) > max_waves:
        best_idx, best_gap = 0, None
        for i in range(len(waves) - 1):
            gap = waves[i + 1][0] - waves[i][-1]
            if best_gap is None or gap < best_gap:
                best_gap, best_idx = gap, i
        waves[best_idx:best_idx + 2] = [waves[best_idx] + waves[best_idx + 1]]
    return waves


def opener_requested_ts(game_date):
    previous_day = game_date - timedelta(days=1)
    opener_dt_et = datetime(previous_day.year, previous_day.month, previous_day.day,
                             OPENER_HOUR_ET, 0, 0, tzinfo=ET)
    return opener_dt_et.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def build_plan(start_date, end_date, schedule_cache):
    plan = []
    d = start_date
    while d <= end_date:
        date_str = d.strftime('%Y-%m-%d')
        games = schedule_cache.get(date_str, [])
        start_times = []
        for g in games:
            game_date_str = g.get('gameDate')
            if game_date_str:
                start_times.append(datetime.strptime(game_date_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc))

        if start_times:
            waves = merge_waves(cluster_waves(start_times), MAX_SNAPSHOTS_PER_DAY - 1)
            closer_waves = []
            for wave in waves:
                requested_ts = (min(wave) - timedelta(minutes=CLOSER_LEAD_MINUTES)).strftime('%Y-%m-%dT%H:%M:%SZ')
                closer_waves.append({'requested_ts_utc': requested_ts, 'n_games': len(wave)})

            plan.append({
                'date': date_str,
                'opener_requested_ts': opener_requested_ts(d),
                'closer_waves': closer_waves,
            })
        d += timedelta(days=1)
    return plan


def print_plan_summary(plan):
    game_days = len(plan)
    opener_count = game_days
    closer_count = sum(len(day['closer_waves']) for day in plan)
    total_snapshots = opener_count + closer_count
    total_credits = total_snapshots * CREDITS_PER_SNAPSHOT
    avg_per_day = total_snapshots / game_days if game_days else 0

    print("=== Backfill Plan ===")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Game days: {game_days}")
    print(f"Opener snapshots: {opener_count}")
    print(f"Closer snapshots: {closer_count}")
    print(f"Total snapshots: {total_snapshots}")
    print(f"Estimated credits ({CREDITS_PER_SNAPSHOT}/snapshot): {total_credits}")
    print(f"Avg snapshots/day: {avg_per_day:.2f}")


# ── Checkpointing ───────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_checkpoint(last_completed_date, total_credits_used):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'last_completed_date': last_completed_date, 'total_credits_used': total_credits_used}, f, indent=2)


# ── Odds API (historical) ───────────────────────────────────────────────
def historical_odds_url(requested_ts_utc):
    return (f'https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds'
            f'?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat=american'
            f'&date={requested_ts_utc}')


def call_historical_odds(requested_ts_utc):
    url = historical_odds_url(requested_ts_utc)
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response
    if response.status_code == 429 or response.status_code >= 500:
        print(f"  API returned {response.status_code}, waiting 10s and retrying...")
        time.sleep(10)
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response
        print(f"  Retry failed with {response.status_code}: {response.text[:200]}")
        return None
    print(f"  API error {response.status_code}: {response.text[:200]}")
    return None


def build_rows(games, snapshot_ts, requested_ts_utc, snapshot_kind):
    rows = []
    for game in games:
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
                        'run_window_mst': 'BACKFILL',
                        'event_id': event_id,
                        'commence_time_utc': commence_time,
                        'home_team': home_team,
                        'away_team': away_team,
                        'book_key': book_key,
                        'book_title': book_title,
                        'market': market_key,
                        'outcome_name': outcome.get('name'),
                        'american_odds': outcome.get('price'),
                        'point': outcome.get('point', ''),
                        'snapshot_kind': snapshot_kind,
                        'requested_ts_utc': requested_ts_utc,
                    })
    return rows


def check_gaps(games, date_str):
    gap_rows = []
    for game in games:
        book_keys = {b.get('key') for b in game.get('bookmakers', [])}
        for required in REQUIRED_BOOKS:
            if required not in book_keys:
                gap_rows.append({
                    'date': date_str,
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'missing_book': required,
                })
    return gap_rows


def append_csv_rows(rows, out_path, columns):
    if not rows:
        return
    file_exists = os.path.exists(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ── Execution ────────────────────────────────────────────────────────────
def run_backfill(plan):
    if not API_KEY:
        print("ERROR: API_KEY not found in .env")
        return

    checkpoint = load_checkpoint()
    last_completed = checkpoint.get('last_completed_date')
    total_credits = checkpoint.get('total_credits_used', 0)

    remaining_plan = [day for day in plan if last_completed is None or day['date'] > last_completed]
    if not remaining_plan:
        print("Nothing to do — all game days already completed per checkpoint.")
        return

    if last_completed:
        print(f"Resuming from checkpoint after {last_completed} ({total_credits} credits used so far).")

    snapshot_counter = 0

    for day in remaining_plan:
        date_str = day['date']
        print(f"\nProcessing {date_str}...")

        snapshots = [('opener', day['opener_requested_ts'])]
        for wave in day['closer_waves']:
            snapshots.append(('closer', wave['requested_ts_utc']))

        day_rows = []
        gap_rows = []

        for kind, requested_ts in snapshots:
            response = call_historical_odds(requested_ts)
            if response is None:
                save_checkpoint(last_completed, total_credits)
                print(f"  Stopping: repeated failure fetching {kind} snapshot for {date_str}. "
                      f"Checkpoint saved at {last_completed}.")
                return

            payload = response.json()
            actual_ts = payload.get('timestamp')
            games = payload.get('data') or []

            day_rows.extend(build_rows(games, actual_ts, requested_ts, kind))
            if kind == 'closer':
                gap_rows.extend(check_gaps(games, date_str))

            total_credits += CREDITS_PER_SNAPSHOT
            snapshot_counter += 1
            if snapshot_counter % 10 == 0:
                print(f"  [{snapshot_counter} snapshots] x-requests-remaining: "
                      f"{response.headers.get('x-requests-remaining', '?')}")

            time.sleep(1)

        append_csv_rows(day_rows, BACKFILL_CSV_PATH, BACKFILL_COLUMNS)
        append_csv_rows(gap_rows, GAPS_CSV_PATH, GAPS_COLUMNS)

        last_completed = date_str
        save_checkpoint(last_completed, total_credits)
        print(f"  Done: {len(snapshots)} snapshots, {len(day_rows)} rows. Total credits used: {total_credits}")

    print(f"\nBackfill complete. Total credits used: {total_credits}")


def main():
    start_date = datetime.strptime(START_DATE, '%Y-%m-%d').date()
    end_date = datetime.strptime(END_DATE, '%Y-%m-%d').date()

    print(f"Loading/building schedule cache for {START_DATE} to {END_DATE}...")
    schedule_cache = load_or_build_schedule_cache(start_date, end_date)

    plan = build_plan(start_date, end_date, schedule_cache)
    print_plan_summary(plan)

    if DRY_RUN:
        print("\nDRY_RUN is True — no Odds API calls made. Set DRY_RUN = False to execute.")
        return

    confirm = input("\nType YES to proceed with live Odds API calls: ")
    if confirm.strip() != 'YES':
        print("Aborted — no calls made.")
        return

    run_backfill(plan)


if __name__ == '__main__':
    main()
