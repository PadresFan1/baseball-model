"""
Fetch 2026 MLB odds from SportsGameOdds API.
Pulls Mar 25 – May 24, 2026 (regular season only, finalized games).
Saves to historical_data/odds_2026.csv which backtest.py merges automatically.

Odds used:
  - Moneyline: pre-game closing line (most recent lastUpdatedAt < startsAt)
  - Totals line: book total (pre-game, stable)
  - Over/Under odds: pre-game juice for ROI calc

Usage: python misc_py/fetch_2026_odds.py
       python misc_py/fetch_2026_odds.py --test
"""

import requests
import json
import csv
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'C:\Users\super\baseball-model\.env')

API_KEY  = os.getenv('SPORTSGAMEODDS_KEY')
BASE_URL = 'https://api.sportsgameodds.com/v2/events'

ODD_IDS = ','.join([
    'points-home-game-ml-home',
    'points-away-game-ml-away',
    'points-all-game-ou-over',
    'points-all-game-ou-under',
])

TEAM_ID_MAP = {
    'ARIZONA_DIAMONDBACKS_MLB':    'ARI',
    'ATLANTA_BRAVES_MLB':          'ATL',
    'BALTIMORE_ORIOLES_MLB':       'BAL',
    'BOSTON_RED_SOX_MLB':          'BOS',
    'CHICAGO_CUBS_MLB':            'CHC',
    'CHICAGO_WHITE_SOX_MLB':       'CHW',
    'CINCINNATI_REDS_MLB':         'CIN',
    'CLEVELAND_GUARDIANS_MLB':     'CLE',
    'COLORADO_ROCKIES_MLB':        'COL',
    'DETROIT_TIGERS_MLB':          'DET',
    'HOUSTON_ASTROS_MLB':          'HOU',
    'KANSAS_CITY_ROYALS_MLB':      'KCR',
    'LOS_ANGELES_ANGELS_MLB':      'LAA',
    'LOS_ANGELES_DODGERS_MLB':     'LAD',
    'MIAMI_MARLINS_MLB':           'MIA',
    'MILWAUKEE_BREWERS_MLB':       'MIL',
    'MINNESOTA_TWINS_MLB':         'MIN',
    'NEW_YORK_METS_MLB':           'NYM',
    'NEW_YORK_YANKEES_MLB':        'NYY',
    'OAKLAND_ATHLETICS_MLB':       'ATH',
    'SACRAMENTO_ATHLETICS_MLB':    'ATH',
    'ATHLETICS_MLB':               'ATH',
    'PHILADELPHIA_PHILLIES_MLB':   'PHI',
    'PITTSBURGH_PIRATES_MLB':      'PIT',
    'SAN_DIEGO_PADRES_MLB':        'SDP',
    'SAN_FRANCISCO_GIANTS_MLB':    'SFG',
    'SEATTLE_MARINERS_MLB':        'SEA',
    'ST_LOUIS_CARDINALS_MLB':      'STL',
    'ST._LOUIS_CARDINALS_MLB':     'STL',
    'STLOUIS_CARDINALS_MLB':       'STL',
    'TAMPA_BAY_RAYS_MLB':          'TBR',
    'TEXAS_RANGERS_MLB':           'TEX',
    'TORONTO_BLUE_JAYS_MLB':       'TOR',
    'WASHINGTON_NATIONALS_MLB':    'WSN',
}

def normalize_team(t):
    return TEAM_ID_MAP.get(t, t)

BOOK_PREFERENCE = [
    'fanduel', 'williamhill', 'pinnacle', 'lowvig',
    'betonline', 'bovada', 'draftkings', 'caesars', 'betmgm',
]

def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return None

_GRACE = __import__('datetime').timedelta(minutes=5)

def pre_game_odds(odd_data, starts_dt, field='odds', max_odds_abs=450):
    """
    Return the value of `field` from the bookmaker whose lastUpdatedAt is
    the most recent timestamp within a 5-minute grace window of game start.

    Grace period rationale: some books post their closing line 1-3 minutes
    after the official start time due to automated triggers; for MLB the
    live line doesn't move significantly in the first 5 minutes of play.
    Magnitude cap (|odds| <= 450) filters obvious in-play blowout lines.
    """
    by_book  = odd_data.get('byBookmaker', {})
    cutoff   = starts_dt + _GRACE
    best_val  = None
    best_time = None

    ordered = list(BOOK_PREFERENCE) + [b for b in by_book if b not in BOOK_PREFERENCE]
    for book_id in ordered:
        book = by_book.get(book_id, {})
        updated = parse_dt(book.get('lastUpdatedAt'))
        if updated and updated <= cutoff:
            val = book.get(field)
            if val is not None:
                # Sanity: filter obvious in-play blowout lines
                if field == 'odds' and abs(float(val)) > max_odds_abs:
                    continue
                if best_time is None or updated > best_time:
                    best_val  = val
                    best_time = updated

    return best_val

def fetch_events(starts_after, starts_before):
    events = []
    cursor = None
    page   = 0

    import time

    while True:
        params = {
            'leagueID':     'MLB',
            'startsAfter':  starts_after,
            'startsBefore': starts_before,
            'oddID':        ODD_IDS,
            'finalized':    'true',
            'limit':        100,
        }
        if cursor:
            params['cursor'] = cursor

        for attempt in range(4):
            resp = requests.get(
                BASE_URL,
                params=params,
                headers={'x-api-key': API_KEY},
                timeout=15
            )
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  429 rate limit — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break

        data = resp.json()

        batch = data.get('data', [])
        events.extend(batch)
        page += 1
        print(f"  Page {page}: {len(batch)} events  (total: {len(events)})")

        cursor = data.get('nextCursor')
        if not cursor or not batch:
            break

        time.sleep(1.5)  # polite delay between pages

    return events

def parse_event(event):
    teams  = event.get('teams', {})
    status = event.get('status', {})
    odds   = event.get('odds', {})

    starts_at = status.get('startsAt', '')
    date      = starts_at[:10] if starts_at else ''
    starts_dt = parse_dt(starts_at)

    home_id = teams.get('home', {}).get('teamID', '')
    away_id = teams.get('away', {}).get('teamID', '')

    home_score = odds.get('points-home-game-ml-home', {}).get('score')
    away_score = odds.get('points-away-game-ml-away', {}).get('score')

    ou_over  = odds.get('points-all-game-ou-over',  {})
    ou_under = odds.get('points-all-game-ou-under', {})

    fd_home_ml = pre_game_odds(odds.get('points-home-game-ml-home', {}), starts_dt, 'odds')
    fd_away_ml = pre_game_odds(odds.get('points-away-game-ml-away', {}), starts_dt, 'odds')

    fd_total = None
    for book_id in BOOK_PREFERENCE:
        val = ou_over.get('byBookmaker', {}).get(book_id, {}).get('overUnder')
        if val is not None:
            fd_total = val
            break
    if fd_total is None:
        for book in ou_over.get('byBookmaker', {}).values():
            if book.get('overUnder') is not None:
                fd_total = book['overUnder']
                break

    fd_over_odds  = pre_game_odds(ou_over,  starts_dt, 'odds')
    fd_under_odds = pre_game_odds(ou_under, starts_dt, 'odds')

    return {
        'date':          date,
        'home_team':     normalize_team(home_id),
        'away_team':     normalize_team(away_id),
        'home_score':    home_score,
        'away_score':    away_score,
        'fd_home_ml':    fd_home_ml,
        'fd_away_ml':    fd_away_ml,
        'fd_total':      fd_total,
        'fd_over_odds':  fd_over_odds,
        'fd_under_odds': fd_under_odds,
    }

if not API_KEY:
    print("Error: SPORTSGAMEODDS_KEY not found in .env")
    sys.exit(1)

# ── Test mode ─────────────────────────────────────────────────────────────────
if '--test' in sys.argv:
    print("Fetching one 2026 event (all bookmakers)...")
    resp = requests.get(
        BASE_URL,
        params={
            'leagueID':     'MLB',
            'startsAfter':  '2026-05-01T00:00:00Z',
            'startsBefore': '2026-05-03T00:00:00Z',
            'oddID':        ODD_IDS,
            'finalized':    'true',
            'limit':        1,
        },
        headers={'x-api-key': API_KEY},
        timeout=15,
    )
    raw  = resp.json()
    data = raw.get('data', [])
    if not data:
        print("No events returned. Raw response:")
        print(json.dumps(raw, indent=2))
        sys.exit(0)

    event = data[0]
    odds  = event.get('odds', {})
    starts_at = event.get('status', {}).get('startsAt', '')

    print(f"\nEvent:    {event.get('teams', {}).get('away', {}).get('teamID')} @ "
          f"{event.get('teams', {}).get('home', {}).get('teamID')}")
    print(f"StartsAt: {starts_at}")
    print()

    for odd_id in [
        'points-home-game-ml-home', 'points-away-game-ml-away',
        'points-all-game-ou-over',  'points-all-game-ou-under',
    ]:
        o = odds.get(odd_id, {})
        print(f"{odd_id}:")
        print(f"  score:         {o.get('score')}")
        for book_id in BOOK_PREFERENCE[:3]:
            bk = o.get('byBookmaker', {}).get(book_id, {})
            if bk:
                print(f"  {book_id:<12} odds={bk.get('odds')}  "
                      f"overUnder={bk.get('overUnder')}  "
                      f"lastUpdatedAt={bk.get('lastUpdatedAt')}")
        print()

    print("=== Parsed result ===")
    print(parse_event(event))
    sys.exit(0)

# ── Full pull ─────────────────────────────────────────────────────────────────
# Pulls in two chunks to avoid cursor pagination limits on large date ranges
OUTPUT = r'C:\Users\super\baseball-model\historical_data\odds_2026.csv'
FIELDS = [
    'date', 'home_team', 'away_team', 'home_score', 'away_score',
    'fd_home_ml', 'fd_away_ml', 'fd_total', 'fd_over_odds', 'fd_under_odds',
]

print("Fetching 2026 MLB odds (Mar 25 - May 24)...")

# Pull in monthly chunks for stability
chunks = [
    ('2026-03-24T00:00:00Z', '2026-04-15T00:00:00Z'),
    ('2026-04-14T00:00:00Z', '2026-05-01T00:00:00Z'),
    ('2026-04-30T00:00:00Z', '2026-05-25T00:00:00Z'),
]

all_events = []
seen_ids   = set()
for start, end in chunks:
    print(f"\nChunk: {start[:10]} -> {end[:10]}")
    chunk_events = fetch_events(start, end)
    for e in chunk_events:
        eid = e.get('eventID') or id(e)
        if eid not in seen_ids:
            seen_ids.add(eid)
            all_events.append(e)

print(f"\nTotal unique events fetched: {len(all_events)}")

rows = [parse_event(e) for e in all_events]

valid = [r for r in rows if r['date']
         and r['home_score'] is not None and r['away_score'] is not None]

with_ml    = sum(1 for r in valid if r['fd_home_ml'] and r['fd_away_ml'])
with_total = sum(1 for r in valid if r['fd_total'])
print(f"Valid rows (with scores):  {len(valid)}")
print(f"  With pre-game ML odds:   {with_ml}  ← these are usable for the 2026_validation backtest")
print(f"  With O/U total line:     {with_total}")

unknown = set()
for r in rows:
    for col in ['home_team', 'away_team']:
        val = r[col]
        if val and '_MLB' in val:
            unknown.add(val)
if unknown:
    print(f"\nUnmapped team IDs (add to TEAM_ID_MAP): {unknown}")

rows_sorted = sorted(valid, key=lambda r: r['date'])
with open(OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows_sorted)

print(f"\nSaved {len(valid)} games to {OUTPUT}")
print(f"ML odds coverage: {with_ml}/{len(valid)} = {with_ml/len(valid)*100:.1f}%" if valid else "")
print("Run backtest.py with RUN_MODE='2026_validation' to evaluate.")
