"""
Fetch remaining 2025 MLB odds from SportsGameOdds API.
Pulls Aug 17 - Oct 1, 2025 (regular season only).
Saves to historical_data/odds_2025_supplement.csv which the backtest merges automatically.

Odds used:
  - Moneyline: closeBookOdds (consensus pre-game closing line, not live/in-play)
  - Totals line: byBookmaker.fanduel.overUnder (pre-game line, frozen at game start)
  - Over/Under odds: closeBookOdds for each side

Usage: python fetch_2025_odds.py
       python fetch_2025_odds.py --test    (prints one raw event and exits)
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

# No bookmakerID filter — return all books to maximize pre-game timestamp coverage.
# Credits are per-event not per-bookmaker, so this costs nothing extra.

# SportsGameOdds teamID format → FG abbreviation
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

def pre_game_odds(odd_data, starts_dt, field='odds'):
    """
    Return the value of `field` from the bookmaker whose lastUpdatedAt
    is the most recent timestamp still before game start.
    Prefers FanDuel, then falls back to any pre-game book.
    """
    by_book = odd_data.get('byBookmaker', {})
    best_val  = None
    best_time = None

    # Walk preferred books first so FanDuel wins ties
    ordered = list(BOOK_PREFERENCE) + [b for b in by_book if b not in BOOK_PREFERENCE]
    for book_id in ordered:
        book = by_book.get(book_id, {})
        updated = parse_dt(book.get('lastUpdatedAt'))
        if updated and updated <= starts_dt:
            if best_time is None or updated > best_time:
                val = book.get(field)
                if val is not None:
                    best_val  = val
                    best_time = updated

    return best_val

def fetch_events(starts_after, starts_before):
    events = []
    cursor = None
    page   = 0

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

        resp = requests.get(
            BASE_URL,
            params=params,
            headers={'x-api-key': API_KEY},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        batch = data.get('data', [])
        events.extend(batch)
        page += 1
        print(f"  Page {page}: {len(batch)} events  (total: {len(events)})")

        cursor = data.get('nextCursor')
        if not cursor or not batch:
            break

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

    # Scores are stored in the oddID score field — always correct for finalized events
    home_score = odds.get('points-home-game-ml-home', {}).get('score')
    away_score = odds.get('points-away-game-ml-away', {}).get('score')

    # Moneyline: find the bookmaker whose lastUpdatedAt is most recent BEFORE game start.
    # closeBookOdds and byBookmaker.fanduel.odds both reflect in-game values for blowouts.
    ou_over  = odds.get('points-all-game-ou-over',  {})
    ou_under = odds.get('points-all-game-ou-under', {})

    fd_home_ml = pre_game_odds(odds.get('points-home-game-ml-home', {}), starts_dt, 'odds')
    fd_away_ml = pre_game_odds(odds.get('points-away-game-ml-away', {}), starts_dt, 'odds')

    # Total line is stable pre-game — take from any bookmaker, no timestamp filter needed
    fd_total = None
    for book_id in BOOK_PREFERENCE:
        val = ou_over.get('byBookmaker', {}).get(book_id, {}).get('overUnder')
        if val is not None:
            fd_total = val
            break
    if fd_total is None:  # try any book
        for book in ou_over.get('byBookmaker', {}).values():
            if book.get('overUnder') is not None:
                fd_total = book['overUnder']
                break

    # Juice: only use actual pre-game value — None if unavailable (ROI skipped for those games)
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
    print("Fetching one event (all bookmakers)...")
    resp = requests.get(
        BASE_URL,
        params={
            'leagueID':     'MLB',
            'startsAfter':  '2025-08-16T00:00:00Z',
            'startsBefore': '2025-08-18T00:00:00Z',
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
        for book_id in ['fanduel', 'williamhill']:
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
print("Fetching 2025 MLB odds (Aug 17 – Oct 1)...")
events = fetch_events('2025-08-16T00:00:00Z', '2025-10-02T00:00:00Z')
print(f"\nTotal events fetched: {len(events)}")

rows = [parse_event(e) for e in events]

# Require: date + actual scores (for result evaluation)
# ML odds optional — only 28/608 games have pre-game ML data from this API
# O/U total line is reliable for all games (set pre-game, doesn't move in-game)
valid = [r for r in rows if r['date']
         and r['home_score'] is not None and r['away_score'] is not None]

with_ml    = sum(1 for r in valid if r['fd_home_ml'] and r['fd_away_ml'])
with_total = sum(1 for r in valid if r['fd_total'])
print(f"Valid rows (with scores):  {len(valid)}")
print(f"  With pre-game ML odds:   {with_ml}  (only these contribute to ML backtesting)")
print(f"  With O/U total line:     {with_total}  (all contribute to O/U threshold optimization)")

# Log any unmapped team IDs so we can add them
unknown = set()
for r in rows:
    for col in ['home_team', 'away_team']:
        val = r[col]
        if val and '_MLB' in val:  # still in raw format = unmapped
            unknown.add(val)
if unknown:
    print(f"\nUnmapped team IDs (add to TEAM_ID_MAP): {unknown}")

output = r'C:\Users\super\baseball-model\historical_data\odds_2025_supplement.csv'
fields = [
    'date', 'home_team', 'away_team', 'home_score', 'away_score',
    'fd_home_ml', 'fd_away_ml', 'fd_total', 'fd_over_odds', 'fd_under_odds',
]

with open(output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(valid)

print(f"\nSaved {len(valid)} games to {output}")
print("Run the backtest — it will merge this file automatically.")
