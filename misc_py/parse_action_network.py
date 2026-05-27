"""
Parse Action Network odds exports (5.21 - 5.25) and merge into
historical_data/odds_2026_complete.csv.

Format notes (reverse-engineered from the files):
  Each game occupies 9 rows:
    Row 1  "[Away] Team Icon"  — away team odds across books
    Row 2  "[Away]"            — home team odds (reuses away team label; Action Network quirk)
    Row 3  rotation (odd)      — away team's rotation number (odd = away in MLB convention)
    Row 4  score               — away team's final score
    Row 5  "[Home] Team Icon"  — home team icon (no odds, just the name)
    Row 6  "[Home]"            — home team (no odds)
    Row 7  rotation (even)     — home team's rotation number
    Row 8  score               — home team's final score
    Row 9  status              — "Final", "Final - N", "PPD", time, or in-progress text

  Column layout (0-indexed):
    0  = team name / status text
    1  = Opening ML for row's team
    2  = Best Odds (number + book name concatenated, e.g. "+155bet365 CO Logo")
    3  = DK CO
    4  = FanDuel CO   ← primary source
    5  = BetMGM CO
    6  = DK NJ
    7  = bet365 CO
    8  = BetRivers CO
    9  = (blank)
    10 = Fanatics CO
    11 = Caesars CO
    12 = bet365 NJ

  Odds preference for each team: FanDuel (4) → DK (3) → bet365 (7) → Open (1)
  For in-progress games: fall back to Open (1) to avoid live-line contamination.
"""

import os
import sys
import re
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FOLDER        = os.path.join(os.path.dirname(__file__), '..', '5.21 - 5.25 odds')
COMPLETE_PATH = os.path.join(os.path.dirname(__file__), '..', 'historical_data', 'odds_2026_complete.csv')
FOLDER        = os.path.normpath(FOLDER)
COMPLETE_PATH = os.path.normpath(COMPLETE_PATH)

# Map Action Network team nicknames -> FanGraphs abbreviations
ACTION_TO_FG = {
    'Guardians':    'CLE',  'Tigers':        'DET',  'Pirates':      'PIT',
    'Cardinals':    'STL',  'Mets':          'NYM',  'Nationals':    'WSN',
    'Braves':       'ATL',  'Marlins':       'MIA',  'Blue Jays':    'TOR',
    'Yankees':      'NYY',  'Athletics':     'ATH',  'Angels':       'LAA',
    'Rockies':      'COL',  'Diamondbacks':  'ARI',  'Padres':       'SDP',
    'Cubs':         'CHC',  'White Sox':     'CHW',  'Giants':       'SFG',
    'Orioles':      'BAL',  'Astros':        'HOU',  'Rangers':      'TEX',
    'Mariners':     'SEA',  'Royals':        'KCR',  'Twins':        'MIN',
    'Red Sox':      'BOS',  'Rays':          'TBR',  'Phillies':     'PHI',
    'Brewers':      'MIL',  'Reds':          'CIN',  'Dodgers':      'LAD',
}

# Files and corresponding 2026 dates
FILES = [
    ('5.21.csv',  '2026-05-21', 'csv'),
    ('5.22.xlsx', '2026-05-22', 'xlsx'),
    ('5.23.csv',  '2026-05-23', 'csv'),
    ('5.24.csv',  '2026-05-24', 'csv'),
    ('5.25.csv',  '2026-05-25', 'csv'),
]

PREF_COLS = [4, 3, 7, 1]   # FanDuel → DK → bet365 → Open


def to_str(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    return str(val).strip()


def parse_ml(val):
    """Extract integer American-format ML from a cell. Returns None if invalid."""
    s = to_str(val)
    if s.upper() in ('N/A', 'NAN', ''):
        return None
    m = re.match(r'^([+-]?\d+)', s)
    if m:
        v = int(m.group(1))
        if 80 <= abs(v) <= 3000:
            return v
    return None


def get_ml(row, prefer_open=False):
    """
    Extract preferred ML from a row list.
    prefer_open=True for in-progress games to avoid live-line contamination.
    """
    cols = [1] + PREF_COLS if prefer_open else PREF_COLS
    for c in cols:
        if c < len(row):
            v = parse_ml(row[c])
            if v is not None:
                return v
    return None


def has_odds(row):
    """True if row has at least one numeric ML in the data columns."""
    for c in PREF_COLS:
        if c < len(row) and parse_ml(row[c]) is not None:
            return True
    return False


def is_rotation(val):
    """True if val is a 3-digit integer in MLB rotation range (900-999)."""
    s = to_str(val)
    if re.fullmatch(r'\d{3}', s):
        v = int(s)
        return 900 <= v <= 999
    return False


def is_score(val):
    """True if val could be a game score (0-30)."""
    s = to_str(val)
    if re.fullmatch(r'\d{1,2}', s):
        return 0 <= int(s) <= 30
    return False


def is_status(val):
    """True if val looks like a game status / scheduled-time line."""
    s = to_str(val).lower()
    return (
        s.startswith('final') or
        s == 'ppd' or
        re.search(r'\d+:\d+\s*(am|pm)', s) is not None or
        any(k in s for k in ('th:', 'st:', 'nd:', 'rd:', 'bot ', 'top ', 'mid '))
    )


def status_is_live(status):
    """True if game was in-progress when the file was captured."""
    s = to_str(status).lower()
    return any(k in s for k in ('th:', 'st:', 'nd:', 'rd:', 'bot ', 'top ', 'mid '))


def parse_file(path, fmt):
    """Read file into a list-of-lists (each row is a list of string values)."""
    if fmt == 'xlsx':
        df = pd.read_excel(path, header=None, dtype=str)
        # The xlsx has no separate header row — all rows are data
        rows = df.values.tolist()
    else:
        df = pd.read_csv(path, header=0, dtype=str)
        rows = df.values.tolist()
    return rows


def parse_games(rows):
    """
    Parse a list of row-lists into game records.
    Returns list of dicts: away_name, home_name, away_ml, home_ml,
                           away_score, home_score, status
    """
    games = []
    i = 0
    while i < len(rows):
        col0 = to_str(rows[i][0])

        # Detect start of a game block: "X Team Icon" row WITH odds
        if 'Team Icon' in col0 and has_odds(rows[i]):
            away_name = col0.replace(' Team Icon', '').strip()
            away_row  = rows[i]

            # Row immediately following = home team odds row
            home_row = rows[i + 1] if i + 1 < len(rows) else []
            i += 2

            # Scan forward within this game block
            away_rot = away_score = None
            home_rot = home_score = None
            home_name = None
            status    = ''

            while i < len(rows):
                c0 = to_str(rows[i][0])

                if 'Team Icon' in c0 and not has_odds(rows[i]):
                    # Home team block (no odds — name only)
                    home_name = c0.replace(' Team Icon', '').strip()
                    i += 2   # skip home icon + label
                    continue

                if is_rotation(rows[i][0]):
                    rot = int(to_str(rows[i][0]))
                    # Score is in the very next row's col[0]
                    score_val = rows[i + 1][0] if i + 1 < len(rows) else None
                    sc = int(to_str(score_val)) if is_score(score_val) else None
                    if rot % 2 == 1:   # odd = away
                        away_rot, away_score = rot, sc
                    else:              # even = home
                        home_rot, home_score = rot, sc
                    i += 2
                    continue

                if is_status(rows[i][0]):
                    status = c0
                    i += 1
                    break

                # Empty row or unknown — end of game block
                if c0 == '':
                    i += 1
                    break

                i += 1

            # Skip postponed games
            if 'ppd' in status.lower():
                continue

            # Use Open ML if game was live when captured
            live = status_is_live(status)
            away_ml = get_ml(away_row, prefer_open=live)
            home_ml = get_ml(home_row, prefer_open=live)

            if away_ml is None or home_ml is None:
                continue

            games.append({
                'away_name':  away_name,
                'home_name':  home_name,
                'away_ml':    away_ml,
                'home_ml':    home_ml,
                'away_score': away_score,
                'home_score': home_score,
                'status':     status,
            })

        else:
            i += 1

    return games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_games = []
for fname, date_str, fmt in FILES:
    fpath = os.path.join(FOLDER, fname)
    if not os.path.exists(fpath):
        print(f"  {fname}: NOT FOUND — skipped")
        continue

    rows  = parse_file(fpath, fmt)
    games = parse_games(rows)
    valid = 0
    for g in games:
        away_fg = ACTION_TO_FG.get(g['away_name'])
        home_fg = ACTION_TO_FG.get(g['home_name'])
        if not away_fg or not home_fg:
            if g['away_name'] or g['home_name']:
                print(f"  [{date_str}] Unmapped: '{g['away_name']}' @ '{g['home_name']}'")
            continue
        all_games.append({
            'date':       date_str,
            'home_team':  home_fg,
            'away_team':  away_fg,
            'fd_home_ml': g['home_ml'],
            'fd_away_ml': g['away_ml'],
            'home_score': g['home_score'],
            'away_score': g['away_score'],
            'status':     g['status'],
        })
        valid += 1
    print(f"  {fname}: {len(games)} game blocks -> {valid} valid ({len(games)-valid} skipped)")

print(f"\nTotal games parsed: {len(all_games)}")

# Sanity check: print all games
rows_out = sorted(all_games, key=lambda r: (r['date'], r['home_team']))
print(f"\n{'Date':<12} {'Home':<5} {'Away':<5} {'Home ML':>8} {'Away ML':>8} {'Score':<10} Status")
print('-' * 70)
for r in rows_out:
    score = f"{r['home_score']}-{r['away_score']}" if r['home_score'] is not None else '?-?'
    print(f"  {r['date']:<10} {r['home_team']:<5} {r['away_team']:<5} "
          f"{r['fd_home_ml']:>8} {r['fd_away_ml']:>8} {score:<10} {r['status']}")

# ---------------------------------------------------------------------------
# Merge into odds_2026_complete.csv
# ---------------------------------------------------------------------------

comp_df = pd.read_csv(COMPLETE_PATH, dtype=str)
print(f"\nLoaded odds_2026_complete.csv: {len(comp_df)} rows")

# Build update lookup: (date, home_team, away_team) -> (home_ml, away_ml)
lookup = {}
for g in all_games:
    k = (g['date'], g['home_team'], g['away_team'])
    lookup[k] = (str(g['fd_home_ml']), str(g['fd_away_ml']))

updated = 0
for idx, row in comp_df.iterrows():
    k = (to_str(row['date']), to_str(row['home_team']), to_str(row['away_team']))
    if k not in lookup:
        continue
    # Always overwrite with Action Network data — it's actual FD closing lines,
    # which are more authoritative than the approximated odds from predictions_log.
    hml, aml = lookup[k]
    comp_df.at[idx, 'fd_home_ml'] = hml
    comp_df.at[idx, 'fd_away_ml'] = aml
    updated += 1

# Append any games from AN that aren't in comp_df yet (e.g. May 25 unfinalized)
existing_keys = set(
    (to_str(r['date']), to_str(r['home_team']), to_str(r['away_team']))
    for _, r in comp_df.iterrows()
)
appended = 0
new_rows = []
for g in all_games:
    k = (g['date'], g['home_team'], g['away_team'])
    if k not in existing_keys:
        new_rows.append({
            'date':          g['date'],
            'home_team':     g['home_team'],
            'away_team':     g['away_team'],
            'home_score':    str(g['home_score']) if g['home_score'] is not None else '',
            'away_score':    str(g['away_score']) if g['away_score'] is not None else '',
            'fd_home_ml':    str(g['fd_home_ml']),
            'fd_away_ml':    str(g['fd_away_ml']),
            'fd_total':      '',
            'fd_over_odds':  '',
            'fd_under_odds': '',
        })
        existing_keys.add(k)
        appended += 1

if new_rows:
    comp_df = pd.concat([comp_df, pd.DataFrame(new_rows)], ignore_index=True)

comp_df = comp_df.sort_values('date').reset_index(drop=True)
comp_df.to_csv(COMPLETE_PATH, index=False)

ml_ok = comp_df[['fd_home_ml', 'fd_away_ml']].apply(
    lambda c: pd.to_numeric(c, errors='coerce')
).notna().all(axis=1).sum()

print(f"\nMerge summary:")
print(f"  Updated (AN closing lines) : {updated}")
print(f"  Appended (new rows)        : {appended}")
print(f"\nFinal odds_2026_complete.csv:")
print(f"  Total rows     : {len(comp_df)}")
print(f"  With ML odds   : {ml_ok}  <- backtest uses these")
print(f"  Without ML odds: {len(comp_df) - ml_ok}")
print(f"\nNext step: python backtest.py  (RUN_MODE='2026_validation')")
