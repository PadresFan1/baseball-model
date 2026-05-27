"""
Download 2026 MLB historical odds + final scores.

Sources tried in priority order
--------------------------------
1. SportsBookReviewOnline (SBR) Excel archive
   URL:  https://www.sportsbookreviewonline.com/scoresoddsarchives/mlb/mlbodds2026.xlsx
   Format: paired rows (visitor / home), columns Date|Rot|VH|Team|...|Final|Open|Close|ML

2. GitHub CSV mirror of SBR data (community-maintained, updated through active seasons)
   Tried if SBR download returns 404 / is empty.

3. MLB Stats API (statsapi) — final scores only, no odds
   Always succeeds. Outputs scores with empty ML columns.
   Use as a last resort to confirm the row count and test the pipeline;
   fill odds manually or from another source before running the backtest.

Output
------
  historical_data/odds_2026_complete.csv
  Columns (schema): date, home_team, away_team, home_score, away_score,
                    fd_home_ml, fd_away_ml, fd_total, fd_over_odds, fd_under_odds

Team abbreviations: FanGraphs standard (ARI ATL BAL BOS CHC CHW CIN CLE COL DET
                    HOU KCR LAA LAD MIA MIL MIN NYM NYY ATH PHI PIT SDP SFG SEA
                    STL TBR TEX TOR WSN)

Usage
-----
  python misc_py/download_sbr_2026.py            # full pipeline, writes CSV
  python misc_py/download_sbr_2026.py --schema   # print schema + manual-entry guide only
  python misc_py/download_sbr_2026.py --scores   # MLB Stats API scores only (no odds)
  python misc_py/download_sbr_2026.py --validate # validate + normalize an existing CSV
"""

import sys
import os
import io
import time
import requests
import pandas as pd
import statsapi
from datetime import date, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SEASON_START = date(2026, 3, 25)
SEASON_END   = date(2026, 5, 24)
OUTPUT       = os.path.join(os.path.dirname(__file__), '..', 'historical_data', 'odds_2026_complete.csv')
OUTPUT       = os.path.normpath(OUTPUT)

FIELDS = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
          'fd_home_ml', 'fd_away_ml', 'fd_total', 'fd_over_odds', 'fd_under_odds']

# ---------------------------------------------------------------------------
# Team name -> FanGraphs abbreviation maps
# ---------------------------------------------------------------------------

# SBR Excel format (partial city/nickname patterns used by SBR)
SBR_TO_FG = {
    'Arizona':        'ARI', 'Diamondbacks':       'ARI',
    'Atlanta':        'ATL', 'Braves':             'ATL',
    'Baltimore':      'BAL', 'Orioles':            'BAL',
    'Boston':         'BOS', 'Red Sox':            'BOS',
    'Chi Cubs':       'CHC', 'Cubs':               'CHC',
    'Chi Sox':        'CHW', 'White Sox':          'CHW', 'Chicago W Sox': 'CHW',
    'Cincinnati':     'CIN', 'Reds':               'CIN',
    'Cleveland':      'CLE', 'Guardians':          'CLE',
    'Colorado':       'COL', 'Rockies':            'COL',
    'Detroit':        'DET', 'Tigers':             'DET',
    'Houston':        'HOU', 'Astros':             'HOU',
    'Kansas City':    'KCR', 'Royals':             'KCR',
    'LA Angels':      'LAA', 'Angels':             'LAA', 'Anaheim': 'LAA', 'Los Angeles Angels': 'LAA',
    'LA Dodgers':     'LAD', 'Dodgers':            'LAD', 'Los Angeles Dodgers': 'LAD',
    'Miami':          'MIA', 'Marlins':            'MIA',
    'Milwaukee':      'MIL', 'Brewers':            'MIL',
    'Minnesota':      'MIN', 'Twins':              'MIN',
    'NY Mets':        'NYM', 'Mets':               'NYM', 'New York Mets': 'NYM',
    'NY Yankees':     'NYY', 'Yankees':            'NYY', 'New York Yankees': 'NYY',
    'Oakland':        'ATH', 'Las Vegas':          'ATH', 'Sacramento':    'ATH',
    'Athletics':      'ATH', 'Las Vegas Athletics':'ATH', 'Sacramento Athletics': 'ATH',
    'Oakland Athletics': 'ATH',
    'Philadelphia':   'PHI', 'Phillies':           'PHI',
    'Pittsburgh':     'PIT', 'Pirates':            'PIT',
    'San Diego':      'SDP', 'Padres':             'SDP',
    'San Francisco':  'SFG', 'Giants':             'SFG',
    'Seattle':        'SEA', 'Mariners':           'SEA',
    'St. Louis':      'STL', 'St Louis':           'STL', 'Cardinals': 'STL',
    'Tampa Bay':      'TBR', 'Rays':               'TBR',
    'Texas':          'TEX', 'Rangers':            'TEX',
    'Toronto':        'TOR', 'Blue Jays':          'TOR',
    'Washington':     'WSN', 'Nationals':          'WSN',
}

# Already-correct FG abbreviations pass through normalize unchanged
FG_ABBREVS = {
    'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
    'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','ATH',
    'PHI','PIT','SDP','SFG','SEA','STL','TBR','TEX','TOR','WSN',
}

# MLB Stats API team ID -> FG abbreviation
MLB_ID_TO_FG = {
    133:'ATH', 134:'PIT', 135:'SDP', 136:'SEA', 137:'SFG', 138:'STL',
    139:'TBR', 140:'TEX', 141:'TOR', 142:'MIN', 143:'PHI', 144:'ATL',
    145:'CHW', 146:'MIA', 147:'NYY', 158:'MIL', 108:'LAA', 109:'ARI',
    110:'BAL', 111:'BOS', 112:'CHC', 113:'CIN', 114:'CLE', 115:'COL',
    116:'DET', 117:'HOU', 118:'KCR', 119:'LAD', 120:'WSN', 121:'NYM',
}

def norm(name):
    if not name:
        return None
    s = str(name).strip()
    if s in FG_ABBREVS:
        return s
    return SBR_TO_FG.get(s) or SBR_TO_FG.get(s.title())


# ---------------------------------------------------------------------------
# Source 1 — SportsBookReviewOnline Excel
# ---------------------------------------------------------------------------

SBR_URLS = [
    'https://www.sportsbookreviewonline.com/scoresoddsarchives/mlb/mlbodds2026.xlsx',
    'https://sportsbookreviewonline.com/scoresoddsarchives/mlb/mlbodds2026.xlsx',
]

def try_sbr_excel():
    """Download and parse the SBR MLB 2026 Excel file. Returns DataFrame or None."""
    for url in SBR_URLS:
        try:
            print(f"  Trying SBR Excel: {url}")
            resp = requests.get(url, timeout=20,
                                headers={'User-Agent': 'Mozilla/5.0 (compatible; research-bot)'})
            if resp.status_code == 200 and len(resp.content) > 5000:
                print(f"  Downloaded {len(resp.content):,} bytes")
                return parse_sbr_excel(resp.content)
            else:
                print(f"  HTTP {resp.status_code} or empty response ({len(resp.content)} bytes)")
        except Exception as e:
            print(f"  Error: {e}")
    return None

def parse_sbr_excel(content):
    """
    Parse SBR Excel bytes into a normalized DataFrame.

    SBR layout (two rows per game, visitor then home):
      Date | Rot | VH | Team | <inning cols> | Final | Open | Close | ML | 2H
    ML column = moneyline for that team (American format, e.g. -145, +135)
    """
    try:
        raw = pd.read_excel(io.BytesIO(content), header=0, dtype=str)
    except Exception as e:
        print(f"  Excel parse error: {e}")
        return None

    print(f"  Raw Excel: {len(raw)} rows x {len(raw.columns)} cols")
    print(f"  Columns: {list(raw.columns)}")

    # Identify key columns flexibly (SBR column names vary slightly by year)
    cols = {c.strip().upper(): c for c in raw.columns}

    def find_col(*candidates):
        for c in candidates:
            if c.upper() in cols:
                return cols[c.upper()]
        return None

    col_date  = find_col('DATE', 'Gametime', 'GAME DATE')
    col_vh    = find_col('VH', 'V/H', 'VISITOR/HOME')
    col_team  = find_col('TEAM', 'TEAMS')
    col_final = find_col('FINAL', 'SCORE', 'F')
    col_ml    = find_col('ML', 'MONEY LINE', 'MONEYLINE', 'OPEN ML')
    col_open  = find_col('OPEN', 'OPENING')
    col_close = find_col('CLOSE', 'CLOSING', 'TOTAL')

    missing = [n for n, c in [('date',col_date),('vh',col_vh),('team',col_team),
                                ('final',col_final),('ml',col_ml)] if c is None]
    if missing:
        print(f"  WARNING: Could not find columns: {missing}")
        print("  Full column list:", list(raw.columns))
        return None

    records = []
    i = 0
    while i + 1 < len(raw):
        v_row = raw.iloc[i]
        h_row = raw.iloc[i + 1]

        # Confirm pairing: one should be V (visitor) and next H (home)
        vh_v = str(v_row.get(col_vh, '')).strip().upper()
        vh_h = str(h_row.get(col_vh, '')).strip().upper()
        if 'V' not in vh_v or 'H' not in vh_h:
            i += 1
            continue

        # Date from visitor row (home row often blank for date)
        raw_date = str(v_row.get(col_date, '')).strip()
        game_date = parse_sbr_date(raw_date)
        if game_date is None:
            i += 2
            continue

        away_name = str(v_row.get(col_team, '')).strip()
        home_name = str(h_row.get(col_team, '')).strip()
        away_fg   = norm(away_name)
        home_fg   = norm(home_name)

        if not away_fg or not home_fg:
            print(f"  Unmapped: '{away_name}' / '{home_name}' on {game_date}")
            i += 2
            continue

        try:
            away_score = int(float(str(v_row.get(col_final, ''))))
            home_score = int(float(str(h_row.get(col_final, ''))))
        except Exception:
            i += 2
            continue

        def to_ml(val):
            try:
                v = int(float(str(val).replace('pk', '100').replace('PK', '100')))
                return v if v != 0 else None
            except Exception:
                return None

        away_ml = to_ml(v_row.get(col_ml))
        home_ml = to_ml(h_row.get(col_ml))

        total = None
        if col_close:
            try:
                total = float(str(h_row.get(col_close, '')).replace('o', '').replace('u', ''))
            except Exception:
                pass

        # Only keep complete regular-season games before cutoff
        if SEASON_START <= game_date <= SEASON_END:
            records.append({
                'date':       game_date.strftime('%Y-%m-%d'),
                'home_team':  home_fg,
                'away_team':  away_fg,
                'home_score': home_score,
                'away_score': away_score,
                'fd_home_ml': home_ml,
                'fd_away_ml': away_ml,
                'fd_total':   total,
                'fd_over_odds':  None,
                'fd_under_odds': None,
            })

        i += 2

    df = pd.DataFrame(records)
    if df.empty:
        print("  No valid rows parsed from SBR Excel.")
        return None
    ml_count = df[['fd_home_ml','fd_away_ml']].notna().all(axis=1).sum()
    print(f"  Parsed {len(df)} games ({ml_count} with ML odds) from SBR Excel.")
    return df


def parse_sbr_date(raw):
    """Convert SBR date strings (MMDD, YYYYMMDD, M/D/YYYY, etc.) to date object."""
    raw = str(raw).strip().split(' ')[0]
    for fmt in ('%Y%m%d', '%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y'):
        try:
            return pd.to_datetime(raw, format=fmt).date()
        except Exception:
            pass
    # Fallback: 4-digit MMDD (SBR sometimes uses this for non-Jan dates)
    if raw.isdigit() and len(raw) == 4:
        try:
            mo, dy = int(raw[:2]), int(raw[2:])
            return date(2026, mo, dy)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Source 2 — GitHub CSV mirrors
# ---------------------------------------------------------------------------

GITHUB_URLS = [
    # Community-maintained SBR mirrors — add URLs here if you know of one
    # Example pattern (fictional, update if a real repo is found):
    # 'https://raw.githubusercontent.com/sport-betting-data/mlb-historical/main/2026.csv',
]

def try_github_csv():
    for url in GITHUB_URLS:
        try:
            print(f"  Trying GitHub mirror: {url}")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                df = normalize_generic_csv(df)
                if df is not None and len(df) > 0:
                    return df
        except Exception as e:
            print(f"  Error: {e}")
    return None


def normalize_generic_csv(df):
    """
    Attempt to normalize a generic odds CSV to our schema.
    Looks for common column name patterns.
    """
    col_map = {}
    cols_upper = {c.upper().strip(): c for c in df.columns}

    candidates = {
        'date':       ['DATE', 'GAME DATE', 'GAMEDATE'],
        'home_team':  ['HOME', 'HOME_TEAM', 'HOMETEAM', 'HOME TEAM'],
        'away_team':  ['AWAY', 'AWAY_TEAM', 'AWAYTEAM', 'AWAY TEAM', 'VISITOR'],
        'home_score': ['HOME_SCORE', 'HOMESCORE', 'HOME SCORE', 'HOME FINAL', 'HSCORE'],
        'away_score': ['AWAY_SCORE', 'AWAYSCORE', 'AWAY SCORE', 'AWAY FINAL', 'VSCORE'],
        'fd_home_ml': ['HOME_ML', 'HOMEML', 'HOME ML', 'HOME MONEY', 'ML_HOME'],
        'fd_away_ml': ['AWAY_ML', 'AWAYML', 'AWAY ML', 'AWAY MONEY', 'ML_AWAY'],
        'fd_total':   ['TOTAL', 'OU_LINE', 'OVER_UNDER', 'CLOSE_TOTAL'],
    }

    for field, options in candidates.items():
        for opt in options:
            if opt in cols_upper:
                col_map[field] = cols_upper[opt]
                break

    required = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    if any(f not in col_map for f in required):
        return None

    out = pd.DataFrame()
    for field in FIELDS:
        if field in col_map:
            out[field] = df[col_map[field]]
        else:
            out[field] = None

    out['date']      = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
    out['home_team'] = out['home_team'].apply(norm)
    out['away_team'] = out['away_team'].apply(norm)
    out = out.dropna(subset=['home_team', 'away_team'])
    out = out[(out['date'] >= SEASON_START.strftime('%Y-%m-%d')) &
              (out['date'] <= SEASON_END.strftime('%Y-%m-%d'))]
    return out if len(out) > 0 else None


# ---------------------------------------------------------------------------
# Source 3 — MLB Stats API (scores only, no odds)
# ---------------------------------------------------------------------------

def pull_mlb_scores():
    """
    Pull all 2026 final scores via statsapi.schedule().
    Returns DataFrame with scores but empty ML columns.
    """
    print(f"\nSource 3: MLB Stats API (scores only)")
    records = []
    cur = SEASON_START
    while cur <= SEASON_END:
        date_str = cur.strftime('%Y-%m-%d')
        try:
            games = statsapi.schedule(
                start_date=date_str, end_date=date_str, sportId=1
            )
            for g in games:
                if g.get('status') != 'Final':
                    continue
                if g.get('game_type') != 'R':
                    continue
                home_fg = MLB_ID_TO_FG.get(g.get('home_id'))
                away_fg = MLB_ID_TO_FG.get(g.get('away_id'))
                if not home_fg or not away_fg:
                    continue
                records.append({
                    'date':          date_str,
                    'home_team':     home_fg,
                    'away_team':     away_fg,
                    'home_score':    g.get('home_score'),
                    'away_score':    g.get('away_score'),
                    'fd_home_ml':    None,
                    'fd_away_ml':    None,
                    'fd_total':      None,
                    'fd_over_odds':  None,
                    'fd_under_odds': None,
                })
        except Exception as e:
            print(f"  {date_str}: error — {e}")
        time.sleep(0.15)
        cur += timedelta(days=1)

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"  Pulled {len(df)} finalized regular-season games (no odds)")
    return df


# ---------------------------------------------------------------------------
# Schema reference — for manual data entry
# ---------------------------------------------------------------------------

SCHEMA_GUIDE = """
================================================================================
  odds_2026_complete.csv  —  MANUAL ENTRY SCHEMA
================================================================================

File location: historical_data/odds_2026_complete.csv
Required columns (comma-separated, header on row 1):

  date          YYYY-MM-DD          2026-03-27
  home_team     FanGraphs abbrev    NYY
  away_team     FanGraphs abbrev    BOS
  home_score    integer             8
  away_score    integer             3
  fd_home_ml    American odds       -145
  fd_away_ml    American odds       +125
  fd_total      decimal line        8.5     (optional — leave blank if unknown)
  fd_over_odds  American odds       -110    (optional)
  fd_under_odds American odds       -110    (optional)

Team abbreviations (FanGraphs standard):
  ARI  ATL  BAL  BOS  CHC  CHW  CIN  CLE  COL  DET
  HOU  KCR  LAA  LAD  MIA  MIL  MIN  NYM  NYY  ATH
  PHI  PIT  SDP  SFG  SEA  STL  TBR  TEX  TOR  WSN

Note: ATH = Athletics (Sacramento/Las Vegas — formerly OAK)

Moneyline format: American (e.g., -145 means bet $145 to win $100;
                  +125 means bet $100 to win $125).
  Use the CLOSING line (line immediately before first pitch).
  Any major book (FanDuel, DraftKings, Pinnacle, BetOnline) is acceptable.
  ML for both teams required for each game to be included in the backtest.

Minimal example row:
  2026-03-27,NYY,BOS,8,3,-145,+125,,,
  (fd_total / fd_over_odds / fd_under_odds can be empty)

Example data sources for closing lines:
  * covers.com — https://www.covers.com/sport/baseball/mlb/scores/YYYY-MM-DD
  * action.com  — https://www.actionnetwork.com/mlb/odds
  * oddsportal  — https://www.oddsportal.com/baseball/usa/mlb/results/
  * SBR archive — https://www.sportsbookreview.com/betting-odds/mlb-odds/results/
================================================================================
"""


# ---------------------------------------------------------------------------
# Validation / normalization of an existing CSV
# ---------------------------------------------------------------------------

def validate_csv(path):
    """Load, normalize, and report on an existing odds_2026_complete.csv."""
    print(f"\nValidating: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  Cannot read file: {e}")
        return

    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

    missing_cols = [f for f in FIELDS if f not in df.columns]
    if missing_cols:
        print(f"  MISSING columns: {missing_cols}")
        return

    # Normalize team names in place
    df['home_team'] = df['home_team'].apply(norm)
    df['away_team'] = df['away_team'].apply(norm)
    bad_teams = df[df['home_team'].isna() | df['away_team'].isna()]
    if len(bad_teams):
        print(f"  WARNING: {len(bad_teams)} rows with unrecognized team names:")
        print(bad_teams[['date','home_team','away_team']].to_string())

    df_clean = df.dropna(subset=['home_team','away_team'])
    ml_ok = df_clean[['fd_home_ml','fd_away_ml']].notna().all(axis=1)
    print(f"\n  Total rows           : {len(df)}")
    print(f"  Valid team names     : {len(df_clean)}")
    print(f"  With ML odds         : {ml_ok.sum()} (these are usable for the 2026_validation backtest)")
    print(f"  Without ML odds      : {(~ml_ok).sum()} (scores only — will be skipped)")
    if 'date' in df_clean.columns:
        dates = pd.to_datetime(df_clean['date'], errors='coerce')
        print(f"  Date range           : {dates.min().date()} -> {dates.max().date()}")

    # Save normalized version back
    df_clean[FIELDS].to_csv(path, index=False)
    print(f"\n  Saved normalized file to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save(df):
    df = df.sort_values('date').reset_index(drop=True)
    df[FIELDS].to_csv(OUTPUT, index=False)
    ml_ok = df[['fd_home_ml','fd_away_ml']].notna().all(axis=1).sum()
    print(f"\nSaved {len(df)} games to {OUTPUT}")
    print(f"  With closing ML odds : {ml_ok}  (backtest uses only these)")
    print(f"  Scores only          : {len(df) - ml_ok}  (skipped by backtest)")
    print(f"\nNext step: python backtest.py  (RUN_MODE='2026_validation')")

if '--schema' in sys.argv:
    print(SCHEMA_GUIDE)
    sys.exit(0)

if '--validate' in sys.argv:
    validate_csv(OUTPUT)
    sys.exit(0)

if '--scores' in sys.argv:
    print("Pulling MLB Stats API scores (no odds)...")
    df = pull_mlb_scores()
    if not df.empty:
        save(df)
        print(SCHEMA_GUIDE)
    sys.exit(0)

# --- Full pipeline ---
print("=" * 70)
print("  2026 MLB Odds + Scores Downloader")
print(f"  Target: {SEASON_START} -> {SEASON_END}")
print(f"  Output: {OUTPUT}")
print("=" * 70)

df = None

print("\nSource 1: SportsBookReviewOnline Excel archive")
df = try_sbr_excel()

if df is None or df.empty:
    print("\nSource 2: GitHub CSV mirrors")
    df = try_github_csv()

if df is None or df.empty:
    print("\nSources 1-2 unavailable. Falling back to MLB Stats API (scores only).")
    df = pull_mlb_scores()
    if not df.empty:
        save(df)
        print("\nWARNING: No odds data found. CSV saved with scores only.")
        print("         Fill in fd_home_ml and fd_away_ml columns with closing")
        print("         moneylines, then re-run with --validate to normalize.")
        print(SCHEMA_GUIDE)
    else:
        print("All sources failed. Use --schema for the manual entry guide.")
    sys.exit(1 if (df is None or df.empty) else 0)

save(df)
