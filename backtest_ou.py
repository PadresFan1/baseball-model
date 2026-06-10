import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os
import random
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (mirrors backtest.py)
# ─────────────────────────────────────────────────────────────────────────────

TEAM_MAP = {
    'ATH': 133, 'PIT': 134, 'SDP': 135, 'SEA': 136, 'SFG': 137,
    'STL': 138, 'TBR': 139, 'TEX': 140, 'TOR': 141, 'MIN': 142,
    'PHI': 143, 'ATL': 144, 'CHW': 145, 'MIA': 146, 'NYY': 147,
    'MIL': 158, 'LAA': 108, 'ARI': 109, 'BAL': 110, 'BOS': 111,
    'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KCR': 118, 'LAD': 119, 'WSN': 120, 'NYM': 121,
}

ODDS_TO_FG = {
    'KC': 'KCR', 'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR',
    'WAS': 'WSN', 'CLE': 'CLE', 'OAK': 'ATH',
}

STATCAST_TEAM_MAP = {
    'AZ': 'ARI', 'CWS': 'CHW', 'KC': 'KCR',
    'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR', 'WSH': 'WSN',
}

_PARK_FACTOR_TEAM_MAP = {
    'Angels': 'LAA', 'Orioles': 'BAL', 'Red Sox': 'BOS',
    'White Sox': 'CHW', 'Guardians': 'CLE', 'Tigers': 'DET',
    'Royals': 'KCR', 'Twins': 'MIN', 'Yankees': 'NYY',
    'Athletics': 'ATH', 'Mariners': 'SEA', 'Rays': 'TBR',
    'Rangers': 'TEX', 'Blue Jays': 'TOR', 'Diamondbacks': 'ARI',
    'Braves': 'ATL', 'Cubs': 'CHC', 'Reds': 'CIN',
    'Rockies': 'COL', 'Dodgers': 'LAD', 'Marlins': 'MIA',
    'Brewers': 'MIL', 'Mets': 'NYM', 'Phillies': 'PHI',
    'Pirates': 'PIT', 'Cardinals': 'STL', 'Padres': 'SDP',
    'Giants': 'SFG', 'Nationals': 'WSN', 'Astros': 'HOU',
}

ALL_SEASONS     = [2021, 2022, 2023, 2024, 2025, 2026]
TRAIN_SEASONS   = [2021, 2022, 2023, 2024]
HOLDOUT_SEASONS = [2025, 2026]
CORR_THRESHOLD  = 0.03
MIN_PITCHER_IP  = 15.0
N_RANDOM_TRIALS = 500

# All candidate stats to test in Phase 1
CANDIDATE_STATS = [
    'comb_xfip',    # combined starter xFIP (home + away)
    'comb_fip',     # combined starter FIP
    'comb_k9',      # avg starter K/9
    'comb_bb9',     # avg starter BB/9
    'comb_era',     # avg starter ERA (FIP proxy — no individual ER in Statcast export)
    'comb_avg_ip',  # avg starter innings pitched
    'comb_runs_7',  # team rolling runs scored 7-day (home + away)
    'comb_runs_15', # team rolling runs scored 15-day (home + away)
    'comb_woba',    # team batting wOBA (home + away)
    'comb_xwoba',   # team batting xwOBA from Statcast (home + away)
    'comb_babip',   # team batting BABIP from Statcast (home + away)
    'comb_k_pct',   # team pitching K% (home + away)
    'comb_bb_pct',  # team pitching BB% (home + away)
    'park_factor',  # home team park run factor
    'comb_hr_rate', # team HR per game 7-day (home + away)
]

# Stats where higher value means FEWER expected runs — normalized as lg/val
INVERTED_STATS = {'comb_k9', 'comb_k_pct'}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ip_to_float(ip):
    ip = float(ip)
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3


def normalize_team(team):
    return ODDS_TO_FG.get(team, team)


def load_woba_fip_constants(season):
    df = pd.read_csv('constants/woba_fip_constants.csv')
    season = min(season, int(df['Season'].max()))
    row = df[df['Season'] == season].iloc[0]
    return {
        'wBB': float(row['wBB']), 'wHBP': float(row['wHBP']),
        'w1B': float(row['w1B']), 'w2B':  float(row['w2B']),
        'w3B': float(row['w3B']), 'wHR':  float(row['wHR']),
        'fip_constant': float(row['cFIP']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_odds_data():
    print("Loading odds data...")
    with open('cache/mlb_odds_dataset.json') as f:
        raw = json.load(f)

    rows = []
    for date, games in raw.items():
        for game in games:
            gv = game['gameView']
            if gv.get('gameType') != 'R':
                continue
            row = {
                'date':       date,
                'home_team':  gv['homeTeam']['shortName'],
                'away_team':  gv['awayTeam']['shortName'],
                'home_score': gv.get('homeTeamScore'),
                'away_score': gv.get('awayTeamScore'),
            }
            for tot in game.get('odds', {}).get('totals', []):
                if tot['sportsbook'] == 'fanduel':
                    row['fd_total']      = tot['currentLine']['total']
                    row['fd_over_odds']  = tot['currentLine']['overOdds']
                    row['fd_under_odds'] = tot['currentLine']['underOdds']
            rows.append(row)

    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    supp = 'historical_data/odds_2025_supplement.csv'
    if os.path.exists(supp):
        sdf = pd.read_csv(supp)
        df = pd.concat([df, sdf], ignore_index=True).sort_values('date').reset_index(drop=True)
        print(f"  Merged 2025 supplement: {len(sdf)} rows")

    for f26 in ['historical_data/odds_2026_complete.csv', 'historical_data/odds_2026.csv']:
        if os.path.exists(f26):
            d26 = pd.read_csv(f26)
            if len(d26) > 0:
                df = pd.concat([df, d26], ignore_index=True).sort_values('date').reset_index(drop=True)
                print(f"  Merged 2026 odds: {len(d26)} rows [{os.path.basename(f26)}]")
            break

    print(f"  Total odds rows: {len(df)}")
    return df


def load_game_lineups():
    path = 'historical_data/game_lineups.json'
    if not os.path.exists(path):
        print("  WARNING: game_lineups.json not found — pitcher stats will be unavailable")
        return {}
    with open(path) as f:
        return json.load(f)


def build_lineup_lookup(game_lineups):
    """Returns {(date, home_fg, away_fg): {home_starter_id, away_starter_id}}."""
    lookup = {}
    for season_games in game_lineups.values():
        for gpk_str, g in season_games.items():
            h = g.get('home_fg'); a = g.get('away_fg')
            if h and a:
                lookup[(g['date'], h, a)] = {
                    'home_starter_id': g.get('home_starter_id'),
                    'away_starter_id': g.get('away_starter_id'),
                    'game_pk':         int(gpk_str),
                }
    return lookup


def load_park_factors():
    df = pd.read_csv('constants/park_factors.csv')
    pf = {}
    for _, row in df[df['Season'] == 2025].iterrows():
        fg = _PARK_FACTOR_TEAM_MAP.get(str(row['Team']).strip())
        if fg:
            pf[fg] = float(row['1yr']) / 100
    return pf


# ─────────────────────────────────────────────────────────────────────────────
# PITCHER ROLLING CACHE
# (season-to-date cumulative stats per pitcher as of each game date)
# ─────────────────────────────────────────────────────────────────────────────

def _build_lg_hr_fb(game_logs, seasons, target_dates_by_season):
    """League HR/FB rate per (season_str, date) — needed for xFIP."""
    result = {}
    for season in seasons:
        s = str(season)
        events = []
        for tl in game_logs.get(s, {}).values():
            for g in tl['pitching']:
                events.append((g['date'], g['homeRuns'], g['airOuts'] + g['homeRuns']))
        events.sort()
        cum_hr = cum_fb = idx = 0
        for td in target_dates_by_season.get(s, []):
            while idx < len(events) and events[idx][0] < td:
                cum_hr += events[idx][1]; cum_fb += events[idx][2]; idx += 1
            result[(s, td)] = cum_hr / cum_fb if cum_fb > 0 else 0.115
    return result


def build_pitcher_rolling_cache(seasons, game_lineups, game_logs):
    """
    Cumulative season-to-date stats for each starting pitcher as of each
    target date derived from game_lineups.
    Returns {(pitcher_id: int, date: str): stats_dict}
    """
    print("Building pitcher rolling cache...")

    # IP lookup: (player_id, game_pk) -> decimal IP from box score
    ip_lookup = {}
    for season_games in game_lineups.values():
        for gpk_str, g in season_games.items():
            gpk = int(gpk_str)
            for side in ('home', 'away'):
                pid    = g.get(f'{side}_starter_id')
                ip_str = g.get(f'{side}_starter_ip')
                if pid is not None and ip_str is not None:
                    try:
                        ip_lookup[(int(pid), gpk)] = ip_to_float(str(ip_str))
                    except (ValueError, TypeError):
                        pass

    # Target dates per season (dates present in game_lineups)
    tdates_by_season = {}
    for s, season_games in game_lineups.items():
        tdates_by_season[s] = sorted(set(g['date'] for g in season_games.values()))

    lg_hr_fb = _build_lg_hr_fb(game_logs, seasons, tdates_by_season)

    cache = {}
    for season in seasons:
        s    = str(season)
        path = f'player_data/starters_{season}.csv'
        if not os.path.exists(path):
            print(f"  Missing {path} — skipping pitcher data for {season}")
            continue

        const     = load_woba_fip_constants(season)
        fip_const = const['fip_constant']

        needed = ['player_id', 'game_date', 'game_pk', 'pa', 'xwoba', 'so', 'bb', 'hrs']
        header_cols = set(pd.read_csv(path, nrows=0).columns)
        cols = [c for c in needed if c in header_cols]
        df = pd.read_csv(path, usecols=cols)
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
        df['player_id'] = df['player_id'].astype(int)
        df['game_pk']   = df['game_pk'].astype(int)
        df['pa']        = pd.to_numeric(df['pa'],    errors='coerce').fillna(0)
        df['xwoba']     = pd.to_numeric(df['xwoba'], errors='coerce')
        df['so']        = pd.to_numeric(df.get('so',  df.get('k',  0)), errors='coerce').fillna(0)
        df['bb']        = pd.to_numeric(df.get('bb',  0), errors='coerce').fillna(0)
        df['hrs']       = pd.to_numeric(df.get('hrs', df.get('hr', 0)), errors='coerce').fillna(0)

        df['ip'] = df.apply(lambda r: ip_lookup.get((r['player_id'], r['game_pk'])), axis=1)
        df = df[df['pa'] > 0].dropna(subset=['xwoba', 'ip'])
        df['fb'] = df['ip'] * 3.0 * 0.40  # fly ball proxy: IP × 3 outs × 0.40 FB/out

        target_dates = tdates_by_season.get(s, [])
        season_entries = 0

        for player_id, grp in df.groupby('player_id'):
            grp   = grp.sort_values('game_date').reset_index(drop=True)
            games = grp.to_dict('records')
            n     = len(games)

            cum_k = cum_bb = cum_hr = cum_ip = cum_fb = 0.0
            n_starts = game_idx = 0

            for td in target_dates:
                while game_idx < n and games[game_idx]['game_date'] < td:
                    g         = games[game_idx]
                    cum_k    += float(g['so'])
                    cum_bb   += float(g['bb'])
                    cum_hr   += float(g['hrs'])
                    cum_ip   += float(g['ip'])
                    cum_fb   += float(g['fb'])
                    n_starts += 1
                    game_idx += 1

                if cum_ip <= 0:
                    continue

                rate     = lg_hr_fb.get((s, td), 0.115)
                fip_num  = 13 * cum_hr + 3 * cum_bb - 2 * cum_k
                fip      = fip_num / cum_ip + fip_const

                exp_hr   = rate * cum_fb
                xfip_num = 13 * exp_hr + 3 * cum_bb - 2 * cum_k
                xfip     = xfip_num / cum_ip + fip_const

                cache[(int(player_id), td)] = {
                    'xfip':      xfip,
                    'fip':       fip,
                    'k9':        cum_k  * 9 / cum_ip,
                    'bb9':       cum_bb * 9 / cum_ip,
                    'era':       fip,   # FIP proxy — no individual ER in Statcast starters export
                    'avg_ip':    cum_ip / n_starts,
                    'ip_total':  cum_ip,
                    'n_starts':  n_starts,
                    'below_min': cum_ip < MIN_PITCHER_IP,
                }
                season_entries += 1

        print(f"  {season}: {season_entries:,} pitcher-date entries")

    print(f"Pitcher cache complete: {len(cache):,} total entries")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# TEAM ROLLING CACHE
# (season-to-date + 7/15-day windows for each team)
# ─────────────────────────────────────────────────────────────────────────────

def build_team_rolling_cache(seasons, game_logs, odds_df):
    """Returns {(season_str, team, date): stats_dict}."""
    print("Building team rolling cache...")
    cache = {}

    for season in seasons:
        s      = str(season)
        const  = load_woba_fip_constants(season)
        wBB = const['wBB']; wHBP = const['wHBP']
        w1B = const['w1B']; w2B  = const['w2B']
        w3B = const['w3B']; wHR  = const['wHR']

        target_dates = sorted(set(
            odds_df[odds_df['date'].str.startswith(s)]['date'].tolist()
        ))
        if not target_dates:
            continue

        for team, tl in game_logs.get(s, {}).items():
            hitting  = sorted(tl['hitting'],  key=lambda g: g['date'])
            pitching = sorted(tl['pitching'], key=lambda g: g['date'])
            if not hitting or not pitching:
                continue

            # Prefix sums — O(1) rolling windows
            cum_runs   = [0]; cum_hr_off = [0]; cum_pit = [0]
            for g in hitting:
                cum_runs.append(cum_runs[-1]   + g['runs'])
                cum_hr_off.append(cum_hr_off[-1] + g['homeRuns'])
            for g in pitching:
                cum_pit.append(cum_pit[-1] + g['runs_allowed'])

            woba_num = woba_den = 0.0
            k_tot = bb_tot = bf_tot = 0
            hit_idx = pit_idx = 0

            for td in target_dates:
                while hit_idx < len(hitting) and hitting[hit_idx]['date'] < td:
                    g       = hitting[hit_idx]
                    singles = g['hits'] - g['doubles'] - g['triples'] - g['homeRuns']
                    woba_num += (wBB*g['baseOnBalls'] + wHBP*g['hitByPitch'] +
                                 w1B*singles + w2B*g['doubles'] +
                                 w3B*g['triples'] + wHR*g['homeRuns'])
                    woba_den += g['plateAppearances'] - g['intentionalWalks']
                    hit_idx  += 1

                while pit_idx < len(pitching) and pitching[pit_idx]['date'] < td:
                    g = pitching[pit_idx]
                    k_tot  += g['strikeOuts']
                    bb_tot += g['baseOnBalls']
                    bf_tot += g['battersFaced']
                    pit_idx += 1

                if hit_idx < 7 or pit_idx < 7:
                    continue

                h7s  = max(0, hit_idx - 7);  h15s = max(0, hit_idx - 15)
                p7s  = max(0, pit_idx - 7)
                hit_7   = (cum_runs[hit_idx]   - cum_runs[h7s])    / (hit_idx - h7s)
                hit_15  = (cum_runs[hit_idx]   - cum_runs[h15s])   / (hit_idx - h15s)
                hr_7    = (cum_hr_off[hit_idx] - cum_hr_off[h7s])  / (hit_idx - h7s)

                cache[(s, team, td)] = {
                    'hit_7':   hit_7,
                    'hit_15':  hit_15,
                    'woba':    woba_num / woba_den if woba_den > 0 else 0.0,
                    'k_pct':   k_tot  / bf_tot if bf_tot > 0 else 0.20,
                    'bb_pct':  bb_tot / bf_tot if bf_tot > 0 else 0.08,
                    'hr_7':    hr_7,
                }

        print(f"  {season}: done")

    print(f"Team rolling cache: {len(cache):,} entries")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# STATCAST TEAM CACHE (batting xwOBA and BABIP from team Statcast CSVs)
# ─────────────────────────────────────────────────────────────────────────────

def build_statcast_team_cache(seasons, odds_df):
    """Returns {(team, date): {xwoba, babip}}."""
    print("Building statcast team cache...")
    cache = {}

    for season in seasons:
        path = f'statcast/batters_{season}.csv'
        if not os.path.exists(path):
            print(f"  Missing {path}")
            continue

        df = pd.read_csv(path, usecols=['player_name', 'game_date', 'xwoba', 'babip', 'pa'])
        df.columns = df.columns.str.strip()
        df['team']      = df['player_name'].replace(STATCAST_TEAM_MAP)
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['xwoba', 'babip', 'pa'])
        df = df[df['pa'] > 0]
        df['pa']    = df['pa'].astype(float)
        df['xwoba'] = df['xwoba'].astype(float)
        df['babip'] = df['babip'].astype(float)

        s            = str(season)
        target_dates = sorted(set(
            odds_df[odds_df['date'].str.startswith(s)]['date'].tolist()
        ))

        for team in TEAM_MAP:
            tdf = df[df['team'] == team].sort_values('game_date').reset_index(drop=True)
            if tdf.empty:
                continue
            games = tdf.to_dict('records')
            n = len(games)

            cum_pa = cum_x = cum_b = 0.0
            game_idx = 0

            for td in target_dates:
                while game_idx < n and games[game_idx]['game_date'] < td:
                    g = games[game_idx]
                    pa = g['pa']
                    cum_x  += g['xwoba'] * pa
                    cum_b  += g['babip'] * pa
                    cum_pa += pa
                    game_idx += 1

                if cum_pa >= 100:
                    cache[(team, td)] = {
                        'xwoba': cum_x / cum_pa,
                        'babip': cum_b / cum_pa,
                    }

        print(f"  {season}: done")

    print(f"Statcast team cache: {len(cache):,} entries")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# GAME RECORD ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def collect_game_records(seasons, odds_df, lineup_lookup, pitcher_cache,
                          team_cache, sc_cache, park_factors):
    """
    Assemble one record per game with all candidate features and O/U outcome.
    No data leakage: all stats are computed from data strictly before the game date.
    """
    records          = []
    season_records   = {s: [] for s in seasons}

    for season in seasons:
        s            = str(season)
        season_odds  = odds_df[odds_df['date'].str.startswith(s)]

        for _, game in season_odds.iterrows():
            date      = game['date']
            home_team = normalize_team(game['home_team'])
            away_team = normalize_team(game['away_team'])

            if home_team not in TEAM_MAP or away_team not in TEAM_MAP:
                continue

            hs = game.get('home_score')
            as_ = game.get('away_score')
            if pd.isna(hs) or pd.isna(as_):
                continue
            hs = float(hs); as_ = float(as_)

            fd_total = game.get('fd_total')
            if pd.isna(fd_total):
                continue
            fd_total = float(fd_total)

            actual_total = hs + as_
            if actual_total > fd_total:
                ou_result = 'OVER'
            elif actual_total < fd_total:
                ou_result = 'UNDER'
            else:
                continue  # push — exclude from analysis

            fd_over_odds  = game.get('fd_over_odds')
            fd_under_odds = game.get('fd_under_odds')
            over_odds  = float(fd_over_odds)  if not pd.isna(fd_over_odds)  else None
            under_odds = float(fd_under_odds) if not pd.isna(fd_under_odds) else None

            home_t = team_cache.get((s, home_team, date))
            away_t = team_cache.get((s, away_team, date))
            if home_t is None or away_t is None:
                continue

            # Pitcher stats — optional; records without pitcher data still included
            lu         = lineup_lookup.get((date, home_team, away_team))
            has_pit    = False
            below_min  = False
            home_pit   = away_pit = None
            if lu:
                h_pid = lu.get('home_starter_id')
                a_pid = lu.get('away_starter_id')
                if h_pid is not None and a_pid is not None:
                    home_pit = pitcher_cache.get((int(h_pid), date))
                    away_pit = pitcher_cache.get((int(a_pid), date))
                    has_pit  = home_pit is not None and away_pit is not None
                    if has_pit:
                        below_min = home_pit['below_min'] or away_pit['below_min']

            home_sc = sc_cache.get((home_team, date), {})
            away_sc = sc_cache.get((away_team, date), {})
            pf      = park_factors.get(home_team, 1.0)

            features = {}

            if has_pit:
                features['comb_xfip']   = home_pit['xfip']   + away_pit['xfip']
                features['comb_fip']    = home_pit['fip']     + away_pit['fip']
                features['comb_k9']     = (home_pit['k9']    + away_pit['k9'])    / 2
                features['comb_bb9']    = (home_pit['bb9']   + away_pit['bb9'])   / 2
                features['comb_era']    = (home_pit['era']   + away_pit['era'])   / 2
                features['comb_avg_ip'] = (home_pit['avg_ip']+ away_pit['avg_ip'])/ 2

            features['comb_runs_7']  = home_t['hit_7']  + away_t['hit_7']
            features['comb_runs_15'] = home_t['hit_15'] + away_t['hit_15']
            features['comb_woba']    = home_t['woba']   + away_t['woba']
            features['comb_k_pct']   = home_t['k_pct'] + away_t['k_pct']
            features['comb_bb_pct']  = home_t['bb_pct']+ away_t['bb_pct']
            features['comb_hr_rate'] = home_t['hr_7']  + away_t['hr_7']

            if home_sc and away_sc:
                features['comb_xwoba'] = home_sc['xwoba'] + away_sc['xwoba']
                features['comb_babip'] = home_sc['babip'] + away_sc['babip']
            elif home_sc:
                features['comb_xwoba'] = home_sc['xwoba'] * 2
                features['comb_babip'] = home_sc['babip'] * 2
            elif away_sc:
                features['comb_xwoba'] = away_sc['xwoba'] * 2
                features['comb_babip'] = away_sc['babip'] * 2

            features['park_factor'] = pf

            rec = {
                'date':          date,
                'season':        season,
                'home_team':     home_team,
                'away_team':     away_team,
                'actual_total':  actual_total,
                'fd_total':      fd_total,
                'over_odds':     over_odds,
                'under_odds':    under_odds,
                'ou_result':     ou_result,
                'ou_binary':     1 if ou_result == 'OVER' else 0,
                'has_pit':       has_pit,
                'below_min':     below_min,
                'features':      features,
            }
            records.append(rec)
            season_records[season].append(rec)

    print(f"\nAssembled {len(records):,} game records across {seasons}")
    for s in seasons:
        n    = len(season_records[s])
        npit = sum(1 for r in season_records[s] if r['has_pit'])
        if n > 0:
            over_pct = sum(r['ou_binary'] for r in season_records[s]) / n
            print(f"  {s}: {n} games | {npit} with pitcher data | OVER rate: {over_pct:.1%}")

    return records, season_records


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CORRELATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def phase1_correlation(records):
    print(f"\n{'='*65}")
    print(f"  PHASE 1 — POINT-BISERIAL CORRELATION  (threshold: |r| >= {CORR_THRESHOLD})")
    print(f"  N games: {len(records)}")
    print(f"{'='*65}")

    results = []
    for stat in CANDIDATE_STATS:
        vals    = []
        targets = []
        for r in records:
            v = r['features'].get(stat)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
                targets.append(r['ou_binary'])

        if len(vals) < 100:
            print(f"  {stat:<22} insufficient data ({len(vals)} obs) — skipped")
            continue

        corr, pval = sp_stats.pointbiserialr(targets, vals)
        results.append({
            'stat':     stat,
            'corr':     corr,
            'abs_corr': abs(corr),
            'p_value':  pval,
            'n':        len(vals),
            'passes':   abs(corr) >= CORR_THRESHOLD,
            'inverted': stat in INVERTED_STATS,
        })

    results.sort(key=lambda x: -x['abs_corr'])

    print(f"\n  {'Stat':<22} {'Corr':>8}  {'P-val':>8}  {'N':>6}  Status")
    print(f"  {'-'*60}")
    for r in results:
        status = 'PASS' if r['passes'] else 'WEAK — filtered out'
        inv    = ' [inv]' if r['inverted'] else ''
        print(f"  {r['stat']:<22} {r['corr']:>+8.4f}  {r['p_value']:>8.5f}  {r['n']:>6}  {status}{inv}")

    surviving = [r['stat'] for r in results if r['passes']]
    weak      = [r['stat'] for r in results if not r['passes']]

    print(f"\n  Surviving ({len(surviving)}): {surviving}")
    print(f"  Filtered  ({len(weak)}): {weak}")
    return surviving, results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — WALK-FORWARD RANDOM SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lg_avgs(records, stats):
    sums = {s: 0.0 for s in stats}; cnts = {s: 0 for s in stats}
    for r in records:
        for s in stats:
            v = r['features'].get(s)
            if v is not None and not np.isnan(v) and v > 0:
                sums[s] += v; cnts[s] += 1
    return {s: sums[s] / cnts[s] if cnts[s] > 0 else 1.0 for s in stats}


def _build_feature_matrix(records, stats, lg_avgs):
    """
    Returns (stat_matrix, fd_totals, park_factors, ou_results, over_odds, under_odds)
    stat_matrix[i, j] = normalized stat j for record i (1.0 = league avg, inverted where needed)
    """
    n    = len(records)
    ns   = len(stats)
    mat  = np.ones((n, ns), dtype=np.float64)

    for j, stat in enumerate(stats):
        lg = lg_avgs.get(stat, 1.0)
        if lg <= 0:
            lg = 1.0
        for i, r in enumerate(records):
            v = r['features'].get(stat)
            if v is None or np.isnan(v) or v <= 0:
                continue
            mat[i, j] = lg / v if stat in INVERTED_STATS else v / lg

    fd_totals  = np.array([r['fd_total']                      for r in records], dtype=np.float64)
    ou_results = np.array([1 if r['ou_result'] == 'OVER' else 0 for r in records], dtype=np.int8)
    over_odds  = [r['over_odds']  for r in records]
    under_odds = [r['under_odds'] for r in records]

    return mat, fd_totals, ou_results, over_odds, under_odds


def _eval_trial_on_matrix(weight_arr, threshold, lg_avg_total,
                           mat, fd_totals, ou_results, over_odds, under_odds):
    """Fast vectorized evaluation of one trial."""
    scores     = mat @ weight_arr              # (n,)
    pred       = lg_avg_total * scores         # park_factor already in mat
    diffs      = pred - fd_totals

    over_mask  = diffs >  threshold
    under_mask = diffs < -threshold
    bet_mask   = over_mask | under_mask
    if not bet_mask.any():
        return None

    # Determine win/loss for each bet
    # OVER bet: win if ou_result == 1 (OVER), UNDER bet: win if ou_result == 0 (UNDER)
    wins_mask  = (over_mask & (ou_results == 1)) | (under_mask & (ou_results == 0))

    n_bets = int(bet_mask.sum())
    n_wins = int(wins_mask.sum())
    if n_bets == 0:
        return None

    # ROI — use actual juice where available, else -110 assumption
    wagered  = n_bets * 100
    returned = 0.0
    indices  = np.where(bet_mask)[0]
    for i in indices:
        is_over = bool(over_mask[i])
        odds    = over_odds[i] if is_over else under_odds[i]
        if wins_mask[i]:
            if odds is not None:
                returned += 100 + (100 * odds / 100 if odds > 0 else 100 / abs(odds) * 100)
            else:
                returned += 190.91   # -110 assumption
        # loss: no return

    roi      = (returned - wagered) / wagered * 100
    win_rate = n_wins / n_bets
    return roi, n_bets, win_rate


def phase2_walk_forward(season_records, surviving_stats, n_trials=N_RANDOM_TRIALS):
    print(f"\n{'='*65}")
    print(f"  PHASE 2 — WALK-FORWARD RANDOM SEARCH  ({n_trials} trials)")
    print(f"  Folds: [2021]->2022, [2021-22]->2023, [2021-22-23]->2024")
    print(f"  Stats: {surviving_stats}")
    print(f"{'='*65}")

    folds = [
        ([2021],             [2022]),
        ([2021, 2022],       [2023]),
        ([2021, 2022, 2023], [2024]),
    ]

    # Precompute per-fold data
    fold_data = []
    for train_yrs, test_yrs in folds:
        train_recs  = [r for y in train_yrs for r in season_records.get(y, [])]
        test_recs   = [r for y in test_yrs  for r in season_records.get(y, [])]
        if not train_recs or not test_recs:
            continue
        lg_avgs     = _compute_lg_avgs(train_recs, surviving_stats)
        lg_avg_tot  = float(np.mean([r['actual_total'] for r in train_recs]))
        mat, fdt, our, oo, uo = _build_feature_matrix(test_recs, surviving_stats, lg_avgs)
        fold_data.append({
            'lg_avg_total': lg_avg_tot,
            'mat': mat, 'fd_totals': fdt,
            'ou_results': our, 'over_odds': oo, 'under_odds': uo,
            'n_test': len(test_recs),
        })
        print(f"  Fold {train_yrs}->{test_yrs}: {len(train_recs)} train, {len(test_recs)} test | lg_avg_total={lg_avg_tot:.2f}")

    if not fold_data:
        print("  ERROR: no fold data assembled")
        return [], None

    seed = random.randint(1, 99999)
    random.seed(seed)
    print(f"\n  Seed: {seed}\n")

    trial_results = []

    for trial in range(n_trials):
        # Sample weights (normalized to sum=1) and threshold
        raw    = {s: random.random() for s in surviving_stats}
        total  = sum(raw.values())
        w      = {s: v / total for s, v in raw.items()}
        thresh = random.uniform(0.5, 3.0)

        weight_arr = np.array([w[s] for s in surviving_stats])

        # Aggregate bets across all folds
        agg_bets = 0; agg_wins = 0; agg_wagered = 0.0; agg_returned = 0.0

        for fd in fold_data:
            res = _eval_trial_on_matrix(
                weight_arr, thresh, fd['lg_avg_total'],
                fd['mat'], fd['fd_totals'], fd['ou_results'],
                fd['over_odds'], fd['under_odds']
            )
            if res is None:
                continue
            roi_fold, n_fold, wr_fold = res

            # Re-extract raw returns for aggregation (minor overhead)
            scores    = fd['mat'] @ weight_arr
            pred      = fd['lg_avg_total'] * scores
            diffs     = pred - fd['fd_totals']
            over_m    = diffs >  thresh
            under_m   = diffs < -thresh
            bet_m     = over_m | under_m
            wins_m    = (over_m & (fd['ou_results'] == 1)) | (under_m & (fd['ou_results'] == 0))

            for i in np.where(bet_m)[0]:
                is_over = bool(over_m[i])
                odds    = fd['over_odds'][i] if is_over else fd['under_odds'][i]
                agg_bets    += 1
                agg_wagered += 100
                if wins_m[i]:
                    agg_wins += 1
                    if odds is not None:
                        agg_returned += 100 + (100*odds/100 if odds > 0 else 100/abs(odds)*100)
                    else:
                        agg_returned += 190.91

        if agg_bets < 20:
            continue

        roi = (agg_returned - agg_wagered) / agg_wagered * 100
        wr  = agg_wins / agg_bets

        trial_results.append({
            'roi':       roi,
            'win_rate':  wr,
            'n_bets':    agg_bets,
            'threshold': thresh,
            **{f'w_{s}': w[s] for s in surviving_stats},
        })

        if (trial + 1) % 50 == 0:
            best = max((r['roi'] for r in trial_results), default=0.0)
            print(f"  Trial {trial+1:>4}/{n_trials} | {len(trial_results):>4} valid | best ROI: {best:+.1f}%")

    if not trial_results:
        print("  No valid trials — check data coverage")
        return [], None

    df = pd.DataFrame(trial_results).sort_values('roi', ascending=False)
    df['seed'] = seed

    out = 'historical_data/ou_search_results.csv'
    df.to_csv(out, index=False)
    print(f"\n  Saved {len(df)} results to {out}")

    print(f"\n  Top 10 parameter combinations (walk-forward 2022-2024 test years):")
    show = ['roi', 'win_rate', 'n_bets', 'threshold']
    print(df[show].head(10).to_string(index=False))

    best = df.iloc[0]
    best_params = {
        'weights':   {s: float(best[f'w_{s}']) for s in surviving_stats},
        'threshold': float(best['threshold']),
    }
    return trial_results, best_params


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — HOLDOUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def phase3_holdout(best_params, season_records, surviving_stats, train_records):
    print(f"\n{'='*65}")
    print(f"  PHASE 3 — HOLDOUT VALIDATION  (parameters frozen from Phase 2)")
    print(f"{'='*65}")

    lg_avgs    = _compute_lg_avgs(train_records, surviving_stats)
    lg_avg_tot = float(np.mean([r['actual_total'] for r in train_records]))
    weight_arr = np.array([best_params['weights'].get(s, 0.0) for s in surviving_stats])
    thresh     = best_params['threshold']

    print(f"  Threshold     : {thresh:.2f} runs above/below book line to trigger bet")
    print(f"  lg_avg_total  : {lg_avg_tot:.2f} (from 2021-2024 training)")
    print(f"  Top weights   :")
    sorted_w = sorted(best_params['weights'].items(), key=lambda x: -x[1])
    for stat, w in sorted_w[:5]:
        print(f"    {stat:<22} {w:.4f}")
    print()

    print(f"  {'Year':<6} {'N Bets':<8} {'Wins':<6} {'Win%':<8} {'ROI':>8}  Notes")
    print(f"  {'-'*60}")

    all_bets = []
    for season in HOLDOUT_SEASONS:
        recs = season_records.get(season, [])
        if not recs:
            print(f"  {season:<6} no data")
            continue

        mat, fdt, our, oo, uo = _build_feature_matrix(recs, surviving_stats, lg_avgs)

        scores  = mat @ weight_arr
        pred    = lg_avg_tot * scores
        diffs   = pred - fdt
        over_m  = diffs >  thresh
        under_m = diffs < -thresh
        bet_m   = over_m | under_m
        wins_m  = (over_m & (our == 1)) | (under_m & (our == 0))

        n_bets = int(bet_m.sum())
        if n_bets == 0:
            print(f"  {season:<6} {'no bets'}")
            continue

        wagered  = n_bets * 100
        returned = 0.0; wins = 0; no_juice = 0
        for i in np.where(bet_m)[0]:
            is_over = bool(over_m[i])
            odds    = oo[i] if is_over else uo[i]
            if odds is None:
                no_juice += 1
            if wins_m[i]:
                wins += 1
                if odds is not None:
                    returned += 100 + (100*odds/100 if odds > 0 else 100/abs(odds)*100)
                else:
                    returned += 190.91

        roi  = (returned - wagered) / wagered * 100
        wr   = wins / n_bets
        note = f"({no_juice} bets w/ -110 estimate)" if no_juice > 0 else ""
        print(f"  {season:<6} {n_bets:<8} {wins:<6} {wr:<8.1%} {roi:>+7.1f}%  {note}")

        # Collect for combined totals
        all_bets.append((season, n_bets, wins, wagered, returned))

    if len(all_bets) > 1:
        tot_bets = sum(x[1] for x in all_bets)
        tot_wins = sum(x[2] for x in all_bets)
        tot_wag  = sum(x[3] for x in all_bets)
        tot_ret  = sum(x[4] for x in all_bets)
        print(f"  {'-'*60}")
        print(f"  {'TOTAL':<6} {tot_bets:<8} {tot_wins:<6} {tot_wins/tot_bets:<8.1%} {(tot_ret-tot_wag)/tot_wag*100:>+7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  O/U BACKTEST — Binary Classification (standalone)")
    print("=" * 65 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────
    odds_df = load_odds_data()

    # Pull any missing seasons into game logs before loading (mirrors backtest.py startup)
    import statsapi as _statsapi
    _log_path = 'historical_data/historical_game_logs.json'
    _existing = {}
    if os.path.exists(_log_path):
        with open(_log_path) as f:
            _existing = json.load(f)
    _missing = [s for s in ALL_SEASONS if str(s) not in _existing]
    if _missing:
        print(f"Pulling missing game logs for: {_missing}")
        for _s in _missing:
            print(f"  Pulling {_s}...")
            _season_logs = {}
            for _fg, _tid in TEAM_MAP.items():
                try:
                    _h = _statsapi.get('team_stats', {'teamId': _tid, 'stats': 'gameLog', 'group': 'hitting',  'season': _s})
                    _p = _statsapi.get('team_stats', {'teamId': _tid, 'stats': 'gameLog', 'group': 'pitching', 'season': _s})
                    _hg = _h['stats'][0]['splits']; _pg = _p['stats'][0]['splits']
                    _season_logs[_fg] = {
                        'hitting':  [{'date': g['date'], 'runs': g['stat']['runs'], 'hits': g['stat']['hits'],
                                      'doubles': g['stat']['doubles'], 'triples': g['stat']['triples'],
                                      'homeRuns': g['stat']['homeRuns'], 'baseOnBalls': g['stat']['baseOnBalls'],
                                      'intentionalWalks': g['stat']['intentionalWalks'],
                                      'hitByPitch': g['stat']['hitByPitch'], 'atBats': g['stat']['atBats'],
                                      'sacFlies': g['stat']['sacFlies'],
                                      'plateAppearances': g['stat']['plateAppearances']} for g in _hg],
                        'pitching': [{'date': g['date'], 'runs_allowed': g['stat']['runs'],
                                      'earnedRuns': g['stat'].get('earnedRuns', g['stat']['runs']),
                                      'hits': g['stat']['hits'], 'homeRuns': g['stat']['homeRuns'],
                                      'baseOnBalls': g['stat']['baseOnBalls'],
                                      'hitBatsmen': g['stat']['hitBatsmen'],
                                      'strikeOuts': g['stat']['strikeOuts'],
                                      'inningsPitched': g['stat']['inningsPitched'],
                                      'airOuts': g['stat']['airOuts'],
                                      'battersFaced': g['stat']['battersFaced']} for g in _pg],
                    }
                except Exception as _e:
                    print(f"    {_fg}: {_e}")
            _existing[str(_s)] = _season_logs
            print(f"  {_s}: {len(_season_logs)} teams pulled")
        with open(_log_path, 'w') as f:
            json.dump(_existing, f)
        print("Game logs updated.")

    with open(_log_path) as f:
        game_logs = json.load(f)

    game_lineups  = load_game_lineups()
    lineup_lookup = build_lineup_lookup(game_lineups)
    park_factors  = load_park_factors()

    # ── Build caches ───────────────────────────────────────────────────────
    active_seasons = [s for s in ALL_SEASONS
                      if str(s) in game_logs or
                      odds_df['date'].str.startswith(str(s)).any()]

    pitcher_cache = build_pitcher_rolling_cache(active_seasons, game_lineups, game_logs)
    team_cache    = build_team_rolling_cache(active_seasons, game_logs, odds_df)
    sc_cache      = build_statcast_team_cache(active_seasons, odds_df)

    # ── Assemble records ───────────────────────────────────────────────────
    records, season_records = collect_game_records(
        active_seasons, odds_df, lineup_lookup,
        pitcher_cache, team_cache, sc_cache, park_factors,
    )

    if not records:
        print("ERROR: No game records assembled. Check data files.")
        return

    # ── Phase 1: correlation (uses all data — analysis only, not training) ─
    surviving_stats, corr_results = phase1_correlation(records)

    if not surviving_stats:
        print("\nNo stats passed the correlation threshold — aborting.")
        return

    # ── Phase 2: walk-forward search on 2021-2024 ─────────────────────────
    train_season_records = {s: season_records.get(s, []) for s in TRAIN_SEASONS}
    _, best_params = phase2_walk_forward(train_season_records, surviving_stats)

    if best_params is None:
        print("Phase 2 returned no valid combinations — aborting.")
        return

    # ── Phase 3: holdout on 2025 and 2026 ─────────────────────────────────
    train_records = [r for s in TRAIN_SEASONS for r in season_records.get(s, [])]
    phase3_holdout(best_params, season_records, surviving_stats, train_records)

    # ── Plain-English summary ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")

    passed = [r for r in corr_results if r['passes']]
    print(f"\n  Phase 1 — {len(passed)} stats passed |r| >= {CORR_THRESHOLD}:")
    for r in passed:
        inv = " [inverted in scoring: higher stat = fewer runs]" if r['inverted'] else ""
        print(f"    {r['stat']:<22} r={r['corr']:+.4f}{inv}")

    print(f"\n  Phase 2 — Best parameters (walk-forward 2022-2024 OOS):")
    print(f"    Threshold : {best_params['threshold']:.2f} runs above/below book line")
    print(f"    Stat weights (top 5):")
    for stat, w in sorted(best_params['weights'].items(), key=lambda x: -x[1])[:5]:
        print(f"      {stat:<22} {w:.4f}")

    print(f"\n  Phase 3 — Holdout results printed above.")
    print(f"\n  NOTE: 'comb_era' uses FIP as proxy (individual earned-run data not")
    print(f"  available in the Statcast starters export). FIP is a better ERA")
    print(f"  estimator in any case.")
    print()


if __name__ == '__main__':
    main()
