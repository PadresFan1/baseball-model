import json
import pandas as pd
import statsapi
import numpy as np
import os
import random
from itertools import product

TEAM_MAP = {
    'ATH': 133, 'PIT': 134, 'SDP': 135, 'SEA': 136, 'SFG': 137,
    'STL': 138, 'TBR': 139, 'TEX': 140, 'TOR': 141, 'MIN': 142,
    'PHI': 143, 'ATL': 144, 'CHW': 145, 'MIA': 146, 'NYY': 147,
    'MIL': 158, 'LAA': 108, 'ARI': 109, 'BAL': 110, 'BOS': 111,
    'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KCR': 118, 'LAD': 119, 'WSN': 120, 'NYM': 121
}

import sys
sys.path.insert(0, '.')

ODDS_TO_FG = {
    'KC': 'KCR',
    'SD': 'SDP',
    'SF': 'SFG',
    'TB': 'TBR',
    'WAS': 'WSN',
    'CLE': 'CLE',
    'OAK': 'ATH'
}
import random
from itertools import product

def random_search(n_trials=100, seed=None):
    if seed:
        random.seed(seed)
    
    # Parameter ranges to search
    param_space = {
        'ml_edge_min': [0.03, 0.04, 0.05, 0.06, 0.07],
        'ml_edge_max': [0.08, 0.10, 0.15, 0.20, 1.0],
        'favorites_only': [True, False],
        'rolling_weight_7': [0.50, 0.60, 0.70, 0.80],
        'rolling_weight_15': None,  # calculated as 1 - rolling_weight_7
        'rating_cap_low': [0.60, 0.65, 0.70],
        'rating_cap_high': [1.30, 1.40, 1.50, 1.60],
        'w_rolling_off': [0.1, 0.2, 0.3],
        'w_woba': [0.1, 0.2, 0.3],
        'w_xwoba': [0.1, 0.2, 0.3],
        'w_k_off': [0.1, 0.2, 0.3],
        'w_bb_off': [0.1, 0.2, 0.3],
        'w_rolling_pit': [0.1, 0.2, 0.3],
        'w_fip': [0.1, 0.2, 0.3],
        'w_xfip': [0.1, 0.2, 0.3],
        'w_k_pit': [0.1, 0.2, 0.3],
        'w_bb_pit': [0.1, 0.2, 0.3],
    }
    
    results = []
    
    for trial in range(n_trials):
        # Sample random parameters
        params = {
            'ml_edge_min': random.choice(param_space['ml_edge_min']),
            'ml_edge_max': random.choice(param_space['ml_edge_max']),
            'favorites_only': random.choice(param_space['favorites_only']),
            'rolling_weight_7': random.choice(param_space['rolling_weight_7']),
            'rating_cap_low': random.choice(param_space['rating_cap_low']),
            'rating_cap_high': random.choice(param_space['rating_cap_high']),
            'w_rolling_off': random.choice(param_space['w_rolling_off']),
            'w_woba': random.choice(param_space['w_woba']),
            'w_xwoba': random.choice(param_space['w_xwoba']),
            'w_k_off': random.choice(param_space['w_k_off']),
            'w_bb_off': random.choice(param_space['w_bb_off']),
            'w_rolling_pit': random.choice(param_space['w_rolling_pit']),
            'w_fip': random.choice(param_space['w_fip']),
            'w_xfip': random.choice(param_space['w_xfip']),
            'w_k_pit': random.choice(param_space['w_k_pit']),
            'w_bb_pit': random.choice(param_space['w_bb_pit']),
        }
        params['rolling_weight_15'] = 1 - params['rolling_weight_7']
        
        # Normalize offense weights
        off_total = params['w_rolling_off'] + params['w_woba'] + params['w_xwoba'] + params['w_k_off'] + params['w_bb_off']
        params['w_rolling_off'] /= off_total
        params['w_woba'] /= off_total
        params['w_xwoba'] /= off_total
        params['w_k_off'] /= off_total
        params['w_bb_off'] /= off_total
        
        # Normalize pitching weights
        pit_total = params['w_rolling_pit'] + params['w_fip'] + params['w_xfip'] + params['w_k_pit'] + params['w_bb_pit']
        params['w_rolling_pit'] /= pit_total
        params['w_fip'] /= pit_total
        params['w_xfip'] /= pit_total
        params['w_k_pit'] /= pit_total
        params['w_bb_pit'] /= pit_total
        
        # Run backtest with these params
        trial_results = run_backtest_with_params(params, seasons=[2021, 2022, 2023, 2024])
        
        if trial_results is not None:
            results.append({**params, **trial_results})
        
        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{n_trials} trials")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)
    results_df.to_csv('historical_data/random_search_results.csv', index=False)
    
    print(f"\nTop 10 parameter combinations:")
    print(results_df[['roi', 'win_rate', 'n_bets', 'ml_edge_min', 'ml_edge_max', 
                       'favorites_only', 'rolling_weight_7', 'rating_cap_low', 
                       'rating_cap_high']].head(10).to_string())
    
    return results_df

def pull_historical_game_logs(team_ids, seasons=[2021, 2022, 2023, 2024]):
    all_logs = {}
    for season in seasons:
        print(f"Pulling {season} game logs...")
        season_logs = {}
        for fg_abbrev, team_id in team_ids.items():
            try:
                hitting = statsapi.get('team_stats', {
                    'teamId': team_id,
                    'stats': 'gameLog',
                    'group': 'hitting',
                    'season': season
                })
                pitching = statsapi.get('team_stats', {
                    'teamId': team_id,
                    'stats': 'gameLog',
                    'group': 'pitching',
                    'season': season
                })
                
                hit_games = hitting['stats'][0]['splits']
                pit_games = pitching['stats'][0]['splits']
                
                season_logs[fg_abbrev] = {
                    'hitting': [{
                        'date': g['date'],
                        'runs': g['stat']['runs'],
                        'hits': g['stat']['hits'],
                        'doubles': g['stat']['doubles'],
                        'triples': g['stat']['triples'],
                        'homeRuns': g['stat']['homeRuns'],
                        'baseOnBalls': g['stat']['baseOnBalls'],
                        'intentionalWalks': g['stat']['intentionalWalks'],
                        'hitByPitch': g['stat']['hitByPitch'],
                        'atBats': g['stat']['atBats'],
                        'sacFlies': g['stat']['sacFlies'],
                        'plateAppearances': g['stat']['plateAppearances']
                    } for g in hit_games],
                    'pitching': [{
                        'date': g['date'],
                        'runs_allowed': g['stat']['runs'],
                        'homeRuns': g['stat']['homeRuns'],
                        'baseOnBalls': g['stat']['baseOnBalls'],
                        'hitBatsmen': g['stat']['hitBatsmen'],
                        'strikeOuts': g['stat']['strikeOuts'],
                        'inningsPitched': g['stat']['inningsPitched'],
                        'airOuts': g['stat']['airOuts'],
                        'battersFaced': g['stat']['battersFaced']
                    } for g in pit_games]
                }
            except Exception as e:
                print(f"Error pulling {fg_abbrev} {season}: {e}")
                continue
        
        all_logs[season] = season_logs
        print(f"Pulled {len(season_logs)} teams for {season}")
    
    with open('historical_data/historical_game_logs.json', 'w') as f:
        json.dump(all_logs, f)
    print("Saved historical game logs")
    return all_logs

# Run this once to pull fresh data, then comment out again
historical_logs = pull_historical_game_logs(TEAM_MAP)

# Load historical game logs
with open('historical_data/historical_game_logs.json', 'r') as f:
    game_logs = json.load(f)

def normalize_team(team):
    return ODDS_TO_FG.get(team, team)

# Load season FanGraphs data
def load_season_stats(year):
    offense = pd.read_csv(f'offense_{year}.csv')
    pitching = pd.read_csv(f'pitching_{year}.csv')
    
    # Normalize columns same as model.py
    for df in [offense, pitching]:
        df.columns = df.columns.str.strip().str.lower()\
            .str.replace('+', 'plus', regex=False)\
            .str.replace('/', '_', regex=False)\
            .str.replace(' ', '_', regex=False)\
            .str.replace('%', '_pct', regex=False)\
            .str.replace('(', '', regex=False)\
            .str.replace(')', '', regex=False)
    
    return offense, pitching

# Load historical odds
with open('mlb_odds_dataset.json', 'r') as f:
    raw_odds = json.load(f)

rows = []
for date, games in raw_odds.items():
    for game in games:
        gv = game['gameView']
        if gv.get('gameType') != 'R':
            continue
        home_team = gv['homeTeam']['shortName']
        away_team = gv['awayTeam']['shortName']
        home_score = gv.get('homeTeamScore')
        away_score = gv.get('awayTeamScore')
        row = {
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score
        }
        odds = game.get('odds', {})
        for ml in odds.get('moneyline', []):
            if ml['sportsbook'] == 'fanduel':
                row['fd_home_ml'] = ml['currentLine']['homeOdds']
                row['fd_away_ml'] = ml['currentLine']['awayOdds']
            elif ml['sportsbook'] == 'draftkings':
                row['dk_home_ml'] = ml['currentLine']['homeOdds']
                row['dk_away_ml'] = ml['currentLine']['awayOdds']
        for tot in odds.get('totals', []):
            if tot['sportsbook'] == 'fanduel':
                row['fd_total'] = tot['currentLine']['total']
                row['fd_over_odds'] = tot['currentLine']['overOdds']
                row['fd_under_odds'] = tot['currentLine']['underOdds']
            elif tot['sportsbook'] == 'draftkings':
                row['dk_total'] = tot['currentLine']['total']
                row['dk_over_odds'] = tot['currentLine']['overOdds']
                row['dk_under_odds'] = tot['currentLine']['underOdds']
        rows.append(row)

odds_df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

# Load historical game logs
with open('historical_data/historical_game_logs.json', 'r') as f:
    game_logs = json.load(f)

# ============================================================
# ROLLING AVERAGES
# ============================================================

def get_rolling_averages_for_date(team, game_logs, target_date, season, woba_weights, fip_constant):
    season_str = str(season)
    if season_str not in game_logs:
        return None
    if team not in game_logs[season_str]:
        return None

    team_logs = game_logs[season_str][team]
    hitting = team_logs['hitting']
    pitching = team_logs['pitching']

    hit_before = [g for g in hitting if g['date'] < target_date]
    pit_before = [g for g in pitching if g['date'] < target_date]

    if len(hit_before) < 7 or len(pit_before) < 7:
        return None

    # --- Rolling runs (unchanged) ---
    hit_7  = sum(g['runs'] for g in hit_before[-7:]) / 7
    hit_15 = sum(g['runs'] for g in hit_before[-15:]) / min(15, len(hit_before))
    pit_7  = sum(g['runs_allowed'] for g in pit_before[-7:]) / 7
    pit_15 = sum(g['runs_allowed'] for g in pit_before[-15:]) / min(15, len(pit_before))

    # --- Cumulative wOBA (season-to-date) ---
    # numerator: weighted hit events; denominator: PA - IBB
    wBB, wHBP, w1B, w2B, w3B, wHR = (
        woba_weights['wBB'], woba_weights['wHBP'], woba_weights['w1B'],
        woba_weights['w2B'], woba_weights['w3B'],  woba_weights['wHR']
    )

    woba_num = 0.0
    woba_den = 0.0
    for g in hit_before:
        singles = g['hits'] - g['doubles'] - g['triples'] - g['homeRuns']
        woba_num += (
            wBB  * g['baseOnBalls'] +
            wHBP * g['hitByPitch'] +
            w1B  * singles +
            w2B  * g['doubles'] +
            w3B  * g['triples'] +
            wHR  * g['homeRuns']
        )
        woba_den += g['plateAppearances'] - g['intentionalWalks']

    woba = woba_num / woba_den if woba_den > 0 else 0.0

    # --- Cumulative FIP (season-to-date) ---
    fip_num = 0.0
    fip_ip  = 0.0
    k_total = 0
    bb_total = 0
    bf_total = 0

    for g in pit_before:
        fip_num += (
            13 * g['homeRuns'] +
            3  * (g['baseOnBalls'] + g['hitBatsmen']) -
            2  * g['strikeOuts']
        )
        fip_ip  += g['inningsPitched']
        k_total  += g['strikeOuts']
        bb_total += g['baseOnBalls']
        bf_total += g['battersFaced']

    fip = (fip_num / fip_ip + fip_constant) if fip_ip > 0 else 4.50

    # --- K% and BB% ---
    k_pct  = k_total  / bf_total if bf_total > 0 else 0.20
    bb_pct = bb_total / bf_total if bf_total > 0 else 0.08

    # --- xFIP: needs league HR/FB rate passed in or calculated externally ---
    # airOuts is a proxy for fly balls (MLB Stats API field)
    total_hr = sum(g['homeRuns']  for g in pit_before)
    total_fb = sum(g['airOuts']   for g in pit_before)  # approximate
    # xFIP uses lg_hr_fb_rate — caller should pass this in; fallback to FIP
    # (see note below — handle xFIP in the calling function)

    return {
        'hit_7':   hit_7,
        'hit_15':  hit_15,
        'pitch_7': pit_7,
        'pitch_15': pit_15,
        'woba':    woba,
        'fip':     fip,
        'k_pct':   k_pct,
        'bb_pct':  bb_pct,
        # raw counts for xFIP calculation upstream
        '_fip_num': fip_num,
        '_fip_ip':  fip_ip,
        '_total_fb': total_fb,
    }


# ============================================================
# BACKTEST PARAMETERS - adjust these to test combinations
# ============================================================
BACKTEST_PARAMS = {
    'rolling_weight_7': 0.60,
    'rolling_weight_15': 0.40,
    'edge_threshold_ml': 0.03,
    'edge_threshold_ou': 1.5,
    'num_simulations': 5000,
    'min_games_required': 7,
    'rating_cap_low': 0.60,
    'rating_cap_high': 1.60
}

def calculate_rolling_rating(rolling, lg_avg_runs):
    if rolling is None:
        return None, None
    
    off = (rolling['hit_7'] * BACKTEST_PARAMS['rolling_weight_7'] + 
           rolling['hit_15'] * BACKTEST_PARAMS['rolling_weight_15'])
    pit = (rolling['pitch_7'] * BACKTEST_PARAMS['rolling_weight_7'] + 
           rolling['pitch_15'] * BACKTEST_PARAMS['rolling_weight_15'])
    
    off_rating = off / lg_avg_runs
    pit_rating = lg_avg_runs / pit if pit > 0 else 1.0
    
    off_rating = max(BACKTEST_PARAMS['rating_cap_low'], min(BACKTEST_PARAMS['rating_cap_high'], off_rating))
    pit_rating = max(BACKTEST_PARAMS['rating_cap_low'], min(BACKTEST_PARAMS['rating_cap_high'], pit_rating))
    
    return off_rating, pit_rating

def run_simulation(home_lambda, away_lambda):
    home_scores = np.random.poisson(home_lambda, BACKTEST_PARAMS['num_simulations'])
    away_scores = np.random.poisson(away_lambda, BACKTEST_PARAMS['num_simulations'])
    
    ties = home_scores == away_scores
    extra_home = np.random.poisson(home_lambda * 0.111, ties.sum())
    extra_away = np.random.poisson(away_lambda * 0.111, ties.sum())
    home_scores[ties] += extra_home
    away_scores[ties] += extra_away
    
    home_wins = np.sum(home_scores > away_scores)
    away_wins = np.sum(away_scores > home_scores)
    remaining = BACKTEST_PARAMS['num_simulations'] - home_wins - away_wins
    
    home_win_pct = (home_wins + remaining / 2) / BACKTEST_PARAMS['num_simulations']
    away_win_pct = (away_wins + remaining / 2) / BACKTEST_PARAMS['num_simulations']
    avg_total = np.mean(home_scores + away_scores)
    
    return home_win_pct, away_win_pct, avg_total

def american_to_prob(line):
    if line < 0:
        return abs(line) / (abs(line) + 100)
    else:
        return 100 / (line + 100)

print("Backtest functions loaded successfully")

# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

def run_backtest_with_params(params, seasons=[2021, 2022, 2023, 2024]):
    all_results = []
    
    for season in seasons:
        season_odds = odds_df[odds_df['date'].str.startswith(str(season))]
        
        all_runs = []
        for team in TEAM_MAP.keys():
            season_str = str(season)
            if season_str in game_logs and team in game_logs[season_str]:
                for g in game_logs[season_str][team]['hitting']:
                    all_runs.append(g['runs'])
        lg_avg_runs = sum(all_runs) / len(all_runs) if all_runs else 4.5
        
        season_ratings = get_season_ratings(season)
        lg_woba = sum(v['woba'] for v in season_ratings.values()) / len(season_ratings)
        lg_xwoba = sum(v['xwoba'] for v in season_ratings.values()) / len(season_ratings)
        lg_fip = sum(v['fip'] for v in season_ratings.values()) / len(season_ratings)
        lg_xfip = sum(v['xfip'] for v in season_ratings.values()) / len(season_ratings)
        lg_k9 = sum(v['k_9_pit'] for v in season_ratings.values()) / len(season_ratings)
        lg_bb9 = sum(v['bb_9_pit'] for v in season_ratings.values()) / len(season_ratings)
        lg_k_off = sum(v['k_pct_off'] for v in season_ratings.values()) / len(season_ratings)
        lg_bb_off = sum(v['bb_pct_off'] for v in season_ratings.values()) / len(season_ratings)
        
        for _, game in season_odds.iterrows():
            date = game['date']
            home_team = normalize_team(game['home_team'])
            away_team = normalize_team(game['away_team'])
            
            if home_team not in TEAM_MAP or away_team not in TEAM_MAP:
                continue
            
            home_rolling = get_rolling_averages_for_date(home_team, game_logs, date, season)
            away_rolling = get_rolling_averages_for_date(away_team, game_logs, date, season)
            
            if home_rolling is None or away_rolling is None:
                continue
            
            # Calculate rolling ratings
            home_off_roll = ((home_rolling['hit_7'] * params['rolling_weight_7'] + 
                             home_rolling['hit_15'] * params['rolling_weight_15']) / lg_avg_runs)
            away_off_roll = ((away_rolling['hit_7'] * params['rolling_weight_7'] + 
                             away_rolling['hit_15'] * params['rolling_weight_15']) / lg_avg_runs)
            home_pit_roll = (lg_avg_runs / (home_rolling['pitch_7'] * params['rolling_weight_7'] + 
                             home_rolling['pitch_15'] * params['rolling_weight_15']))
            away_pit_roll = (lg_avg_runs / (away_rolling['pitch_7'] * params['rolling_weight_7'] + 
                             away_rolling['pitch_15'] * params['rolling_weight_15']))
            
            home_sr = season_ratings.get(home_team, {})
            away_sr = season_ratings.get(away_team, {})
            
            home_woba_r = home_sr.get('woba', lg_woba) / lg_woba
            away_woba_r = away_sr.get('woba', lg_woba) / lg_woba
            home_xwoba_r = home_sr.get('xwoba', lg_xwoba) / lg_xwoba
            away_xwoba_r = away_sr.get('xwoba', lg_xwoba) / lg_xwoba
            home_k_off_r = lg_k_off / home_sr.get('k_pct_off', lg_k_off) if home_sr.get('k_pct_off', 0) > 0 else 1.0
            away_k_off_r = lg_k_off / away_sr.get('k_pct_off', lg_k_off) if away_sr.get('k_pct_off', 0) > 0 else 1.0
            home_bb_off_r = home_sr.get('bb_pct_off', lg_bb_off) / lg_bb_off
            away_bb_off_r = away_sr.get('bb_pct_off', lg_bb_off) / lg_bb_off
            home_fip_r = lg_fip / home_sr.get('fip', lg_fip) if home_sr.get('fip', 0) > 0 else 1.0
            away_fip_r = lg_fip / away_sr.get('fip', lg_fip) if away_sr.get('fip', 0) > 0 else 1.0
            home_xfip_r = lg_xfip / home_sr.get('xfip', lg_xfip) if home_sr.get('xfip', 0) > 0 else 1.0
            away_xfip_r = lg_xfip / away_sr.get('xfip', lg_xfip) if away_sr.get('xfip', 0) > 0 else 1.0
            home_k_pit_r = home_sr.get('k_9_pit', lg_k9) / lg_k9
            away_k_pit_r = away_sr.get('k_9_pit', lg_k9) / lg_k9
            home_bb_pit_r = lg_bb9 / home_sr.get('bb_9_pit', lg_bb9) if home_sr.get('bb_9_pit', 0) > 0 else 1.0
            away_bb_pit_r = lg_bb9 / away_sr.get('bb_9_pit', lg_bb9) if away_sr.get('bb_9_pit', 0) > 0 else 1.0
            
            home_off = (home_off_roll * params['w_rolling_off'] + home_woba_r * params['w_woba'] +
                       home_xwoba_r * params['w_xwoba'] + home_k_off_r * params['w_k_off'] +
                       home_bb_off_r * params['w_bb_off'])
            away_off = (away_off_roll * params['w_rolling_off'] + away_woba_r * params['w_woba'] +
                       away_xwoba_r * params['w_xwoba'] + away_k_off_r * params['w_k_off'] +
                       away_bb_off_r * params['w_bb_off'])
            home_pit = (home_pit_roll * params['w_rolling_pit'] + home_fip_r * params['w_fip'] +
                       home_xfip_r * params['w_xfip'] + home_k_pit_r * params['w_k_pit'] +
                       home_bb_pit_r * params['w_bb_pit'])
            away_pit = (away_pit_roll * params['w_rolling_pit'] + away_fip_r * params['w_fip'] +
                       away_xfip_r * params['w_xfip'] + away_k_pit_r * params['w_k_pit'] +
                       away_bb_pit_r * params['w_bb_pit'])
            
            home_off = max(params['rating_cap_low'], min(params['rating_cap_high'], home_off))
            away_off = max(params['rating_cap_low'], min(params['rating_cap_high'], away_off))
            home_pit = max(params['rating_cap_low'], min(params['rating_cap_high'], home_pit))
            away_pit = max(params['rating_cap_low'], min(params['rating_cap_high'], away_pit))
            
            home_lambda = (home_off / away_pit) * lg_avg_runs
            away_lambda = (away_off / home_pit) * lg_avg_runs
            
            home_win_pct, away_win_pct, avg_total = run_simulation(home_lambda, away_lambda)
            
            fd_home_ml = game.get('fd_home_ml')
            fd_away_ml = game.get('fd_away_ml')
            
            if pd.isna(fd_home_ml) or pd.isna(fd_away_ml):
                continue
            
            home_market_prob = american_to_prob(fd_home_ml)
            away_market_prob = american_to_prob(fd_away_ml)
            home_edge = home_win_pct - home_market_prob
            away_edge = away_win_pct - away_market_prob
            
            home_score = game.get('home_score')
            away_score = game.get('away_score')
            if pd.isna(home_score) or pd.isna(away_score):
                continue
            
            actual_winner = 'home' if home_score > away_score else 'away'
            
            ml_bet = None
            ml_odds = None
            ml_result = None
            ml_edge_val = None
            
            if home_edge > params['ml_edge_min'] and home_edge <= params['ml_edge_max']:
                if not params['favorites_only'] or fd_home_ml < 0:
                    ml_bet = 'home'
                    ml_odds = fd_home_ml
                    ml_edge_val = home_edge
                    ml_result = 'WIN' if actual_winner == 'home' else 'LOSS'
            elif away_edge > params['ml_edge_min'] and away_edge <= params['ml_edge_max']:
                if not params['favorites_only'] or fd_away_ml < 0:
                    ml_bet = 'away'
                    ml_odds = fd_away_ml
                    ml_edge_val = away_edge
                    ml_result = 'WIN' if actual_winner == 'away' else 'LOSS'
            
            if ml_bet:
                all_results.append({
                    'ml_odds': ml_odds,
                    'ml_result': ml_result
                })
    
    if not all_results:
        return None
    
    results_df = pd.DataFrame(all_results)
    wins = len(results_df[results_df['ml_result'] == 'WIN'])
    n_bets = len(results_df)
    win_rate = wins / n_bets
    total_bet = n_bets * 100
    total_return = 0
    for _, row in results_df.iterrows():
        if row['ml_result'] == 'WIN':
            odds = row['ml_odds']
            if odds > 0:
                total_return += 100 + (100 * odds / 100)
            else:
                total_return += 100 + (100 / abs(odds) * 100)
    profit = total_return - total_bet
    roi = profit / total_bet * 100
        
    return {
        'n_bets': n_bets,
        'win_rate': round(win_rate, 4),
        'roi': round(roi, 4),
        'profit': round(profit, 2)
    }

def get_season_ratings(year):
    offense, pitching = load_season_stats(year)
    
    woba_fip = pd.read_csv('constants/woba_fip_constants.csv')
    row = woba_fip[woba_fip['Season'] == year].iloc[0]
    fip_constant = row['cFIP']
    
    team_ratings = {}
    
    for _, off_row in offense.iterrows():
        team = off_row['team']
        team_ratings[team] = {
            'woba': float(off_row.get('woba', 0.320)),
            'xwoba': float(off_row.get('xwoba', 0.320)),
            'k_pct_off': float(off_row.get('k_pct', 0.220)),
            'bb_pct_off': float(off_row.get('bb_pct', 0.085))
        }
    
    for _, pit_row in pitching.iterrows():
        team = pit_row['team']
        if team not in team_ratings:
            team_ratings[team] = {}
        team_ratings[team]['fip'] = float(pit_row.get('fip', 4.0))
        team_ratings[team]['xfip'] = float(pit_row.get('xfip', 4.0))
        team_ratings[team]['k_9_pit'] = float(pit_row.get('k_9', 8.0))
        team_ratings[team]['bb_9_pit'] = float(pit_row.get('bb_9', 3.0))
        team_ratings[team]['gb_pct_pit'] = float(pit_row.get('gb_pct', 0.44))
            
    return team_ratings

def calculate_roi(bets_df, bet_col, odds_col, result_col):
    total_bet = len(bets_df) * 100
    total_return = 0
    for _, row in bets_df.iterrows():
        if row[result_col] == 'WIN':
            odds = row[odds_col]
            if odds > 0:
                total_return += 100 + (100 * odds / 100)
            else:
                total_return += 100 + (100 / abs(odds) * 100)
        
    profit = total_return - total_bet
    roi = profit / total_bet * 100
    return roi, profit

# - delete everything before print when ready to run - print("\nStarting random search - 100 trials...")
# - delete everything before search when ready to runsearch_results = random_search(n_trials=100)
