import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import math
import requests
import statsapi
import json
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=r'C:\Users\super\baseball-model\.env')
from datetime import date
import numpy as np
from io import StringIO

# Team Abbreviation Translation - MLB API to Fangraphs
TEAM_MAP = {
    'ATH': 'ATH',
    'PIT': 'PIT',
    'SD': 'SDP',
    'SEA': 'SEA',
    'SF': 'SFG',
    'STL': 'STL',
    'TB': 'TBR',
    'TEX': 'TEX',
    'TOR': 'TOR',
    'MIN': 'MIN',
    'PHI': 'PHI',
    'ATL': 'ATL',
    'CWS': 'CHW',
    'MIA': 'MIA',
    'NYY': 'NYY',
    'MIL': 'MIL',
    'LAA': 'LAA',
    'AZ': 'ARI',
    'BAL': 'BAL',
    'BOS': 'BOS',
    'CHC': 'CHC',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'COL': 'COL',
    'DET': 'DET',
    'HOU': 'HOU',
    'KC': 'KCR',
    'LAD': 'LAD',
    'WSH': 'WSN',
    'NYM': 'NYM'
}

NAME_TO_FG = {
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
    'New York Mets': 'NYM'
}

PARK_FACTOR_TEAM_MAP = {
    'Angels': 'LAA', 'Orioles': 'BAL', 'Red Sox': 'BOS',
    'White Sox': 'CHW', 'Guardians': 'CLE', 'Tigers': 'DET',
    'Royals': 'KCR', 'Twins': 'MIN', 'Yankees': 'NYY',
    'Athletics': 'ATH', 'Mariners': 'SEA', 'Rays': 'TBR',
    'Rangers': 'TEX', 'Blue Jays': 'TOR', 'Diamondbacks': 'ARI',
    'Braves': 'ATL', 'Cubs': 'CHC', 'Reds': 'CIN',
    'Rockies': 'COL', 'Dodgers': 'LAD', 'Marlins': 'MIA',
    'Brewers': 'MIL', 'Mets': 'NYM', 'Phillies': 'PHI',
    'Pirates': 'PIT', 'Cardinals': 'STL', 'Padres': 'SDP',
    'Giants': 'SFG', 'Nationals': 'WSN', 'Astros': 'HOU'
}

from datetime import timezone, timedelta, datetime

API_KEY = os.getenv('API_KEY')
print(f"API Key loaded: {API_KEY is not None}")

MST = timezone(timedelta(hours=-7))

# Starter/bullpen split
STARTER_INNINGS = 5
TOTAL_INNINGS = 9
STARTER_WEIGHT = STARTER_INNINGS / TOTAL_INNINGS
BULLPEN_WEIGHT = 1 - STARTER_WEIGHT
MIN_INNINGS = 15
MAX_RA9 = 7.00

# Lambda weights - adjust for backtesting
OFFENSE_WEIGHTS = {
    'rolling': 1/5,
    'woba': 1/5,
    'xwoba': 1/5,
    'k_pct': 1/5,
    'bb_pct': 1/5
}

PITCHING_WEIGHTS = {
    'rolling': 1/5,
    'fip': 1/5,
    'xfip': 1/5,
    'k_pct': 1/5,
    'bb_pct': 1/5
}

NUM_SIMULATIONS = 5000

def load_constants(year=2026):
    woba_fip = pd.read_csv('constants/woba_fip_constants.csv')
    row = woba_fip[woba_fip['Season'] == year].iloc[0]
    
    woba_weights = {
        'wBB': row['wBB'],
        'wHBP': row['wHBP'],
        'w1B': row['w1B'],
        'w2B': row['w2B'],
        'w3B': row['w3B'],
        'wHR': row['wHR']
    }
    fip_constant = row['cFIP']
    
    park_factors = pd.read_csv('constants/park_factors.csv')
    park_factors_hand = pd.read_csv('constants/park_factors_handedness.csv')
    
    return woba_weights, fip_constant, park_factors, park_factors_hand

WOBA_WEIGHTS, FIP_CONSTANT, PARK_FACTORS, PARK_FACTORS_HAND = load_constants(2026)

# Run schedule for predictions log
RUN_SCHEDULE = {1: '9am', 2: '12pm', 3: '4pm'}

def american_to_prob(line):
    if line < 0:
        return abs(line) / (abs(line) + 100)
    else:
        return 100 / (line + 100)

def get_cached_odds():
    cache_file='odds_cache.json'

    # Check if cache exists and is fresh
    if os.path.exists(cache_file):
        modified_time = os.path.getmtime(cache_file)
        from datetime import datetime
        age_hours = (datetime.now().timestamp() - modified_time) / 3600
        if age_hours < 2.5:
            print ("Loading odds from cache. . .")
            with open(cache_file,'r') as f:
                return json.load(f)
            
    # Cache is old or doesn't exist - call the API
    print("Fetching fresh odds from API . . .")
    data = get_mlb_odds()

    # Save to cache
    with open(cache_file,'w') as f:
        json.dump(data,f)

    return data

def get_upcoming_games(odds_data):
    now = datetime.now(timezone.utc)
    today_mst = now.astimezone(MST).date()
    upcoming = []
    for game in odds_data:
        game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        game_time_mst = game_time.astimezone(MST)
        if game_time > now and game_time_mst.date() == today_mst:
            upcoming.append(game)
    return upcoming

def get_todays_game_statuses():
    today = date.today().strftime('%Y-%m-%d')
    schedule = statsapi.schedule(date=today)
    statuses = {}
    for game in schedule:
        home = game['home_name']
        away = game['away_name']
        if home in NAME_TO_FG and away in NAME_TO_FG:
            key = f"{NAME_TO_FG[away]}_{NAME_TO_FG[home]}"
            statuses[key] = {
                'status': game['status'],
                'home_score': game.get('home_score', '-'),
                'away_score': game.get('away_score', '-')
            }
    return statuses

def get_mlb_odds():
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={API_KEY}&regions=us&markets=h2h,totals&oddsFormat=american&bookmakers=draftkings,fanduel'
    response = requests.get(url)
    data = response.json()
    return data

def get_park_factor(home_team, season=2025):
    pf_df = PARK_FACTORS[PARK_FACTORS['Season'] == season]
    team_name = next((k for k, v in PARK_FACTOR_TEAM_MAP.items() if v == home_team), None)
    if team_name is None:
        return 1.0
    row = pf_df[pf_df['Team'] == team_name]
    if row.empty:
        return 1.0
    return row['1yr'].values[0] / 100

# Load offense data
offense_2021 = pd.read_csv('offense_2021.csv')
offense_2022 = pd.read_csv('offense_2022.csv')
offense_2023 = pd.read_csv('offense_2023.csv')
offense_2024 = pd.read_csv('offense_2024.csv')
offense_2025 = pd.read_csv('offense_2025.csv')
offense_2026 = pd.read_csv('offense_2026.csv')

# Assign years
offense_2021['year'] = 2021
offense_2022['year'] = 2022
offense_2023['year'] = 2023
offense_2024['year'] = 2024
offense_2025['year'] = 2025
offense_2026['year'] = 2026

# Combine offense
offense = pd.concat([offense_2021, offense_2022, offense_2023, offense_2024,offense_2025,offense_2026])

# Load pitching data
pitching_2021 = pd.read_csv('pitching_2021.csv')
pitching_2022 = pd.read_csv('pitching_2022.csv')
pitching_2023 = pd.read_csv('pitching_2023.csv')
pitching_2024 = pd.read_csv('pitching_2024.csv')
pitching_2025 = pd.read_csv('pitching_2025.csv')
pitching_2026 = pd.read_csv('pitching_2026.csv')

# Assign years
pitching_2021['year'] = 2021
pitching_2022['year'] = 2022
pitching_2023['year'] = 2023
pitching_2024['year'] = 2024
pitching_2025['year'] = 2025
pitching_2026['year'] = 2026

# Combine pitching
pitching = pd.concat([pitching_2021, pitching_2022, pitching_2023, pitching_2024,pitching_2025,pitching_2026])

# Normalize column names
offense.columns = offense.columns.str.strip().str.lower().str.replace('+', 'plus', regex=False).str.replace('/', '_', regex=False).str.replace(' ', '_', regex=False).str.replace('%', '_pct', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)
pitching.columns = pitching.columns.str.strip().str.lower().str.replace('+', 'plus', regex=False).str.replace('/', '_', regex=False).str.replace(' ', '_', regex=False).str.replace('%', '_pct', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)

# Calculate runs per game for each team
offense['r_per_game'] = offense['r'] / offense['tg']
offense['sample_weight'] = (offense['tg']/162).clip(0,1)

# Calculate league average runs per game for each year
league_avg = offense.groupby('year')['r_per_game'].mean().reset_index()
league_avg.columns = ['year', 'lg_avg_runs']

# Merge league average into offense data
offense = offense.merge(league_avg, on='year')

# Calculate offensive rating relative to league average
offense['off_rating'] = offense['r_per_game']/offense['lg_avg_runs']

# Calculate league average FIP per year
lg_fip = pitching.groupby('year')['fip'].mean().reset_index()
lg_fip.columns = ['year','lg_avg_fip']

# Merge into pitching data
pitching = pitching.merge(lg_fip,on='year')

# Calculate pitching rating - not we invert it so better pitching = higher number
# A team with FIP below league average is BETTER, so we flip the ratio
pitching['pitch_rating'] = pitching['lg_avg_fip'] / pitching['fip']

model_data = offense[['team','year','r_per_game','off_rating']].merge(pitching[['team','year','pitch_rating']],on=['team','year'])

# Calculate lamba = expected runs scored against average pitching
# Adjusted for the opponent pitching quality
model_data = model_data.merge(league_avg, on='year')
model_data['lambda'] = model_data['off_rating']*model_data['lg_avg_runs']
  
# Calculate expected runs for both teams - New Beginning
def calculate_matchup(home_team, away_team, year, rolling=None, starters=None):
    if rolling and home_team in rolling and away_team in rolling:
        lg_avg_runs = league_avg[league_avg['year'] == 2026]['lg_avg_runs'].values[0]
        
        home_off_rating, home_pit_rating = calculate_team_ratings(
            home_team, rolling, all_woba, all_fip, team_xwoba, lg_avgs, lg_avg_runs)
        away_off_rating, away_pit_rating = calculate_team_ratings(
            away_team, rolling, all_woba, all_fip, team_xwoba, lg_avgs, lg_avg_runs)
        
        # Get park factor for home stadium
        park_factor = get_park_factor(home_team)
        
        # Calculate blended pitching including starter
        home_pitch_roll = rolling[home_team]['pitch_7'] * 0.60 + rolling[home_team]['pitch_15'] * 0.40
        away_pitch_roll = rolling[away_team]['pitch_7'] * 0.60 + rolling[away_team]['pitch_15'] * 0.40

        if starters and home_team in starters and starters[home_team]:
            home_innings = starters[home_team]['innings']
            home_ra9 = starters[home_team]['ra9']
            if home_innings >= MIN_INNINGS:
                home_starter_ra9 = min(home_ra9, MAX_RA9)
                home_blended_pitch = (home_starter_ra9 * STARTER_WEIGHT) + (home_pitch_roll * BULLPEN_WEIGHT)
            else:
                home_blended_pitch = home_pitch_roll
        else:
            home_blended_pitch = home_pitch_roll

        if starters and away_team in starters and starters[away_team]:
            away_innings = starters[away_team]['innings']
            away_ra9 = starters[away_team]['ra9']
            if away_innings >= MIN_INNINGS:
                away_starter_ra9 = min(away_ra9, MAX_RA9)
                away_blended_pitch = (away_starter_ra9 * STARTER_WEIGHT) + (away_pitch_roll * BULLPEN_WEIGHT)
            else:
                away_blended_pitch = away_pitch_roll
        else:
            away_blended_pitch = away_pitch_roll

        # Convert blended pitch to rating
        home_starter_pit_rating = lg_avg_runs / home_blended_pitch if home_blended_pitch > 0 else 1.0
        away_starter_pit_rating = lg_avg_runs / away_blended_pitch if away_blended_pitch > 0 else 1.0

        # Blend starter rating with full pitching rating
        home_final_pit = (home_starter_pit_rating * 0.50) + (home_pit_rating * 0.50)
        away_final_pit = (away_starter_pit_rating * 0.50) + (away_pit_rating * 0.50)

        # Calculate lambdas with park factor
        home_lambda = (home_off_rating / away_final_pit) * lg_avg_runs * park_factor
        away_lambda = (away_off_rating / home_final_pit) * lg_avg_runs * park_factor

    else:
        home_off = model_data[(model_data['team'] == home_team) & (model_data['year'] == year)]['lambda'].values[0]
        away_off = model_data[(model_data['team'] == away_team) & (model_data['year'] == year)]['lambda'].values[0]
        home_pitch = model_data[(model_data['team'] == home_team) & (model_data['year'] == year)]['pitch_rating'].values[0]
        away_pitch = model_data[(model_data['team'] == away_team) & (model_data['year'] == year)]['pitch_rating'].values[0]
        home_lambda = home_off / away_pitch
        away_lambda = away_off / home_pitch


    home_scores = np.random.poisson(home_lambda, NUM_SIMULATIONS)
    away_scores = np.random.poisson(away_lambda, NUM_SIMULATIONS)

    ties = home_scores == away_scores
    extra_home = np.random.poisson(home_lambda * 0.111, ties.sum())
    extra_away = np.random.poisson(away_lambda * 0.111, ties.sum())

    home_scores[ties] += extra_home
    away_scores[ties] += extra_away

    home_wins_sim = np.sum(home_scores > away_scores)
    away_wins_sim = np.sum(away_scores > home_scores)
    total_runs = home_scores + away_scores

    remaining = NUM_SIMULATIONS - home_wins_sim - away_wins_sim
    home_win_pct = (home_wins_sim + remaining / 2) / NUM_SIMULATIONS
    away_win_pct = (away_wins_sim + remaining / 2) / NUM_SIMULATIONS

    avg_total = np.mean(total_runs)

    return home_win_pct, away_win_pct, avg_total, home_lambda, away_lambda

def calculate_team_woba(team_id, season=2026):
    try:
        stats = statsapi.get('team_stats', {
            'teamId': team_id,
            'stats': 'season',
            'group': 'hitting',
            'season': season
        })
        s = stats['stats'][0]['splits'][0]['stat']
        
        bb = float(s.get('baseOnBalls', 0))
        hbp = float(s.get('hitByPitch', 0))
        hits = float(s.get('hits', 0))
        doubles = float(s.get('doubles', 0))
        triples = float(s.get('triples', 0))
        hr = float(s.get('homeRuns', 0))
        ab = float(s.get('atBats', 0))
        ibb = float(s.get('intentionalWalks', 0))
        sf = float(s.get('sacFlies', 0))
        pa = float(s.get('plateAppearances', 0))
        
        singles = hits - doubles - triples - hr
        
        numerator = (WOBA_WEIGHTS['wBB'] * bb + 
                    WOBA_WEIGHTS['wHBP'] * hbp + 
                    WOBA_WEIGHTS['w1B'] * singles + 
                    WOBA_WEIGHTS['w2B'] * doubles + 
                    WOBA_WEIGHTS['w3B'] * triples + 
                    WOBA_WEIGHTS['wHR'] * hr)
        
        denominator = ab + bb - ibb + sf + hbp
        
        if denominator == 0:
            return None
            
        woba = numerator / denominator
        k_pct = float(s.get('strikeOuts', 0)) / pa if pa > 0 else 0
        bb_pct = bb / pa if pa > 0 else 0
        
        return {
            'woba': round(woba, 3),
            'k_pct': round(k_pct, 3),
            'bb_pct': round(bb_pct, 3)
        }
    except Exception as e:
        return None

def compare_to_market(game, model_home_prob, model_away_prob):
    home_team = game['home_team']
    away_team = game['away_team']
    
    if len(game['bookmakers']) == 0:
        return
    
    bookmaker = game['bookmakers'][0]
    outcomes = bookmaker['markets'][0]['outcomes']
    
    for outcome in outcomes:
        if outcome['name'] == home_team:
            market_home_line = outcome['price']
        else:
            market_away_line = outcome['price']
    
    market_home_prob = american_to_prob(market_home_line)
    market_away_prob = american_to_prob(market_away_line)
    
    home_edge = model_home_prob - market_home_prob
    away_edge = model_away_prob - market_away_prob
        
def get_team_logs(team_id, season=2026):
    try:
        stats=statsapi.get('team_stats', {'teamId':team_id,'stats':'gamelog','group':'hitting','season':season})
        games=stats['stats'][0]['splits']
        log = []
        for game in games:
            log.append({'date':game['date'],'runs':game['stat']['runs'],'atBats':game['stat']['atBats']})
        return log
    except Exception as e:
        print(f"Error fetching game log: {e}")
        return[]

def get_rolling_average(game_log, days=7, column='runs'):
    if not game_log or len(game_log) ==0:
        return None
    df = pd.DataFrame(game_log)
    df = df.sort_values('date')
    recent = df.tail(days)
    return recent[column].mean()
    
def get_pitching_game_logs(team_id,season=2026):
    try:
        stats=statsapi.get('team_stats',{'teamId': team_id,'stats':'gamelog','group':'pitching','season':season})
        games=stats['stats'][0]['splits']
        log = []
        for game in games:
            log.append({'date':game['date'],'runs_allowed':game['stat']['runs'],'inningsPitched': game['stat']['inningsPitched']})
        return log
    except Exception as e:
        print(f"Error fetching pitching log: {e}")
        return []


def get_all_team_ids():
    teams = statsapi.get('teams',{'sportId':1,'season':2026})
    teams_ids={}
    for team in teams['teams']:
        abbrev=team.get('abbreviation')
        if abbrev in TEAM_MAP:
            fg_abbrev=TEAM_MAP[abbrev]
            teams_ids[fg_abbrev]=team['id']
    return teams_ids

def get_all_team_woba(team_ids, season=2026):
    results = {}
    for fg_abbrev, team_id in team_ids.items():
        stats = calculate_team_woba(team_id, season)
        if stats:
            results[fg_abbrev] = stats
    return results

def get_team_xwoba(team_ids, season=2026):
    try:
        url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={season}&position=&team=&min=10&csv=true"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0'}
        response = requests.get(url, headers=headers)
        xwoba_df = pd.read_csv(StringIO(response.text))
        
        player_team = {}
        for fg_abbrev, team_id in team_ids.items():
            try:
                roster_data = statsapi.get('team_roster', {
                    'teamId': team_id,
                    'rosterType': 'active'
                })
                for player in roster_data['roster']:
                    player_team[player['person']['id']] = fg_abbrev
            except:
                continue
        
        xwoba_df['team'] = xwoba_df['player_id'].map(player_team)
        xwoba_df = xwoba_df.dropna(subset=['team'])
        
        team_xwoba = xwoba_df.groupby('team').apply(
            lambda x: pd.Series({
                'xwoba': (x['est_woba'] * x['pa']).sum() / x['pa'].sum(),
                'woba_savant': (x['woba'] * x['pa']).sum() / x['pa'].sum()
            })
        ).reset_index()
        
        return dict(zip(team_xwoba['team'], team_xwoba['xwoba']))
    except Exception as e:
        print(f"xwOBA fetch failed: {e}")
        return {}

def calculate_lg_hr_fb(team_ids, season=2026):
    total_hr = 0
    total_fb = 0
    for fg_abbrev, team_id in team_ids.items():
        try:
            stats = statsapi.get('team_stats', {
                'teamId': team_id,
                'stats': 'season',
                'group': 'pitching',
                'season': season
            })
            s = stats['stats'][0]['splits'][0]['stat']
            hr = float(s.get('homeRuns', 0))
            air_outs = float(s.get('airOuts', 0))
            total_hr += hr
            total_fb += (air_outs + hr)
        except:
            continue
    if total_fb == 0:
        return 0.115
    return total_hr / total_fb

def calculate_team_fip(team_id, lg_hr_fb, season=2026):
    try:
        stats = statsapi.get('team_stats', {
            'teamId': team_id,
            'stats': 'season',
            'group': 'pitching',
            'season': season
        })
        s = stats['stats'][0]['splits'][0]['stat']
        
        hr = float(s.get('homeRuns', 0))
        bb = float(s.get('baseOnBalls', 0))
        hbp = float(s.get('hitBatsmen', 0))
        k = float(s.get('strikeOuts', 0))
        air_outs = float(s.get('airOuts', 0))
        bf = float(s.get('battersFaced', 1))
        ip_str = s.get('inningsPitched', '0')
        ip = float(ip_str)
        
        if ip == 0:
            return None
        
        fb = air_outs + hr
        expected_hr = fb * lg_hr_fb
        
        fip = ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + FIP_CONSTANT
        xfip = ((13 * expected_hr) + (3 * (bb + hbp)) - (2 * k)) / ip + FIP_CONSTANT
        
        k_pct = k / bf
        bb_pct = bb / bf
        
        return {
            'fip': round(fip, 3),
            'xfip': round(xfip, 3),
            'k_pct': round(k_pct, 3),
            'bb_pct': round(bb_pct, 3),
            'lg_hr_fb': round(lg_hr_fb, 4)
        }
    except Exception as e:
        return None

def get_all_team_fip(team_ids, season=2026):
    lg_hr_fb = calculate_lg_hr_fb(team_ids, season)
    results = {}
    for fg_abbrev, team_id in team_ids.items():
        stats = calculate_team_fip(team_id, lg_hr_fb, season)
        if stats:
            results[fg_abbrev] = stats
    return results

def get_all_rolling_averages(team_ids, days_short=7, days_long=15):
    results = {}
    for fg_abbrev, team_id in team_ids.items():
        hitting_log = get_team_logs(team_id)
        pitching_log = get_pitching_game_logs(team_id)
        hit_short = get_rolling_average(hitting_log, days=days_short)
        hit_long = get_rolling_average(hitting_log, days=days_long)
        pitch_short = get_rolling_average(pitching_log, days=days_short, column='runs_allowed')
        pitch_long = get_rolling_average(pitching_log, days=days_long, column='runs_allowed')
        results[fg_abbrev] = {
            'hit_7': hit_short,
            'hit_15': hit_long,
            'pitch_7': pitch_short,
            'pitch_15': pitch_long
        }
    return results

from datetime import datetime
run_time = datetime.now().strftime('%H:%M')
todays_predictions = []

def get_yesterdays_results():
    yesterday = (date.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    schedule = statsapi.schedule(date=yesterday)
    results = {}
    for game in schedule:
        if game['status'] == 'Final':
            home = game['home_name']
            away = game['away_name']
            home_score = game['home_score']
            away_score = game['away_score']
            winner = home if home_score > away_score else away
            if home in NAME_TO_FG and away in NAME_TO_FG:
                results[f"{NAME_TO_FG[away]}_{NAME_TO_FG[home]}"] = {
                    'home': NAME_TO_FG[home],
                    'away': NAME_TO_FG[away],
                    'home_score': home_score,
                    'away_score': away_score,
                    'winner': NAME_TO_FG[home] if home_score > away_score else NAME_TO_FG[away]
                }
    return results

yesterdays_results = get_yesterdays_results()

def log_predictions(predictions, results):
    log_file = 'predictions_log.csv'
    
    # Determine which run this is based on current hour
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 11:
        run_num = 1
        run_label = '9am'
    elif hour < 15:
        run_num = 2
        run_label = '12pm'
    else:
        run_num = 3
        run_label = '4pm'
    
    if not predictions:
        return

    # Load existing log or create new one
    columns = [
        'date', 'home_team', 'away_team', 'bet_type',
        'run1_bet_team', 'run1_model_pct', 'run1_book_line', 'run1_edge',
        'run2_bet_team', 'run2_model_pct', 'run2_book_line', 'run2_edge',
        'run3_bet_team', 'run3_model_pct', 'run3_book_line', 'run3_edge',
        'home_score', 'away_score', 'winner', 'result', 'flag'
    ]
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file, dtype=str)
    else:
        df = pd.DataFrame(columns=columns)
    
    for pred in predictions:
        game_date = pred['date']
        home_fg = pred['home_fg']
        away_fg = pred['away_fg']
        bet_type = pred['bet_type']
        
        # Find existing row for this game and bet type
        mask = (df['date'] == game_date) & \
               (df['home_team'] == home_fg) & \
               (df['away_team'] == away_fg) & \
               (df['bet_type'] == bet_type)
        
        if df[mask].empty:
            # Create new row
            new_row = {col: None for col in columns}
            new_row['date'] = game_date
            new_row['home_team'] = home_fg
            new_row['away_team'] = away_fg
            new_row['bet_type'] = bet_type
            new_row[f'run{run_num}_bet_team'] = pred['bet_team']
            new_row[f'run{run_num}_model_pct'] = pred['model_pct']
            new_row[f'run{run_num}_book_line'] = pred['book_line']
            new_row[f'run{run_num}_edge'] = pred['edge']
            
            # Add results if available
            game_key = f"{away_fg}_{home_fg}"
            if game_key in results:
                r = results[game_key]
                new_row['home_score'] = str(r['home_score'])
                new_row['away_score'] = str(r['away_score'])
                new_row['winner'] = r['winner']
                new_row['result'] = evaluate_result(pred, r)
            
            new_row['flag'] = 'CONSISTENT'
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Update existing row
            idx = df[mask].index[0]
            df.at[idx, f'run{run_num}_bet_team'] = pred['bet_team']
            df.at[idx, f'run{run_num}_model_pct'] = pred['model_pct']
            df.at[idx, f'run{run_num}_book_line'] = pred['book_line']
            df.at[idx, f'run{run_num}_edge'] = pred['edge']
            
            # Add results if available
            game_key = f"{away_fg}_{home_fg}"
            if game_key in results:
                r = results[game_key]
                df.at[idx, 'home_score'] = str(r['home_score'])
                df.at[idx, 'away_score'] = str(r['away_score'])
                df.at[idx, 'winner'] = r['winner']
                df.at[idx, 'result'] = evaluate_result(pred, r)
            
            # Update flag
            df.at[idx, 'flag'] = evaluate_flag(df.iloc[idx])
    
    df.to_csv(log_file, index=False)
    generate_excel_log(df)
    update_google_sheets(df)

def evaluate_result(pred, game_result):
    bet_type = pred['bet_type']
    if bet_type == 'Moneyline':
        if game_result['winner'] == pred['bet_team']:
            return 'WIN'
        elif game_result['winner'] is None:
            return 'PENDING'
        else:
            return 'LOSS'
    elif bet_type in ['Over', 'Under']:
        home_score = game_result.get('home_score')
        away_score = game_result.get('away_score')
        if home_score is None or away_score is None:
            return 'PENDING'
        total = float(home_score) + float(away_score)
        book_line = float(pred['book_line'])
        if bet_type == 'Over':
            if total > book_line:
                return 'WIN'
            elif total == book_line:
                return 'PUSH'
            else:
                return 'LOSS'
        else:
            if total < book_line:
                return 'WIN'
            elif total == book_line:
                return 'PUSH'
            else:
                return 'LOSS'
    return 'PENDING'

def evaluate_flag(row):
    runs = []
    for i in range(1, 4):
        team = row.get(f'run{i}_bet_team')
        edge = row.get(f'run{i}_edge')
        if team is not None and edge is not None:
            runs.append((i, team, edge))
    
    if len(runs) <= 1:
        return 'CONSISTENT'
    
    teams = [r[1] for r in runs]
    if len(set(teams)) > 1:
        return 'TEAM CHANGED'
    
    # Check if edge was lost or gained
    has_edge = [r[2] != 'NO BET' for r in runs]
    if True in has_edge and False in has_edge:
        if has_edge[0] and not has_edge[-1]:
            return 'EDGE LOST'
        elif not has_edge[0] and has_edge[-1]:
            return 'EDGE GAINED'
    
    return 'CONSISTENT'

def generate_excel_log(df):
    try:
        import openpyxl
        from openpyxl.styles import PatternFill
        
        excel_file = 'predictions_log.xlsx'
        df.to_excel(excel_file, index=False)
        
        wb = openpyxl.load_workbook(excel_file)
        ws = wb.active
        
        green = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
        red = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
        yellow = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
        
        flag_col = df.columns.get_loc('flag') + 1
        result_col = df.columns.get_loc('result') + 1
        
        for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
            flag = ws.cell(row=row_idx, column=flag_col).value
            result = ws.cell(row=row_idx, column=result_col).value
            
            if flag in ['TEAM CHANGED', 'EDGE LOST', 'EDGE GAINED']:
                fill = yellow
            elif result == 'WIN':
                fill = green
            elif result == 'LOSS':
                fill = red
            else:
                fill = None
            
            if fill:
                for cell in row:
                    cell.fill = fill
        
        wb.save(excel_file)
    except Exception as e:
        print(f"Excel generation failed: {e}")

def calculate_rolling_lambda(team,rolling):
    if team not in rolling:
        return None
    
    lg_avg = league_avg[league_avg['year'] == 2026]['lg_avg_runs'].values[0]
    
    hit_7 = rolling[team]['hit_7']
    hit_15 = rolling[team]['hit_15']
    pitch_7 = rolling[team]['pitch_7']
    pitch_15 = rolling[team]['pitch_15']

    off_rating_7 = hit_7 / lg_avg
    off_rating_15 = hit_15 / lg_avg
    pitch_rating_7 = lg_avg / pitch_7
    pitch_rating_15 = lg_avg / pitch_15

    off_rating = (off_rating_7 * 0.60) + (off_rating_15 * 0.40)
    pitch_rating = (pitch_rating_7 * 0.60) + (pitch_rating_15 * 0.40)

    off_rating = max(0.60, min(1.60, off_rating))
    pitch_rating = max(0.60, min(1.60, pitch_rating))

    return {
        'off_rating': off_rating,
        'pitch_rating': pitch_rating
    }

def get_pitcher_stats(pitcher_name, season=2026):
    try:
        search = statsapi.lookup_player(pitcher_name)
        if not search:
            return None
        pitcher_id = search[0]['id']
        stats = statsapi.player_stat_data(pitcher_id, group='pitching', type='season', sportId=1)
        if not stats['stats']:
            return None
        s = stats['stats'][0]['stats']
        innings = float(s.get('inningsPitched',0))
        runs = int(s.get('runs',0))
        ra9 = round((runs / innings * 9), 2) if innings > 0 else None
        return {
            'name': pitcher_name,
            'era': float(s.get('era', 0)),
            'innings': innings,
            'runs_allowed': runs,
            'games_started': int(s.get('gamesStarted', 0)),
            'ra9': ra9
        }
    except Exception as e:
        return None

def get_todays_starters(game_date=None):
    if game_date is None:
        game_date = date.today().strftime('%Y-%m-%d')
    schedule = statsapi.schedule(date=game_date)
    starters = {}
    for game in schedule:
        home = game['home_name']
        away = game['away_name']
        home_pitcher = game.get('home_probable_pitcher', '')
        away_pitcher = game.get('away_probable_pitcher', '')
        if home in NAME_TO_FG:
            home_fg = NAME_TO_FG[home]
            starters[home_fg] = get_pitcher_stats(home_pitcher) if home_pitcher else None
        if away in NAME_TO_FG:
            away_fg = NAME_TO_FG[away]
            starters[away_fg] = get_pitcher_stats(away_pitcher) if away_pitcher else None
    return starters

def get_total_line(game):
    best_total = None
    best_over_price = None
    best_under_price = None
    best_total_book = None
    
    for bookmaker in game['bookmakers']:
        for market in bookmaker['markets']:
            if market['key'] == 'totals':
                for outcome in market['outcomes']:
                    if outcome['name'] == 'Over':
                        if best_over_price is None or outcome['price'] > best_over_price:
                            best_over_price = outcome['price']
                            best_total = outcome['point']
                            best_total_book = bookmaker['title']
                    elif outcome['name'] == 'Under':
                        if best_under_price is None or outcome['price'] > best_under_price:
                            best_under_price = outcome['price']
    
    if best_total is None:
        return None
    
    return {
        'total': best_total,
        'over_price': best_over_price,
        'under_price': best_under_price,
        'book': best_total_book
    }

def calculate_league_averages(all_woba, all_fip, team_xwoba):
    # Offensive averages
    lg_woba = np.mean([v['woba'] for v in all_woba.values()])
    lg_xwoba = np.mean(list(team_xwoba.values()))
    lg_k_pct_off = np.mean([v['k_pct'] for v in all_woba.values()])
    lg_bb_pct_off = np.mean([v['bb_pct'] for v in all_woba.values()])
    
    # Pitching averages
    lg_fip = np.mean([v['fip'] for v in all_fip.values()])
    lg_xfip = np.mean([v['xfip'] for v in all_fip.values()])
    lg_k_pct_pit = np.mean([v['k_pct'] for v in all_fip.values()])
    lg_bb_pct_pit = np.mean([v['bb_pct'] for v in all_fip.values()])
    
    return {
        'lg_woba': lg_woba,
        'lg_xwoba': lg_xwoba,
        'lg_k_pct_off': lg_k_pct_off,
        'lg_bb_pct_off': lg_bb_pct_off,
        'lg_fip': lg_fip,
        'lg_xfip': lg_xfip,
        'lg_k_pct_pit': lg_k_pct_pit,
        'lg_bb_pct_pit': lg_bb_pct_pit
    }

def calculate_team_ratings(team, rolling, all_woba, all_fip, team_xwoba, lg_avgs, lg_avg_runs):
    # Get rolling averages
    hit_7 = rolling[team]['hit_7']
    hit_15 = rolling[team]['hit_15']
    pitch_7 = rolling[team]['pitch_7']
    pitch_15 = rolling[team]['pitch_15']
    rolling_off = (hit_7 * 0.60) + (hit_15 * 0.40)
    rolling_pit = (pitch_7 * 0.60) + (pitch_15 * 0.40)

    # Convert to ratings relative to league average
    rolling_off_rating = rolling_off / lg_avg_runs
    rolling_pit_rating = lg_avg_runs / rolling_pit if rolling_pit > 0 else 1.0

    # Offensive ratings
    woba_rating = all_woba[team]['woba'] / lg_avgs['lg_woba'] if team in all_woba else 1.0
    xwoba_rating = team_xwoba[team] / lg_avgs['lg_xwoba'] if team in team_xwoba else 1.0
    
    # K% and BB% for offense - higher K% is bad, higher BB% is good
    k_off_rating = lg_avgs['lg_k_pct_off'] / all_woba[team]['k_pct'] if team in all_woba and all_woba[team]['k_pct'] > 0 else 1.0
    bb_off_rating = all_woba[team]['bb_pct'] / lg_avgs['lg_bb_pct_off'] if team in all_woba and lg_avgs['lg_bb_pct_off'] > 0 else 1.0

    # Pitching ratings - lower FIP/xFIP is better so we invert
    fip_rating = lg_avgs['lg_fip'] / all_fip[team]['fip'] if team in all_fip and all_fip[team]['fip'] > 0 else 1.0
    xfip_rating = lg_avgs['lg_xfip'] / all_fip[team]['xfip'] if team in all_fip and all_fip[team]['xfip'] > 0 else 1.0
    
    # K% and BB% for pitching - higher K% is good, higher BB% is bad
    k_pit_rating = all_fip[team]['k_pct'] / lg_avgs['lg_k_pct_pit'] if team in all_fip and lg_avgs['lg_k_pct_pit'] > 0 else 1.0
    bb_pit_rating = lg_avgs['lg_bb_pct_pit'] / all_fip[team]['bb_pct'] if team in all_fip and all_fip[team]['bb_pct'] > 0 else 1.0

    # Combine with equal weights
    off_rating = (
        rolling_off_rating * OFFENSE_WEIGHTS['rolling'] +
        woba_rating * OFFENSE_WEIGHTS['woba'] +
        xwoba_rating * OFFENSE_WEIGHTS['xwoba'] +
        k_off_rating * OFFENSE_WEIGHTS['k_pct'] +
        bb_off_rating * OFFENSE_WEIGHTS['bb_pct']
    )

    pit_rating = (
        rolling_pit_rating * PITCHING_WEIGHTS['rolling'] +
        fip_rating * PITCHING_WEIGHTS['fip'] +
        xfip_rating * PITCHING_WEIGHTS['xfip'] +
        k_pit_rating * PITCHING_WEIGHTS['k_pct'] +
        bb_pit_rating * PITCHING_WEIGHTS['bb_pct']
    )

    # Apply caps
    off_rating = max(0.60, min(1.60, off_rating))
    pit_rating = max(0.60, min(1.60, pit_rating))

    return off_rating, pit_rating

def get_cached_stats(team_ids, game_date=None):
    cache_file = 'stats_cache.json'
    
    if os.path.exists(cache_file):
        modified_time = os.path.getmtime(cache_file)
        from datetime import datetime
        age_hours = (datetime.now().timestamp() - modified_time) / 3600
        if age_hours < 4:
            print("Loading stats from cache...")
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    print("Fetching fresh stats...")
    stats = {
        'all_woba': get_all_team_woba(team_ids),
        'all_fip': get_all_team_fip(team_ids),
        'team_xwoba': get_team_xwoba(team_ids),
        'rolling': get_all_rolling_averages(team_ids),
        'starters': get_todays_starters(game_date=game_date)
    }
    
    with open(cache_file, 'w') as f:
        json.dump(convert_to_serializable(stats), f)
    
    return stats
    
def convert_to_serializable(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def update_google_sheets(df):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        creds_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'google_credentials.json')
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        client = gspread.authorize(creds)
        
        sheet = client.open('Baseball Model Predictions').sheet1
        
        sheet.clear()
        
        headers = df.columns.tolist()
        sheet.append_row(headers)
        
        for _, row in df.iterrows():
            sheet.append_row([str(x) if x is not None else '' for x in row.tolist()])
        
        print("Google Sheets updated successfully")
    except Exception as e:
        print(f"Google Sheets update failed: {e}")

def _roi_for_bets(bets_df):
    profit = 0
    wagered = 0
    for _, row in bets_df.iterrows():
        if row['result'] not in ('WIN', 'LOSS'):
            continue
        try:
            line = float(row['book_line'])
        except (ValueError, TypeError):
            continue
        wagered += 100
        if row['result'] == 'WIN':
            profit += line if line > 0 else 100 * (100 / abs(line))
        else:
            profit -= 100
    return (profit / wagered * 100) if wagered > 0 else 0

def print_accuracy_report():
    log_file = 'predictions_log.csv'
    if not os.path.exists(log_file):
        return

    df = pd.read_csv(log_file, dtype=str)

    # Build one row per prediction using the latest available run
    rows = []
    for _, row in df.iterrows():
        bet_team = edge = book_line = None
        for run in [3, 2, 1]:
            t = row.get(f'run{run}_bet_team')
            if pd.notna(t) and t not in ('No Bet', 'nan', 'None'):
                bet_team = t
                edge = row.get(f'run{run}_edge')
                book_line = row.get(f'run{run}_book_line')
                break
        rows.append({
            'date': row['date'],
            'bet_type': row['bet_type'],
            'bet_team': bet_team,
            'edge': edge,
            'book_line': book_line,
            'result': row.get('result'),
        })

    analysis = pd.DataFrame(rows)
    bets = analysis[analysis['bet_team'].notna()]
    settled = bets[bets['result'].isin(['WIN', 'LOSS', 'PUSH'])]
    pending = bets[bets['result'] == 'PENDING']

    print("\n=== ACCURACY REPORT ===\n")

    if settled.empty:
        print(f"No settled bets yet. ({len(pending)} pending)")
        return

    w = (settled['result'] == 'WIN').sum()
    l = (settled['result'] == 'LOSS').sum()
    p = (settled['result'] == 'PUSH').sum()
    wr = w / (w + l) if (w + l) > 0 else 0
    print(f"Overall:    {w}W - {l}L - {p}P  |  Win Rate: {wr:.1%}  |  Pending: {len(pending)}")
    print()

    for bet_type, label in [('Moneyline', 'Moneyline '), ('Over/Under', 'Over/Under')]:
        sub = settled[settled['bet_type'] == bet_type]
        sw = (sub['result'] == 'WIN').sum()
        sl = (sub['result'] == 'LOSS').sum()
        sp = (sub['result'] == 'PUSH').sum()
        swr = sw / (sw + sl) if (sw + sl) > 0 else 0
        if bet_type == 'Moneyline':
            roi = _roi_for_bets(sub)
            print(f"{label}:  {sw}W - {sl}L - {sp}P  |  Win Rate: {swr:.1%}  |  ROI: {roi:+.1f}%")
        else:
            print(f"{label}: {sw}W - {sl}L - {sp}P  |  Win Rate: {swr:.1%}")

    # Edge tier breakdown (moneyline only, settled)
    ml = settled[settled['bet_type'] == 'Moneyline'].copy()
    if not ml.empty:
        def edge_tier(e):
            if pd.isna(e):
                return None
            try:
                pct = float(str(e).replace('%', '').replace('+', ''))
                if pct >= 10:
                    return '10%+'
                elif pct >= 5:
                    return '5-10%'
                else:
                    return '3-5%'
            except ValueError:
                return None

        ml['tier'] = ml['edge'].apply(edge_tier)
        tier_group = ml[ml['tier'].notna()].groupby('tier')
        print()
        print("Edge tiers (Moneyline):")
        for tier in ['3-5%', '5-10%', '10%+']:
            if tier in tier_group.groups:
                g = tier_group.get_group(tier)
                tw = (g['result'] == 'WIN').sum()
                tl = (g['result'] == 'LOSS').sum()
                twr = tw / (tw + tl) if (tw + tl) > 0 else 0
                print(f"  {tier:6s}: {tw}W - {tl}L  |  {twr:.1%}")

    print()

odds_data = get_cached_odds()

upcoming=get_upcoming_games(odds_data)

game_statuses = get_todays_game_statuses()

in_progress_games = []
completed_games = []

team_ids = get_all_team_ids()

from datetime import date
first_game_date = date.today().strftime('%Y-%m-%d')
cached_stats = get_cached_stats(team_ids, game_date=first_game_date)

all_woba = cached_stats['all_woba']
all_fip = cached_stats['all_fip']
team_xwoba = cached_stats['team_xwoba']
rolling = cached_stats['rolling']
todays_starters = cached_stats.get('starters', {})
lg_avgs = calculate_league_averages(all_woba, all_fip, team_xwoba)

for key, status in game_statuses.items():
    parts = key.split('_')
    away_fg = parts[0]
    home_fg = parts[1]
    
    away_name = next((k for k, v in NAME_TO_FG.items() if v == away_fg), away_fg)
    home_name = next((k for k, v in NAME_TO_FG.items() if v == home_fg), home_fg)
    
    if status['status'] == 'In Progress':
        in_progress_games.append(f"⚾ {away_name} @ {home_name} | {status['away_score']} - {status['home_score']}")
    elif status['status'] == 'Final':
        completed_games.append(f"✔️  {away_name} @ {home_name} | Final: {status['away_score']} - {status['home_score']}")

print(f"\nUpcoming games today: {len(upcoming)}")

print("\n=== TODAY'S EDGES ===\n")

edge_games = []
no_edge_games = []
skipped_games = []

# Game Loop
upcoming = sorted(upcoming, key=lambda x: x['commence_time'])
seen_games = set()
for game in upcoming:
    game_key = f"{game['home_team']}_{game['away_team']}"
    if game_key in seen_games:
        continue
    seen_games.add(game_key)

    home_name = game['home_team']
    away_name = game['away_team']

    if home_name not in NAME_TO_FG or away_name not in NAME_TO_FG:
        continue

    home_fg = NAME_TO_FG[home_name]
    away_fg = NAME_TO_FG[away_name]

    if home_fg not in rolling or away_fg not in rolling:
        continue

    if len(game['bookmakers']) == 0:
        continue

    best_home_line = None
    best_away_line = None
    best_home_book = None
    best_away_book = None

    for bookmaker in game['bookmakers']:
        for market in bookmaker['markets']:
            if market['key'] == 'h2h':
                for outcome in market['outcomes']:
                    if outcome['name'] == home_name:
                        if best_home_line is None or outcome['price'] > best_home_line:
                            best_home_line = outcome['price']
                            best_home_book = bookmaker['title']
                    elif outcome['name'] == away_name:
                        if best_away_line is None or outcome['price'] > best_away_line:
                            best_away_line = outcome['price']
                            best_away_book = bookmaker['title']

    if best_home_line is None or best_away_line is None:
        continue

    home_market_line = best_home_line
    away_market_line = best_away_line

    home_market_prob = american_to_prob(home_market_line)
    away_market_prob = american_to_prob(away_market_line)

    game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).astimezone(MST)
    time_str = game_time.strftime('%a %b %d - %I:%M %p MST')

    home_starter = todays_starters.get(home_fg)
    away_starter = todays_starters.get(away_fg)

    if home_starter is None or away_starter is None:
        missing = []
        if home_starter is None:
            missing.append(home_name)
        if away_starter is None:
            missing.append(away_name)
        skipped_games.append(f"⚠️  {away_name} @ {home_name} - {time_str} | Starter not yet announced for: {', '.join(missing)}")
        continue

    low_sample = []
    if home_starter and home_starter['innings'] < MIN_INNINGS:
        low_sample.append(f"{home_name} starter {home_starter['name']} ({home_starter['innings']} IP)")
    if away_starter and away_starter['innings'] < MIN_INNINGS:
        low_sample.append(f"{away_name} starter {away_starter['name']} ({away_starter['innings']} IP)")

    result = calculate_matchup(home_fg, away_fg, 2026, rolling=rolling, starters=todays_starters)
    if result is None:
        continue

    home_win_pct, away_win_pct, avg_total, home_lambda, away_lambda = result
    total_data = get_total_line(game)

    home_edge = home_win_pct - home_market_prob
    away_edge = away_win_pct - away_market_prob

    warning = f"\n   ⚠️  Low sample: {', '.join(low_sample)}" if low_sample else ""

    moneyline_edge = False
    ou_edge = False
    game_text = f"{away_name} @ {home_name} - {time_str}"
    
    if home_edge > 0.03:
        moneyline_text = f"   💰 Moneyline: Bet {home_name} | Model: {home_win_pct:.1%} | {best_home_book}: {home_market_line} | Edge: +{home_edge:.1%}"
        moneyline_edge = True
        bet_fg = home_fg
        bet_pct = home_win_pct
        bet_line = home_market_line
        bet_edge = home_edge
    elif away_edge > 0.03:
        moneyline_text = f"   💰 Moneyline: Bet {away_name} | Model: {away_win_pct:.1%} | {best_away_book}: {away_market_line} | Edge: +{away_edge:.1%}"
        moneyline_edge = True
        bet_fg = away_fg
        bet_pct = away_win_pct
        bet_line = away_market_line
        bet_edge = away_edge
    else:
        moneyline_text = f"   ➖ Moneyline: No edge found"

    if total_data:
        book_total = total_data['total']
        ou_diff = avg_total - book_total
        if ou_diff > 1.5:
            ou_text = f"   📊 Over/Under: OVER | Model: {avg_total:.1f} | {total_data['book']}: {book_total} (Over {total_data['over_price']}) | Edge: +{ou_diff:.1f} runs"
            ou_edge = True

        elif ou_diff < -1.5:
            ou_text = f"   📊 Over/Under: UNDER | Model: {avg_total:.1f} | {total_data['book']}: {book_total} (Under {total_data['under_price']}) | Edge: {ou_diff:.1f} runs"
            ou_edge = True

        else:
            ou_text = f"   ➖ Over/Under: No edge | Model: {avg_total:.1f} | Book: {book_total}"
    else:
        ou_text = f"   ➖ Over/Under: No line available"

    proj_text = f"   {home_name} est: {home_lambda:.1f} | {away_name} est: {away_lambda:.1f} | Model total: {avg_total:.1f}"

    # Log moneyline for all games
    if moneyline_edge:
        ml_bet_team = bet_fg
        ml_model_pct = f"{bet_pct:.1%}"
        ml_book_line = str(bet_line)
        ml_edge = f"+{bet_edge:.1%}"
    else:
        ml_bet_team = 'No Bet'
        ml_model_pct = f"{max(home_win_pct, away_win_pct):.1%}"
        ml_book_line = str(home_market_line if home_win_pct > away_win_pct else away_market_line)
        ml_edge = 'No Edge'

    todays_predictions.append({
        'date': first_game_date,
        'run_time': run_time,
        'home_fg': home_fg,
        'away_fg': away_fg,
        'bet_type': 'Moneyline',
        'bet_team': ml_bet_team,
        'model_pct': ml_model_pct,
        'book_line': ml_book_line,
        'edge': ml_edge
    })

    # Log O/U for all games
    if total_data:
        if ou_edge:
            ou_bet_team = 'Over' if ou_diff > 0 else 'Under'
            ou_edge_str = f"{ou_diff:+.1f} runs"
        else:
            ou_bet_team = 'No Bet'
            ou_edge_str = 'No Edge'
        
        todays_predictions.append({
            'date': first_game_date,
            'run_time': run_time,
            'home_fg': home_fg,
            'away_fg': away_fg,
            'bet_type': 'Over/Under',
            'bet_team': ou_bet_team,
            'model_pct': f"{avg_total:.1f}",
            'book_line': str(book_total),
            'edge': ou_edge_str
        })
    else:
        todays_predictions.append({
            'date': first_game_date,
            'run_time': run_time,
            'home_fg': home_fg,
            'away_fg': away_fg,
            'bet_type': 'Over/Under',
            'bet_team': 'No Bet',
            'model_pct': f"{avg_total:.1f}",
            'book_line': 'N/A',
            'edge': 'No Line'
        })

    if moneyline_edge or ou_edge:
        edge_games.append({
            'text': f"✅ {game_text}\n{proj_text}\n{moneyline_text}\n{ou_text}{warning}"
        })
    else:
        no_edge_games.append(f"❌ {game_text}\n{proj_text}\n{moneyline_text}\n{ou_text}{warning}")

# Print edges
if edge_games:
    for game in edge_games:
        print(game['text'])
        print()
else:
    print("No edges found today.\n")

# Print no edge games
if no_edge_games:
    print("=== NO EDGE FOUND ===\n")
    for game in no_edge_games:
        print(game)
        print()

# Print skipped games
if skipped_games:
    print("=== AWAITING STARTERS ===\n")
    for game in skipped_games:
        print(game)
        print()

if in_progress_games:
    print("\n=== IN PROGRESS ===\n")
    for game in in_progress_games:
        print(game)
        print()

if completed_games:
    print("\n=== COMPLETED ===\n")
    for game in completed_games:
        print(game)
        print()

todays_starters = get_todays_starters()

log_predictions(todays_predictions, yesterdays_results)
print_accuracy_report()
