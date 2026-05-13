import pandas as pd
import math
import requests
import statsapi
import json
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=r'C:\Users\super\baseball-model\.env')
from datetime import date

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

from datetime import timezone, timedelta, datetime

API_KEY = os.getenv('API_KEY')
print(f"API Key loaded: {API_KEY is not None}")

MST = timezone(timedelta(hours=-7))

STARTER_INNINGS = 5
TOTAL_INNINGS = 9
STARTER_WEIGHT = STARTER_INNINGS / TOTAL_INNINGS
BULLPEN_WEIGHT = 1 - STARTER_WEIGHT
MIN_INNINGS = 15
MAX_RA9 = 7

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
        if age_hours < 6:
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
    upcoming = []
    for game in odds_data:
        game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        if game_time > now:
            upcoming.append(game)
    return upcoming

def get_mlb_odds():
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={API_KEY}&regions=us&markets=h2h&oddsFormat=american'
    response = requests.get(url)
    data = response.json()
    return data

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
        home = calculate_rolling_lambda(home_team, rolling)
        away = calculate_rolling_lambda(away_team, rolling)

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

        lg_avg = league_avg[league_avg['year'] == 2026]['lg_avg_runs'].values[0]
        home_pitch_rating = lg_avg / home_blended_pitch
        away_pitch_rating = lg_avg / away_blended_pitch

        home_pitch_rating = max(0.60, min(1.60, home_pitch_rating))
        away_pitch_rating = max(0.60, min(1.60, away_pitch_rating))

        home_lambda = (home['off_rating'] / away_pitch_rating) * lg_avg
        away_lambda = (away['off_rating'] / home_pitch_rating) * lg_avg

    else:
        home_off = model_data[(model_data['team'] == home_team) & (model_data['year'] == year)]['lambda'].values[0]
        away_off = model_data[(model_data['team'] == away_team) & (model_data['year'] == year)]['lambda'].values[0]
        home_pitch = model_data[(model_data['team'] == home_team) & (model_data['year'] == year)]['pitch_rating'].values[0]
        away_pitch = model_data[(model_data['team'] == away_team) & (model_data['year'] == year)]['pitch_rating'].values[0]
        home_lambda = home_off / away_pitch
        away_lambda = away_off / home_pitch

    home_wins = 0
    away_wins = 0
    tie = 0

    for i in range(20):
        for j in range(20):
            prob = (math.exp(-home_lambda) * home_lambda**i / math.factorial(i)) * (math.exp(-away_lambda) * away_lambda**j / math.factorial(j))
            if i > j:
                home_wins += prob
            elif j > i:
                away_wins += prob
            else:
                tie += prob

    home_win_pct = home_wins + tie / 2
    away_win_pct = away_wins + tie / 2

    if home_win_pct > 0.5:
        home_line = f"-{abs(round((home_win_pct / (1 - home_win_pct)) * 100))}"
        away_line = f"+{abs(round(((1 - home_win_pct) / home_win_pct) * 100))}"
    else:
        home_line = f"+{abs(round(((1 - home_win_pct) / home_win_pct) * 100))}"
        away_line = f"-{abs(round((away_win_pct / (1 - away_win_pct)) * 100))}"

    return home_win_pct, away_win_pct

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

yankees_pitching = get_pitching_game_logs(147)
avg_pitch_7 = get_rolling_average(yankees_pitching,days=7,column='runs_allowed')
avg_pitch_15 = get_rolling_average(yankees_pitching,days=15,column='runs_allowed')

def get_all_team_ids():
    teams = statsapi.get('teams',{'sportId':1,'season':2026})
    teams_ids={}
    for team in teams['teams']:
        abbrev=team.get('abbreviation')
        if abbrev in TEAM_MAP:
            fg_abbrev=TEAM_MAP[abbrev]
            teams_ids[fg_abbrev]=team['id']
    return teams_ids

team_ids = get_all_team_ids()

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

#New Def Log Predictions - Beginning
def log_predictions(predictions, results):
    import csv
    from datetime import datetime
    log_file = 'predictions_log.csv'
    file_exists = os.path.exists(log_file)
    
    # Check if this run has already been logged
    if file_exists and predictions:
        existing = pd.read_csv(log_file)
        run_time_check = predictions[0]['run_time']
        date_check = predictions[0]['date']
        if not existing[(existing['run_time'] == run_time_check) & 
                        (existing['date'] == date_check)].empty:
            print("Predictions already logged for this run - skipping")
            return
#New Def Log Predictions - End

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

odds_data = get_cached_odds()

upcoming=get_upcoming_games(odds_data)

rolling = get_all_rolling_averages(team_ids)
if upcoming:
    first_game_date = upcoming[0]['commence_time'][:10]
    todays_starters = get_todays_starters(game_date=first_game_date)
else:
    todays_starters = {}

print(f"\nUpcoming games today: {len(upcoming)}")

print("\n=== TODAY'S EDGES ===\n")

edge_games = []
no_edge_games = []
skipped_games = []

# Game Loop
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

    bookmaker = game['bookmakers'][0]
    outcomes = bookmaker['markets'][0]['outcomes']

    market_lines = {}
    for outcome in outcomes:
        market_lines[outcome['name']] = outcome['price']

    if home_name not in market_lines or away_name not in market_lines:
        continue

    home_market_line = market_lines[home_name]
    away_market_line = market_lines[away_name]

    home_market_prob = american_to_prob(home_market_line)
    away_market_prob = american_to_prob(away_market_line)

    game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).astimezone(MST)
    time_str = game_time.strftime('%I:%M %p MST')

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

    home_win_pct, away_win_pct = result
    home_edge = home_win_pct - home_market_prob
    away_edge = away_win_pct - away_market_prob

    warning = f"\n   ⚠️  Low sample: {', '.join(low_sample)}" if low_sample else ""
    if home_edge > 0.03:
        edge_games.append({
            'text': f"✅ {away_name} @ {home_name} - {time_str}\n   Bet: {home_name} | Model: {home_win_pct:.1%} | Book: {home_market_line} | Edge: +{home_edge:.1%}{warning}"
        })
        todays_predictions.append({
            'date': first_game_date,
            'run_time': run_time,
            'home_fg': home_fg,
            'away_fg': away_fg,
            'bet_fg': home_fg,
            'model_pct': f"{home_win_pct:.1%}",
            'book_line': home_market_line,
            'edge': f"+{home_edge:.1%}"
         })   
    elif away_edge > 0.03:
        edge_games.append({
            'text': f"✅ {away_name} @ {home_name} - {time_str}\n   Bet: {away_name} | Model: {away_win_pct:.1%} | Book: {away_market_line} | Edge: +{away_edge:.1%}{warning}"
        })
        todays_predictions.append({
            'date': first_game_date,
            'run_time': run_time,
            'home_fg': home_fg,
            'away_fg': away_fg,
            'bet_fg': away_fg,
            'model_pct': f"{away_win_pct:.1%}",
            'book_line': away_market_line,
            'edge': f"+{away_edge:.1%}"
        })
    else:
        no_edge_games.append(f"❌ {away_name} @ {home_name} - {time_str}\n   {home_name}: {home_win_pct:.1%} (Book: {home_market_line}) | {away_name}: {away_win_pct:.1%} (Book: {away_market_line}){warning}")

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

todays_starters = get_todays_starters()

log_predictions(todays_predictions, yesterdays_results)
if os.path.exists('predictions_log.csv'):
    log = pd.read_csv('predictions_log.csv')
    print(log)