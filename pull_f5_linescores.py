import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import csv
import json
import calendar
import requests

SEASONS = [2021, 2022, 2023, 2024, 2025]
MONTHS = range(3, 11)  # March - October covers all regular season games

OUTPUT_PATH = 'historical_data/f5_linescores.csv'
SKIPPED_PATH = 'historical_data/f5_skipped.csv'
PROGRESS_PATH = 'historical_data/f5_progress.json'

# FanGraphs abbreviation -> MLB team ID (TEAM_MAP convention from backtest.py)
TEAM_MAP = {
    'ATH': 133, 'PIT': 134, 'SDP': 135, 'SEA': 136, 'SFG': 137,
    'STL': 138, 'TBR': 139, 'TEX': 140, 'TOR': 141, 'MIN': 142,
    'PHI': 143, 'ATL': 144, 'CHW': 145, 'MIA': 146, 'NYY': 147,
    'MIL': 158, 'LAA': 108, 'ARI': 109, 'BAL': 110, 'BOS': 111,
    'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KCR': 118, 'LAD': 119, 'WSN': 120, 'NYM': 121
}
MLB_ID_TO_FG = {v: k for k, v in TEAM_MAP.items()}

OUTPUT_COLUMNS = [
    'date', 'gamePk', 'season', 'home_team', 'away_team',
    'home_runs_f5', 'away_runs_f5', 'total_f5',
    'home_runs_final', 'away_runs_final'
]
SKIPPED_COLUMNS = ['date', 'gamePk', 'season', 'home_team', 'away_team', 'innings_played', 'reason']


def get_schedule_chunk(season, month):
    last_day = calendar.monthrange(season, month)[1]
    start_date = f'{season}-{month:02d}-01'
    end_date = f'{season}-{month:02d}-{last_day:02d}'
    url = (f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}'
           f'&gameType=R&hydrate=linescore&startDate={start_date}&endDate={end_date}')
    resp = requests.get(url, timeout=30).json()
    games = []
    for date_entry in resp.get('dates', []):
        games.extend(date_entry.get('games', []))
    return games


def get_f5_runs(linescore):
    innings = linescore.get('innings', [])
    if len(innings) < 5:
        return None
    home_f5 = 0
    away_f5 = 0
    for inning in innings[:5]:
        away_runs = inning.get('away', {}).get('runs')
        home_runs = inning.get('home', {}).get('runs')
        if away_runs is None or home_runs is None:
            return None
        away_f5 += away_runs
        home_f5 += home_runs
    return home_f5, away_f5


def process_game(game, season):
    status = game.get('status', {})
    if status.get('abstractGameState') != 'Final':
        return None, None

    teams = game.get('teams', {})
    home_id = teams.get('home', {}).get('team', {}).get('id')
    away_id = teams.get('away', {}).get('team', {}).get('id')
    home_team = MLB_ID_TO_FG.get(home_id)
    away_team = MLB_ID_TO_FG.get(away_id)
    if home_team is None or away_team is None:
        return None, None

    date = game.get('officialDate')
    game_pk = game.get('gamePk')
    linescore = game.get('linescore', {})

    ls_teams = linescore.get('teams', {})
    home_runs_final = ls_teams.get('home', {}).get('runs')
    away_runs_final = ls_teams.get('away', {}).get('runs')
    if home_runs_final is None:
        home_runs_final = teams.get('home', {}).get('score')
    if away_runs_final is None:
        away_runs_final = teams.get('away', {}).get('score')

    f5 = get_f5_runs(linescore)
    if f5 is None:
        skipped_row = {
            'date': date, 'gamePk': game_pk, 'season': season,
            'home_team': home_team, 'away_team': away_team,
            'innings_played': len(linescore.get('innings', [])),
            'reason': 'fewer than 5 completed innings',
        }
        return None, skipped_row

    home_f5, away_f5 = f5
    row = {
        'date': date, 'gamePk': game_pk, 'season': season,
        'home_team': home_team, 'away_team': away_team,
        'home_runs_f5': home_f5, 'away_runs_f5': away_f5,
        'total_f5': home_f5 + away_f5,
        'home_runs_final': home_runs_final, 'away_runs_final': away_runs_final,
    }
    return row, None


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


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


def print_summary():
    if not os.path.exists(OUTPUT_PATH):
        print("No output file found.")
        return

    counts = {}
    totals = {}
    with open(OUTPUT_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            season = row['season']
            counts[season] = counts.get(season, 0) + 1
            totals[season] = totals.get(season, 0) + float(row['total_f5'])

    print("\n=== Summary ===")
    for season in sorted(counts, key=int):
        n = counts[season]
        avg = totals[season] / n if n else 0
        print(f"{season}: {n} games, avg total_f5/game = {avg:.2f}")


def main():
    progress = load_progress()
    completed_seasons = set(progress.get('completed_seasons', []))

    for season in SEASONS:
        if season in completed_seasons:
            print(f"{season}: already completed, skipping.")
            continue

        print(f"Processing season {season}...")
        season_rows = []
        season_skipped = []
        for month in MONTHS:
            games = get_schedule_chunk(season, month)
            for game in games:
                row, skipped = process_game(game, season)
                if row:
                    season_rows.append(row)
                if skipped:
                    season_skipped.append(skipped)
            print(f"  {season}-{month:02d}: {len(games)} games scanned")

        append_csv_rows(season_rows, OUTPUT_PATH, OUTPUT_COLUMNS)
        append_csv_rows(season_skipped, SKIPPED_PATH, SKIPPED_COLUMNS)

        completed_seasons.add(season)
        progress['completed_seasons'] = sorted(completed_seasons)
        save_progress(progress)

        print(f"  {season}: {len(season_rows)} games, {len(season_skipped)} skipped (rain-shortened)")

    print_summary()


if __name__ == '__main__':
    main()
