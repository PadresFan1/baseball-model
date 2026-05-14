import requests
import pandas as pd
import statsapi
from io import StringIO

TEAM_MAP = {
    'ATH': 133, 'PIT': 134, 'SDP': 135, 'SEA': 136, 'SFG': 137,
    'STL': 138, 'TBR': 139, 'TEX': 140, 'TOR': 141, 'MIN': 142,
    'PHI': 143, 'ATL': 144, 'CHW': 145, 'MIA': 146, 'NYY': 147,
    'MIL': 158, 'LAA': 108, 'ARI': 109, 'BAL': 110, 'BOS': 111,
    'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KCR': 118, 'LAD': 119, 'WSN': 120, 'NYM': 121
}

def get_player_team_map(team_ids):
    player_team = {}
    for fg_abbrev, team_id in team_ids.items():
        try:
            roster_data = statsapi.get('team_roster', {
                'teamId': team_id,
                'rosterType': 'active'
            })
            for player in roster_data['roster']:
                player_team[player['person']['id']] = fg_abbrev
        except Exception as e:
            continue
    return player_team

# Pull xwOBA data from Baseball Savant
url = "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year=2026&position=&team=&min=10&csv=true"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0'}
response = requests.get(url, headers=headers)
xwoba_df = pd.read_csv(StringIO(response.text))

player_team_map = get_player_team_map(TEAM_MAP)

# Match players to teams and aggregate xwOBA
xwoba_df['team'] = xwoba_df['player_id'].map(player_team_map)

# Drop players not on active rosters
xwoba_df = xwoba_df.dropna(subset=['team'])

print(f"Players matched to teams: {len(xwoba_df)}")

# Aggregate to team level weighted by PA
team_xwoba = xwoba_df.groupby('team').apply(
    lambda x: pd.Series({
        'xwoba': (x['est_woba'] * x['pa']).sum() / x['pa'].sum(),
        'woba': (x['woba'] * x['pa']).sum() / x['pa'].sum(),
        'total_pa': x['pa'].sum()
    })
).reset_index()

print("\nTeam xwOBA rankings:")
team_xwoba_sorted = team_xwoba.sort_values('xwoba', ascending=False)
for _, row in team_xwoba_sorted.iterrows():
    print(f"{row['team']}: xwOBA={row['xwoba']:.3f} | wOBA={row['woba']:.3f} | PA={row['total_pa']:.0f}")