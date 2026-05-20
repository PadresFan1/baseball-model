import json
import os
import statsapi

TEAM_MAP = {
    'ATH': 133, 'PIT': 134, 'SDP': 135, 'SEA': 136, 'SFG': 137,
    'STL': 138, 'TBR': 139, 'TEX': 140, 'TOR': 141, 'MIN': 142,
    'PHI': 143, 'ATL': 144, 'CHW': 145, 'MIA': 146, 'NYY': 147,
    'MIL': 158, 'LAA': 108, 'ARI': 109, 'BAL': 110, 'BOS': 111,
    'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KCR': 118, 'LAD': 119, 'WSN': 120, 'NYM': 121
}

SEASONS = [2021, 2022, 2023, 2024, 2025]
OUTPUT = 'historical_data/player_team_map.json'

def build_player_team_map():
    if os.path.exists(OUTPUT):
        print(f"Map already exists at {OUTPUT} — delete it to rebuild.")
        with open(OUTPUT) as f:
            return json.load(f)

    player_map = {}  # {season: {player_id: fg_team_abbrev}}

    for season in SEASONS:
        print(f"\nPulling rosters for {season}...")
        season_map = {}
        for fg_abbrev, team_id in TEAM_MAP.items():
            try:
                roster = statsapi.get('team_roster', {
                    'teamId': team_id,
                    'rosterType': 'fullSeason',
                    'season': season
                })
                for player in roster['roster']:
                    pid = str(player['person']['id'])
                    season_map[pid] = fg_abbrev
            except Exception as e:
                print(f"  Error pulling {fg_abbrev} {season}: {e}")
                continue
        player_map[str(season)] = season_map
        print(f"  {season}: mapped {len(season_map)} players across {len(TEAM_MAP)} teams")

    with open(OUTPUT, 'w') as f:
        json.dump(player_map, f)
    print(f"\nSaved to {OUTPUT}")
    return player_map

if __name__ == '__main__':
    build_player_team_map()
