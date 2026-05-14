import statsapi

# Get team stats for a specific team
team_stats = statsapi.get('team_stats', {
    'teamId': 147,
    'stats': 'season',
    'group': 'hitting',
    'season': 2026
})

print(team_stats['stats'][0]['splits'][0]['stat'].keys())
