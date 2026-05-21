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

STATCAST_TEAM_MAP = {
    'AZ': 'ARI', 'CWS': 'CHW', 'KC': 'KCR',
    'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR', 'WSH': 'WSN'
}

import random
from itertools import product

def load_woba_fip_constants(constants_path, season):
    df = pd.read_csv(constants_path)
    row = df[df['Season'] == season].iloc[0]
    woba_weights = {
        'wBB':  row['wBB'],
        'wHBP': row['wHBP'],
        'w1B':  row['w1B'],
        'w2B':  row['w2B'],
        'w3B':  row['w3B'],
        'wHR':  row['wHR'],
    }
    fip_constant = row['cFIP']
    return woba_weights, fip_constant

def random_search(n_trials=100, seed=None):
    import time
    import ctypes

    # Prevent Windows from sleeping while the backtest runs
    ES_CONTINUOUS      = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    start_time = time.time()
    checkpoint_file = 'historical_data/random_search_checkpoint.json'

    # Auto-generate seed if not provided so each run explores different params
    if seed is None:
        seed = random.randint(1, 99999)
    print(f"Seed: {seed}  (pass seed={seed} to reproduce this run)")
    random.seed(seed)

    PARAM_ROUND = 5  # increment this whenever param_space changes
    # Round 1: base params, no statcast
    # Round 2: statcast added (xwoba_bat, babip_bat, barrel_bat/pit)
    # Round 3: bullpen added (bullpen_qual, bullpen_fat)
    # Round 4: bullpen removed, favorites_only locked False — 53% profitable
    # Round 5: locked edge/cap params, dropped barrel_bat, rebalanced bb_pit/fip/barrel_pit

    param_space = {
        'ml_edge_min':    [0.07],             # locked — clear winner across all rounds
        'ml_edge_max':    [0.08],             # locked — +4.98% avg in top 25%
        'favorites_only': [False],            # locked — True was small-sample artifact
        'rolling_weight_7': [0.50, 0.60, 0.70, 0.80],
        'rolling_weight_15': None,
        'rating_cap_low':  [0.65],            # locked — 0.70 not meaningfully different
        'rating_cap_high': [1.50],            # locked — appeared in 8/10 top r4 trials
        # Offense weights (barrel_bat dropped — -0.298 correlation in r4)
        'w_rolling_off': [0.05, 0.08, 0.10],
        'w_woba':        [0.20, 0.30, 0.40],
        'w_xwoba_bat':   [0.10, 0.20, 0.30],
        'w_babip_bat':   [0.15, 0.25, 0.35], # raised — +0.115 in r4
        # Pitching weights
        'w_rolling_pit': [0.05, 0.08, 0.10],
        'w_fip':         [0.10, 0.20, 0.30], # pulled back — flipped negative in r4
        'w_xfip':        [0.10, 0.20, 0.30],
        'w_k_pit':       [0.10, 0.20, 0.30],
        'w_bb_pit':      [0.15, 0.25, 0.35], # pulled back — over-corrected in r4
        'w_xwoba_pit':   [0.10, 0.20, 0.30],
        'w_babip_pit':   [0.05, 0.08, 0.12], # reduced — consistently negative
        'w_barrel_pit':  [0.15, 0.25, 0.35], # raised — +0.136 in r4
    }

    # Resume from checkpoint if one exists
    results = []
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        if checkpoint.get('seed') == seed and checkpoint.get('n_trials') == n_trials:
            results = checkpoint['results']
            start_trial = checkpoint['completed_trials']
            # Fast-forward the RNG to the correct position
            temp_seed = seed
            random.seed(temp_seed)
            for _ in range(start_trial):
                for key in ['ml_edge_min', 'ml_edge_max', 'favorites_only', 'rolling_weight_7',
                            'rating_cap_low', 'rating_cap_high', 'w_rolling_off', 'w_woba',
                            'w_xwoba_bat', 'w_babip_bat', 'w_rolling_pit',
                            'w_fip', 'w_xfip', 'w_k_pit', 'w_bb_pit',
                            'w_xwoba_pit', 'w_babip_pit', 'w_barrel_pit']:
                    random.choice(param_space[key] or [0.0])
            print(f"Resuming from trial {start_trial}/{n_trials} ({len(results)} results so far)")

    for trial in range(start_trial, n_trials):
        # Sample random parameters
        params = {
            'ml_edge_min':    random.choice(param_space['ml_edge_min']),
            'ml_edge_max':    random.choice(param_space['ml_edge_max']),
            'favorites_only': random.choice(param_space['favorites_only']),
            'rolling_weight_7': random.choice(param_space['rolling_weight_7']),
            'rating_cap_low':  random.choice(param_space['rating_cap_low']),
            'rating_cap_high': random.choice(param_space['rating_cap_high']),
            'w_rolling_off': random.choice(param_space['w_rolling_off']),
            'w_woba':        random.choice(param_space['w_woba']),
            'w_xwoba_bat':   random.choice(param_space['w_xwoba_bat']),
            'w_babip_bat':   random.choice(param_space['w_babip_bat']),
            'w_rolling_pit': random.choice(param_space['w_rolling_pit']),
            'w_fip':         random.choice(param_space['w_fip']),
            'w_xfip':        random.choice(param_space['w_xfip']),
            'w_k_pit':       random.choice(param_space['w_k_pit']),
            'w_bb_pit':      random.choice(param_space['w_bb_pit']),
            'w_xwoba_pit':   random.choice(param_space['w_xwoba_pit']),
            'w_babip_pit':   random.choice(param_space['w_babip_pit']),
            'w_barrel_pit':  random.choice(param_space['w_barrel_pit']),
        }
        params['rolling_weight_15'] = 1 - params['rolling_weight_7']
        if params['ml_edge_min'] >= params['ml_edge_max']:
            continue

        # Normalize offense weights
        off_keys = ['w_rolling_off', 'w_woba', 'w_xwoba_bat', 'w_babip_bat']
        off_total = sum(params[k] for k in off_keys)
        for k in off_keys:
            params[k] /= off_total

        # Normalize pitching weights
        pit_keys = ['w_rolling_pit', 'w_fip', 'w_xfip', 'w_k_pit', 'w_bb_pit', 'w_xwoba_pit', 'w_babip_pit', 'w_barrel_pit']
        pit_total = sum(params[k] for k in pit_keys)
        for k in pit_keys:
            params[k] /= pit_total

        # Run backtest with these params
        trial_results = run_backtest_with_params(params, seasons=[2021, 2022, 2023, 2024])

        if trial_results is not None:
            results.append({**params, **trial_results})

        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            done_this_run = trial + 1 - start_trial
            per_trial = elapsed / done_this_run
            remaining = per_trial * (n_trials - trial - 1)
            print(f"Completed {trial + 1}/{n_trials} trials | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")

            # Save checkpoint every 10 trials
            with open(checkpoint_file, 'w') as f:
                json.dump({'seed': seed, 'n_trials': n_trials, 'completed_trials': trial + 1, 'results': results}, f)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)

    # Save with metadata in filename so multiple runs are preserved
    from datetime import datetime as _dt
    run_date = _dt.now().strftime('%Y-%m-%d')
    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    results_df['run_date'] = run_date
    results_df['seed'] = seed
    results_df['n_trials'] = n_trials
    results_df['run_time_mins'] = round(total_time / 60, 1)
    results_df['param_round'] = PARAM_ROUND

    run_file = f'historical_data/search_{run_date}_seed{seed}_{n_trials}trials_r{PARAM_ROUND}.csv'
    results_df.to_csv(run_file, index=False)
    print(f"\nResults saved to {run_file}")

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # Re-enable sleep
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    print(f"\nFinished {n_trials} trials in {mins}m {secs}s")
    print(f"\nTop 10 parameter combinations:")
    print(results_df[['roi', 'win_rate', 'n_bets', 'ml_edge_min', 'ml_edge_max',
                       'favorites_only', 'rolling_weight_7', 'rating_cap_low',
                       'rating_cap_high']].head(10).to_string())

    return results_df


def evaluate_runs():
    import glob
    run_files = sorted(glob.glob('historical_data/search_*.csv'))

    # Also pick up the legacy fixed-name file from before per-run naming was added
    legacy = 'historical_data/random_search_results.csv'
    if os.path.exists(legacy) and legacy not in run_files:
        run_files = [legacy] + run_files

    if not run_files:
        print("No completed runs found in historical_data/.")
        return

    all_dfs = []
    for f in run_files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"EVALUATION: {len(run_files)} run(s), {len(combined)} total trials")
    print(f"{'='*60}\n")

    # Per-round summary
    if 'param_round' in combined.columns:
        print("Round summary:")
        for rnd, grp in combined.groupby('param_round'):
            pct_pos = (grp['roi'] > 0).mean()
            print(f"  Round {int(rnd)}: {len(grp):>5} trials | top ROI: {grp['roi'].max():+.1f}% | median: {grp['roi'].median():+.1f}% | % profitable: {pct_pos:.1%}")
        print()

    # Per-run summary
    print("Run summary:")
    for f in run_files:
        df = pd.read_csv(f)
        meta = os.path.basename(f)
        rnd = f' r{int(df["param_round"].iloc[0])}' if 'param_round' in df.columns else ''
        top_roi = df['roi'].max()
        med_roi = df['roi'].median()
        print(f"  {meta}{rnd}  |  top ROI: {top_roi:+.1f}%  |  median ROI: {med_roi:+.1f}%  |  {len(df)} trials")

    # Top 10 across all runs
    top_cols = ['roi', 'win_rate', 'n_bets', 'ml_edge_min', 'ml_edge_max',
                'favorites_only', 'rolling_weight_7', 'rating_cap_low', 'rating_cap_high', 'param_round', 'run_date']
    available = [c for c in top_cols if c in combined.columns]
    print(f"\nTop 10 results (all sample sizes):")
    print(combined.nlargest(10, 'roi')[available].to_string(index=False))

    # Minimum sample size filter — more reliable signal
    min_bets = 300
    reliable = combined[combined['n_bets'] >= min_bets]
    if not reliable.empty:
        pct = len(reliable) / len(combined)
        print(f"\nTop 10 results (n_bets >= {min_bets}, {len(reliable)} trials / {pct:.0%} of total):")
        print(reliable.nlargest(10, 'roi')[available].to_string(index=False))

    # Parameter analysis — compare top 25% vs bottom 75%
    threshold = combined['roi'].quantile(0.75)
    top_q = combined[combined['roi'] >= threshold]

    discrete_params = ['ml_edge_min', 'ml_edge_max', 'favorites_only',
                       'rating_cap_low', 'rating_cap_high']

    print(f"\nParameter breakdown (top 25% of trials, ROI >= {threshold:+.1f}%):")
    for param in discrete_params:
        if param not in combined.columns:
            continue
        grouped = top_q.groupby(param)['roi'].agg(count='count', avg_roi='mean').sort_values('avg_roi', ascending=False)
        print(f"\n  {param}:")
        for val, row in grouped.iterrows():
            print(f"    {val!s:<10}  count: {int(row['count']):>4}  avg ROI: {row['avg_roi']:+.1f}%")

    # Continuous weight params — show correlation with ROI
    weight_params = ['rolling_weight_7',
                     'w_rolling_off', 'w_woba', 'w_xwoba_bat', 'w_babip_bat',
                     'w_rolling_pit', 'w_fip', 'w_xfip', 'w_k_pit', 'w_bb_pit',
                     'w_xwoba_pit', 'w_babip_pit', 'w_barrel_pit']
    available_weights = [p for p in weight_params if p in combined.columns]
    if available_weights:
        corrs = combined[available_weights + ['roi']].corr()['roi'].drop('roi').sort_values(ascending=False)
        print(f"\nWeight correlations with ROI (all trials):")
        for param, corr in corrs.items():
            bar = '+' * int(abs(corr) * 20) if corr > 0 else '-' * int(abs(corr) * 20)
            print(f"  {param:<20} {corr:+.3f}  {bar}")

    # Save combined results sorted by ROI
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime('%Y-%m-%d_%H-%M')

    combined_sorted = combined.sort_values('roi', ascending=False)
    combined_file = f'historical_data/evaluation_{timestamp}_{len(run_files)}runs.csv'
    combined_sorted.to_csv(combined_file, index=False)
    print(f"\nAll results saved to: {combined_file}")

def pull_historical_game_logs(team_ids, seasons=[2021, 2022, 2023, 2024]):
    log_path = 'historical_data/historical_game_logs.json'
    if os.path.exists(log_path):
        print("Loading historical game logs from file...")
        with open(log_path) as f:
            return json.load(f)
    
    # Only hits the API if the file doesn't exist
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

    with open(log_path, 'w') as f:
        json.dump(all_logs, f)
    print("Saved historical game logs")
    return all_logs

# Run this once to pull fresh data, then comment out again
historical_logs = pull_historical_game_logs(TEAM_MAP)

# Load historical game logs
with open('historical_data/historical_game_logs.json', 'r') as f:
    game_logs = json.load(f)

def load_statcast_data(seasons=[2021, 2022, 2023, 2024]):
    cache_file = 'historical_data/statcast_cache.json'
    if os.path.exists(cache_file):
        print("Loading statcast data from cache...")
        with open(cache_file) as f:
            return json.load(f)

    print("Loading statcast CSV files...")
    all_data = {}
    for season in seasons:
        season_data = {'batting': {}, 'pitching': {}}
        for side, fname in [('batting', f'Statcast/batters_{season}.csv'),
                            ('pitching', f'Statcast/pitchers_{season}.csv')]:
            df = pd.read_csv(fname)
            df['team'] = df['player_name'].replace(STATCAST_TEAM_MAP)
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=['xwoba', 'babip', 'barrels_per_pa_percent', 'pa'])
            df = df[df['pa'] > 0]
            for team in df['team'].unique():
                if team not in TEAM_MAP:
                    continue
                cols = df[df['team'] == team][['game_date', 'xwoba', 'babip', 'barrels_per_pa_percent', 'pa']]
                season_data[side][team] = cols.rename(
                    columns={'barrels_per_pa_percent': 'barrel_pct'}).to_dict('records')
        all_data[str(season)] = season_data
        print(f"  Loaded statcast for {season}")

    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print("Statcast data cached.")
    return all_data

statcast_data = load_statcast_data([2021, 2022, 2023, 2024])

def precompute_statcast(season, statcast_data, target_dates):
    season_str = str(season)
    sorted_dates = sorted(target_dates)
    team_lookup = {}

    for side in ['batting', 'pitching']:
        for team, games in statcast_data.get(season_str, {}).get(side, {}).items():
            games_sorted = sorted(games, key=lambda g: g['game_date'])
            cum_pa = 0; cum_x = 0.0; cum_b = 0.0; cum_br = 0.0
            game_count = 0; game_idx = 0

            for target_date in sorted_dates:
                while game_idx < len(games_sorted) and games_sorted[game_idx]['game_date'] < target_date:
                    g = games_sorted[game_idx]
                    pa = g['pa']
                    cum_x  += g['xwoba']      * pa
                    cum_b  += g['babip']      * pa
                    cum_br += g['barrel_pct'] * pa
                    cum_pa += pa
                    game_count += 1
                    game_idx += 1
                if game_count >= 7 and cum_pa > 0:
                    team_lookup[(team, side, target_date)] = {
                        'xwoba':      cum_x  / cum_pa,
                        'babip':      cum_b  / cum_pa,
                        'barrel_pct': cum_br / cum_pa,
                    }

    lg_avgs = {}
    for target_date in sorted_dates:
        xb, bb, brb, xp, bp, brp = [], [], [], [], [], []
        for team in TEAM_MAP:
            bs = team_lookup.get((team, 'batting',  target_date))
            ps = team_lookup.get((team, 'pitching', target_date))
            if bs:
                xb.append(bs['xwoba']); bb.append(bs['babip']); brb.append(bs['barrel_pct'])
            if ps:
                xp.append(ps['xwoba']); bp.append(ps['babip']); brp.append(ps['barrel_pct'])
        if len(xb) >= 10:
            lg_avgs[target_date] = {
                'xwoba_bat':  sum(xb)  / len(xb),
                'babip_bat':  sum(bb)  / len(bb),
                'barrel_bat': sum(brb) / len(brb),
                'xwoba_pit':  sum(xp)  / len(xp),
                'babip_pit':  sum(bp)  / len(bp),
                'barrel_pit': sum(brp) / len(brp),
            }
    return team_lookup, lg_avgs

def load_bullpen_data():
    cache = 'historical_data/bullpen_cache.json'
    if os.path.exists(cache):
        print("Loading bullpen data from cache...")
        with open(cache) as f:
            return json.load(f)

    print("Loading bullpen RP files...")
    import glob as _glob

    with open('historical_data/player_team_map.json') as f:
        player_map = json.load(f)

    cols = ['player_id', 'game_date', 'total_pitches', 'babip', 'woba',
            'xwoba', 'k_percent', 'bb_percent', 'barrels_per_pa_percent', 'pa']

    frames = []
    for fpath in _glob.glob('rp/*.csv'):
        df = pd.read_csv(fpath, usecols=cols)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=['woba', 'xwoba', 'pa'])
    all_df = all_df[all_df['pa'] > 0]
    all_df['player_id'] = all_df['player_id'].astype(str)
    all_df['game_date'] = pd.to_datetime(all_df['game_date']).dt.strftime('%Y-%m-%d')
    all_df['season'] = all_df['game_date'].str[:4]

    def get_team(row):
        return player_map.get(row['season'], {}).get(row['player_id'])
    all_df['team'] = all_df.apply(get_team, axis=1)
    all_df = all_df.dropna(subset=['team'])

    # Aggregate per team per game date
    result = {}
    grp_cols = ['team', 'season', 'game_date']
    for (team, season, date), g in all_df.groupby(grp_cols):
        total_pa = g['pa'].sum()
        if total_pa == 0:
            continue
        def wpct(col): return float((g[col] * g['pa']).sum() / total_pa)
        entry = {
            'date':       date,
            'pitches':    int(g['total_pitches'].sum()),
            'woba':       wpct('woba'),
            'xwoba':      wpct('xwoba'),
            'babip':      wpct('babip'),
            'k_pct':      wpct('k_percent'),
            'bb_pct':     wpct('bb_percent'),
            'barrel_pct': wpct('barrels_per_pa_percent'),
            'pa':         int(total_pa),
        }
        result.setdefault(season, {}).setdefault(team, []).append(entry)

    for season in result:
        for team in result[season]:
            result[season][team].sort(key=lambda g: g['date'])

    with open(cache, 'w') as f:
        json.dump(result, f)
    print("Bullpen data cached.")
    return result

bullpen_data = load_bullpen_data()

def precompute_bullpen(season, bullpen_data, target_dates):
    """
    For each (team, date) return:
      - pitches_7d  : total bullpen pitches in last 7 calendar days (fatigue)
      - season-to-date PA-weighted: xwoba, k_pct, bb_pct (quality)
    Also returns league averages per date.
    """
    season_str = str(season)
    sorted_dates = sorted(target_dates)
    team_lookup = {}

    for team, games in bullpen_data.get(season_str, {}).items():
        cum_pa = 0; cum_x = 0.0; cum_k = 0.0; cum_bb = 0.0
        game_count = 0; game_idx = 0

        for target_date in sorted_dates:
            # Advance cumulative season stats
            while game_idx < len(games) and games[game_idx]['date'] < target_date:
                g = games[game_idx]
                pa = g['pa']
                cum_x  += g['xwoba'] * pa
                cum_k  += g['k_pct'] * pa
                cum_bb += g['bb_pct'] * pa
                cum_pa += pa
                game_count += 1
                game_idx += 1

            if game_count < 5 or cum_pa == 0:
                continue

            # 7-day pitches window
            cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            pitches_7d = sum(g['pitches'] for g in games[:game_idx] if g['date'] >= cutoff)

            team_lookup[(team, target_date)] = {
                'pitches_7d': pitches_7d,
                'xwoba':      cum_x  / cum_pa,
                'k_pct':      cum_k  / cum_pa,
                'bb_pct':     cum_bb / cum_pa,
            }

    # League averages per date
    lg_avgs = {}
    for target_date in sorted_dates:
        p7, xw, kp, bbp = [], [], [], []
        for team in TEAM_MAP:
            s = team_lookup.get((team, target_date))
            if s:
                p7.append(s['pitches_7d']); xw.append(s['xwoba'])
                kp.append(s['k_pct']);      bbp.append(s['bb_pct'])
        if len(xw) >= 10:
            lg_avgs[target_date] = {
                'pitches_7d': sum(p7) / len(p7),
                'xwoba':      sum(xw) / len(xw),
                'k_pct':      sum(kp) / len(kp),
                'bb_pct':     sum(bbp) / len(bbp),
            }
    return team_lookup, lg_avgs

def ip_to_float(ip):
    ip = float(ip)
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3

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
        fip_ip  += ip_to_float(g['inningsPitched'])
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
        '_fip_num':     fip_num,
        '_fip_ip':      fip_ip,
        '_total_fb':    total_fb,
        '_k_total':     k_total,
        '_bb_hbp_total': bb_total + sum(g.get('hitBatsmen', 0) for g in pit_before),
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
# PRECOMPUTE ROLLING — called once at startup, reused every trial
# ============================================================

def precompute_all_rolling(seasons, game_logs, odds_df):
    """
    For every (season, team, date) compute rolling stats using a single sweep
    per team — O(games + dates) instead of O(games × dates) per game.
    Returns two dicts used as O(1) lookups inside run_backtest_with_params:
      rolling_team_cache[(season_str, team, date)] -> stats dict
      rolling_lg_cache[(season_str, date)]          -> league averages dict
    """
    print("Precomputing rolling averages (one-time)...")
    team_cache = {}
    lg_cache   = {}

    for season in seasons:
        season_str   = str(season)
        woba_weights, fip_constant = load_woba_fip_constants('constants/woba_fip_constants.csv', season)
        season_odds  = odds_df[odds_df['date'].str.startswith(season_str)]
        target_dates = sorted(set(season_odds['date'].tolist()))
        if not target_dates:
            continue

        wBB = woba_weights['wBB']; wHBP = woba_weights['wHBP']
        w1B = woba_weights['w1B']; w2B  = woba_weights['w2B']
        w3B = woba_weights['w3B']; wHR  = woba_weights['wHR']

        # ── Per-team sweep ──────────────────────────────────────────────────
        team_lookup = {}
        for team, team_logs in game_logs.get(season_str, {}).items():
            hitting  = sorted(team_logs['hitting'],  key=lambda g: g['date'])
            pitching = sorted(team_logs['pitching'], key=lambda g: g['date'])
            if not hitting or not pitching:
                continue

            # Prefix sums → O(1) sliding-window averages
            cum_hit = [0]
            for g in hitting:
                cum_hit.append(cum_hit[-1] + g['runs'])
            cum_pit = [0]
            for g in pitching:
                cum_pit.append(cum_pit[-1] + g['runs_allowed'])

            # Cumulative season stats (wOBA, FIP, K%, BB%)
            woba_num = woba_den = fip_num = fip_ip = 0.0
            k_total = bb_total = hbp_total = bf_total = total_fb = 0
            hit_idx = pit_idx = 0

            for target_date in target_dates:
                # Advance hitting
                while hit_idx < len(hitting) and hitting[hit_idx]['date'] < target_date:
                    g = hitting[hit_idx]
                    singles   = g['hits'] - g['doubles'] - g['triples'] - g['homeRuns']
                    woba_num += (wBB*g['baseOnBalls'] + wHBP*g['hitByPitch'] +
                                 w1B*singles + w2B*g['doubles'] +
                                 w3B*g['triples'] + wHR*g['homeRuns'])
                    woba_den += g['plateAppearances'] - g['intentionalWalks']
                    hit_idx  += 1

                # Advance pitching
                while pit_idx < len(pitching) and pitching[pit_idx]['date'] < target_date:
                    g = pitching[pit_idx]
                    fip_num  += 13*g['homeRuns'] + 3*(g['baseOnBalls']+g['hitBatsmen']) - 2*g['strikeOuts']
                    fip_ip   += ip_to_float(g['inningsPitched'])
                    k_total  += g['strikeOuts']
                    bb_total += g['baseOnBalls']
                    hbp_total+= g.get('hitBatsmen', 0)
                    bf_total += g['battersFaced']
                    total_fb += g['airOuts']
                    pit_idx  += 1

                if hit_idx < 7 or pit_idx < 7:
                    continue

                # Rolling windows — O(1) via prefix sums
                h7s  = max(0, hit_idx - 7);  h15s = max(0, hit_idx - 15)
                p7s  = max(0, pit_idx - 7);  p15s = max(0, pit_idx - 15)
                hit_7  = (cum_hit[hit_idx] - cum_hit[h7s])  / (hit_idx - h7s)
                hit_15 = (cum_hit[hit_idx] - cum_hit[h15s]) / (hit_idx - h15s)
                pit_7  = (cum_pit[pit_idx] - cum_pit[p7s])  / (pit_idx - p7s)
                pit_15 = (cum_pit[pit_idx] - cum_pit[p15s]) / (pit_idx - p15s)

                woba  = woba_num / woba_den if woba_den > 0 else 0.0
                fip   = (fip_num / fip_ip + fip_constant) if fip_ip > 0 else 4.50
                k_pct = k_total  / bf_total if bf_total > 0 else 0.20
                bb_pct= bb_total / bf_total if bf_total > 0 else 0.08

                team_lookup[(team, target_date)] = {
                    'hit_7': hit_7, 'hit_15': hit_15,
                    'pitch_7': pit_7, 'pitch_15': pit_15,
                    'woba': woba, 'fip': fip,
                    'k_pct': k_pct, 'bb_pct': bb_pct,
                    '_fip_ip':       fip_ip,
                    '_total_fb':     total_fb,
                    '_k_total':      k_total,
                    '_bb_hbp_total': bb_total + hbp_total,
                }

        # ── League HR/FB rate per date (needed for xFIP) ───────────────────
        all_pit_events = []
        for tl in game_logs.get(season_str, {}).values():
            for g in tl['pitching']:
                all_pit_events.append((g['date'], g['homeRuns'], g['airOuts'] + g['homeRuns']))
        all_pit_events.sort(key=lambda x: x[0])

        cum_hr = cum_fb = pit_ev_idx = 0
        lg_hr_fb_by_date = {}
        for target_date in target_dates:
            while pit_ev_idx < len(all_pit_events) and all_pit_events[pit_ev_idx][0] < target_date:
                cum_hr += all_pit_events[pit_ev_idx][1]
                cum_fb += all_pit_events[pit_ev_idx][2]
                pit_ev_idx += 1
            lg_hr_fb_by_date[target_date] = cum_hr / cum_fb if cum_fb > 0 else 0.115

        # ── Add xFIP to each team entry ────────────────────────────────────
        for target_date in target_dates:
            lg_hr_fb = lg_hr_fb_by_date[target_date]
            for team in TEAM_MAP:
                r = team_lookup.get((team, target_date))
                if r:
                    expected_hr = lg_hr_fb * r['_total_fb']
                    num = 13*expected_hr + 3*r['_bb_hbp_total'] - 2*r['_k_total']
                    r['xfip'] = (num / r['_fip_ip'] + fip_constant) if r['_fip_ip'] > 0 else 4.50

        # ── League averages per date ───────────────────────────────────────
        all_runs = [g['runs'] for tl in game_logs.get(season_str, {}).values()
                    for g in tl['hitting']]
        lg_avg_runs = sum(all_runs) / len(all_runs) if all_runs else 4.5

        for target_date in target_dates:
            woba_v = []; fip_v = []; xfip_v = []; k_v = []; bb_v = []
            for team in TEAM_MAP:
                r = team_lookup.get((team, target_date))
                if r:
                    woba_v.append(r['woba']); fip_v.append(r['fip'])
                    xfip_v.append(r['xfip']); k_v.append(r['k_pct'])
                    bb_v.append(r['bb_pct'])
            if len(woba_v) >= 10:
                lg_cache[(season_str, target_date)] = {
                    'woba':     sum(woba_v) / len(woba_v),
                    'fip':      sum(fip_v)  / len(fip_v),
                    'xfip':     sum(xfip_v) / len(xfip_v),
                    'k_pct':    sum(k_v)    / len(k_v),
                    'bb_pct':   sum(bb_v)   / len(bb_v),
                    'avg_runs': lg_avg_runs,
                }

        for (team, date), stats in team_lookup.items():
            team_cache[(season_str, team, date)] = stats

        print(f"  {season}: {len(team_lookup)} team-date entries precomputed")

    print("Rolling precompute complete.")
    return team_cache, lg_cache

rolling_team_cache, rolling_lg_cache = precompute_all_rolling(
    [2021, 2022, 2023, 2024], game_logs, odds_df
)

# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

def get_league_hr_fb_rate(game_logs, target_date, season):
    season_str = str(season)
    if season_str not in game_logs:
        return 0.115
    total_hr = 0
    total_fb = 0
    for team_logs in game_logs[season_str].values():
        for g in team_logs['pitching']:
            if g['date'] < target_date:
                total_hr += g['homeRuns']
                total_fb += g['airOuts'] + g['homeRuns']
    return total_hr / total_fb if total_fb > 0 else 0.115

def run_backtest_with_params(params, seasons=[2021, 2022, 2023, 2024]):
    all_results = []
    
    for season in seasons:
        season_odds = odds_df[odds_df['date'].str.startswith(str(season))]
        
        woba_weights, fip_constant = load_woba_fip_constants(
            'constants/woba_fip_constants.csv', season
        )

        season_str = str(season)

        # Precompute statcast + bullpen stats for all teams/dates this season
        season_dates = set(season_odds['date'].tolist())
        sc_lookup, sc_lg = precompute_statcast(season, statcast_data, season_dates)
        bp_lookup, bp_lg = precompute_bullpen(season, bullpen_data, season_dates)

        for _, game in season_odds.iterrows():
            date = game['date']
            home_team = normalize_team(game['home_team'])
            away_team = normalize_team(game['away_team'])

            if home_team not in TEAM_MAP or away_team not in TEAM_MAP:
                continue

            # O(1) lookups — no more per-game log scanning
            home_rolling = rolling_team_cache.get((season_str, home_team, date))
            away_rolling = rolling_team_cache.get((season_str, away_team, date))
            lg_roll      = rolling_lg_cache.get((season_str, date))

            if home_rolling is None or away_rolling is None or lg_roll is None:
                continue

            lg_avg_runs = lg_roll['avg_runs']
            lg_woba     = lg_roll['woba']
            lg_fip      = lg_roll['fip']
            lg_xfip     = lg_roll['xfip']
            lg_k_pit    = lg_roll['k_pct']
            lg_bb_pit   = lg_roll['bb_pct']

            home_xfip = home_rolling['xfip']
            away_xfip = away_rolling['xfip']

            home_off_roll = ((home_rolling['hit_7'] * params['rolling_weight_7'] +
                             home_rolling['hit_15'] * params['rolling_weight_15']) / lg_avg_runs)
            away_off_roll = ((away_rolling['hit_7'] * params['rolling_weight_7'] +
                             away_rolling['hit_15'] * params['rolling_weight_15']) / lg_avg_runs)
            home_pit_roll = (lg_avg_runs / (home_rolling['pitch_7'] * params['rolling_weight_7'] +
                             home_rolling['pitch_15'] * params['rolling_weight_15']))
            away_pit_roll = (lg_avg_runs / (away_rolling['pitch_7'] * params['rolling_weight_7'] +
                             away_rolling['pitch_15'] * params['rolling_weight_15']))

            home_woba_r = home_rolling['woba'] / lg_woba if lg_woba > 0 else 1.0
            away_woba_r = away_rolling['woba'] / lg_woba if lg_woba > 0 else 1.0
            home_fip_r  = lg_fip  / home_rolling['fip'] if home_rolling['fip'] > 0 else 1.0
            away_fip_r  = lg_fip  / away_rolling['fip'] if away_rolling['fip'] > 0 else 1.0
            home_xfip_r = lg_xfip / home_xfip           if home_xfip           > 0 else 1.0
            away_xfip_r = lg_xfip / away_xfip           if away_xfip           > 0 else 1.0
            home_k_pit_r  = home_rolling['k_pct'] / lg_k_pit  if lg_k_pit  > 0 else 1.0
            away_k_pit_r  = away_rolling['k_pct'] / lg_k_pit  if lg_k_pit  > 0 else 1.0
            home_bb_pit_r = lg_bb_pit / home_rolling['bb_pct'] if home_rolling['bb_pct'] > 0 else 1.0
            away_bb_pit_r = lg_bb_pit / away_rolling['bb_pct'] if away_rolling['bb_pct'] > 0 else 1.0

            # Statcast ratings — batting: higher is better; pitching: lower is better (inverted)
            lg_sc = sc_lg.get(date, {})
            home_sc_bat = sc_lookup.get((home_team, 'batting',  date), {})
            away_sc_bat = sc_lookup.get((away_team, 'batting',  date), {})
            home_sc_pit = sc_lookup.get((home_team, 'pitching', date), {})
            away_sc_pit = sc_lookup.get((away_team, 'pitching', date), {})

            def sc_bat_r(sc, key, lg_key): return sc[key] / lg_sc[lg_key] if sc and lg_sc and lg_sc.get(lg_key, 0) > 0 else 1.0
            def sc_pit_r(sc, key, lg_key): return lg_sc[lg_key] / sc[key]  if sc and lg_sc and sc.get(key, 0) > 0 else 1.0

            home_xwoba_bat_r  = sc_bat_r(home_sc_bat, 'xwoba',      'xwoba_bat')
            away_xwoba_bat_r  = sc_bat_r(away_sc_bat, 'xwoba',      'xwoba_bat')
            home_babip_bat_r  = sc_bat_r(home_sc_bat, 'babip',      'babip_bat')
            away_babip_bat_r  = sc_bat_r(away_sc_bat, 'babip',      'babip_bat')
            home_barrel_bat_r = sc_bat_r(home_sc_bat, 'barrel_pct', 'barrel_bat')
            away_barrel_bat_r = sc_bat_r(away_sc_bat, 'barrel_pct', 'barrel_bat')
            home_xwoba_pit_r  = sc_pit_r(home_sc_pit, 'xwoba',      'xwoba_pit')
            away_xwoba_pit_r  = sc_pit_r(away_sc_pit, 'xwoba',      'xwoba_pit')
            home_babip_pit_r  = sc_pit_r(home_sc_pit, 'babip',      'babip_pit')
            away_babip_pit_r  = sc_pit_r(away_sc_pit, 'babip',      'babip_pit')
            home_barrel_pit_r = sc_pit_r(home_sc_pit, 'barrel_pct', 'barrel_pit')
            away_barrel_pit_r = sc_pit_r(away_sc_pit, 'barrel_pct', 'barrel_pit')

            # Bullpen ratings — quality (lower xwOBA/BB%, higher K% = better) + fatigue
            lg_bp = bp_lg.get(date, {})
            home_bp = bp_lookup.get((home_team, date), {})
            away_bp = bp_lookup.get((away_team, date), {})

            def bp_r(bp, key, lg_key, invert=True):
                if not bp or not lg_bp or lg_bp.get(lg_key, 0) == 0 or bp.get(key, 0) == 0:
                    return 1.0
                return lg_bp[lg_key] / bp[key] if invert else bp[key] / lg_bp[lg_key]

            home_bp_qual_r = (bp_r(home_bp, 'xwoba',   'xwoba',   invert=True)  +
                              bp_r(home_bp, 'k_pct',   'k_pct',   invert=False) +
                              bp_r(home_bp, 'bb_pct',  'bb_pct',  invert=True)) / 3
            away_bp_qual_r = (bp_r(away_bp, 'xwoba',   'xwoba',   invert=True)  +
                              bp_r(away_bp, 'k_pct',   'k_pct',   invert=False) +
                              bp_r(away_bp, 'bb_pct',  'bb_pct',  invert=True)) / 3
            home_bp_fat_r  = bp_r(home_bp, 'pitches_7d', 'pitches_7d', invert=True)
            away_bp_fat_r  = bp_r(away_bp, 'pitches_7d', 'pitches_7d', invert=True)

            home_off = (home_off_roll    * params['w_rolling_off'] +
                        home_woba_r      * params['w_woba'] +
                        home_xwoba_bat_r * params['w_xwoba_bat'] +
                        home_babip_bat_r * params['w_babip_bat'])
            away_off = (away_off_roll    * params['w_rolling_off'] +
                        away_woba_r      * params['w_woba'] +
                        away_xwoba_bat_r * params['w_xwoba_bat'] +
                        away_babip_bat_r * params['w_babip_bat'])
            home_pit = (home_pit_roll    * params['w_rolling_pit'] +
                        home_fip_r       * params['w_fip'] +
                        home_xfip_r      * params['w_xfip'] +
                        home_k_pit_r     * params['w_k_pit'] +
                        home_bb_pit_r    * params['w_bb_pit'] +
                        home_xwoba_pit_r * params['w_xwoba_pit'] +
                        home_babip_pit_r * params['w_babip_pit'] +
                        home_barrel_pit_r* params['w_barrel_pit'])
            away_pit = (away_pit_roll    * params['w_rolling_pit'] +
                        away_fip_r       * params['w_fip'] +
                        away_xfip_r      * params['w_xfip'] +
                        away_k_pit_r     * params['w_k_pit'] +
                        away_bb_pit_r    * params['w_bb_pit'] +
                        away_xwoba_pit_r * params['w_xwoba_pit'] +
                        away_babip_pit_r * params['w_babip_pit'] +
                        away_barrel_pit_r* params['w_barrel_pit'])
            
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

N_RUNS = 10  # ~2 hours at 12 min/run

for run in range(N_RUNS):
    print(f"\n{'='*55}")
    print(f"  Run {run + 1} of {N_RUNS}")
    print(f"{'='*55}")
    random_search(n_trials=100)

evaluate_runs()
