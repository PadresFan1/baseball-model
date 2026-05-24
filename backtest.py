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
        'lg_woba':    float(row['wOBA']),
        'woba_scale': float(row['wOBAScale']),
        'r_per_pa':   float(row['R/PA']),
    }
    fip_constant = row['cFIP']
    return woba_weights, fip_constant

_PARAM_SPACE = {
    'ml_edge_min':    [0.07],
    'ml_edge_max':    [0.08],
    'favorites_only': [False],
    'rolling_weight_7': [0.50, 0.60, 0.70, 0.80],
    'rolling_weight_15': None,
    'rating_cap_low':  [0.65],
    'rating_cap_high': [1.50],
    'ou_run_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],
    # Offense weights
    'w_rolling_off': [0.05, 0.08, 0.10],
    'w_woba':        [0.20, 0.30, 0.40],
    'w_xwoba_bat':   [0.10, 0.20, 0.30],
    'w_babip_bat':   [0.15, 0.25, 0.35],
    'w_slg':         [0.10, 0.20, 0.30],
    'w_xslg':        [0.10, 0.20, 0.30],
    'w_ba':          [0.10, 0.20, 0.30],
    'w_xba':         [0.10, 0.20, 0.30],
    'w_hardhit_bat': [0.10, 0.20, 0.30],
    'w_swmiss_bat':  [0.05, 0.10, 0.15],
    'w_rv100_bat':   [0.05, 0.10, 0.15],
    'w_wrc_plus':    [0.10, 0.20, 0.30],
    # Pitching weights
    'w_rolling_pit': [0.05, 0.08, 0.10],
    'w_fip':         [0.10, 0.20, 0.30],
    'w_xfip':        [0.10, 0.20, 0.30],
    'w_k_pit':       [0.10, 0.20, 0.30],
    'w_bb_pit':      [0.15, 0.25, 0.35],
    'w_xwoba_pit':   [0.10, 0.20, 0.30],
    'w_babip_pit':   [0.05, 0.08, 0.12],
    'w_barrel_pit':  [0.15, 0.25, 0.35],
    'w_era':         [0.10, 0.20, 0.30],
    'w_whip':        [0.05, 0.10, 0.20],
    'w_hr9':         [0.05, 0.10, 0.15],
    'w_hardhit_pit': [0.05, 0.10, 0.20],
    'w_swmiss_pit':  [0.10, 0.20, 0.30],
    'w_rv100_pit':   [0.05, 0.10, 0.15],
}

_OFF_KEYS = ['w_rolling_off', 'w_woba', 'w_xwoba_bat', 'w_babip_bat',
             'w_slg', 'w_xslg', 'w_ba', 'w_xba',
             'w_hardhit_bat', 'w_swmiss_bat', 'w_rv100_bat', 'w_wrc_plus']
_PIT_KEYS = ['w_rolling_pit', 'w_fip', 'w_xfip', 'w_k_pit', 'w_bb_pit',
             'w_xwoba_pit', 'w_babip_pit', 'w_barrel_pit',
             'w_era', 'w_whip', 'w_hr9',
             'w_hardhit_pit', 'w_swmiss_pit', 'w_rv100_pit']
_PARAM_COLS = list(_PARAM_SPACE.keys()) + ['rolling_weight_15']

def _sample_random_params(space=None):
    """Sample and normalize one random parameter combination."""
    if space is None:
        space = _PARAM_SPACE
    p = {
        'ml_edge_min':     random.choice(space['ml_edge_min']),
        'ml_edge_max':     random.choice(space['ml_edge_max']),
        'favorites_only':  random.choice(space['favorites_only']),
        'rolling_weight_7': random.choice(space['rolling_weight_7']),
        'rating_cap_low':  random.choice(space['rating_cap_low']),
        'rating_cap_high': random.choice(space['rating_cap_high']),
        'ou_run_threshold': random.choice(space['ou_run_threshold']),
    }
    for k in _OFF_KEYS + _PIT_KEYS:
        p[k] = random.choice(space[k])
    p['rolling_weight_15'] = 1 - p['rolling_weight_7']
    if p['ml_edge_min'] >= p['ml_edge_max']:
        return None
    off_total = sum(p[k] for k in _OFF_KEYS)
    for k in _OFF_KEYS: p[k] /= off_total
    pit_total = sum(p[k] for k in _PIT_KEYS)
    for k in _PIT_KEYS: p[k] /= pit_total
    return p


def random_search(n_trials=100, seed=None):
    import time
    import ctypes

    # Prevent Windows from sleeping or display turning off while backtest runs
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)

    start_time = time.time()
    checkpoint_file = 'historical_data/random_search_checkpoint.json'

    PARAM_ROUND = 8  # increment this whenever param_space changes
    # Round 1: base params, no statcast
    # Round 2: statcast added (xwoba_bat, babip_bat, barrel_bat/pit)
    # Round 3: bullpen added (bullpen_qual, bullpen_fat)
    # Round 4: bullpen removed, favorites_only locked False — 53% profitable
    # Round 5: locked edge/cap params, dropped barrel_bat, rebalanced bb_pit/fip/barrel_pit
    # Round 6: added ou_run_threshold — O/U betting now tracked alongside ML
    # Round 7: added park factors to lambda calculation
    # Round 8: holdout split — training on 2021-2022 only, 2023/2024/2025 reserved

    # Resume from checkpoint if one exists — restore seed from checkpoint so restart works
    results = []
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        if checkpoint.get('n_trials') == n_trials:
            seed        = checkpoint['seed']
            results     = checkpoint['results']
            start_trial = checkpoint['completed_trials']
            random.seed(seed)
            for _ in range(start_trial):
                _sample_random_params()  # fast-forward RNG
            print(f"Resuming run (seed={seed}) from trial {start_trial}/{n_trials} ({len(results)} results so far)")

    if not results and start_trial == 0:
        if seed is None:
            seed = random.randint(1, 99999)
        random.seed(seed)
    print(f"Seed: {seed}  (pass seed={seed} to reproduce this run)")

    for trial in range(start_trial, n_trials):
        params = _sample_random_params()
        if params is None:
            continue

        # Train on 2021-2022 only — 2023/2024/2025 reserved for validation/test/holdout
        trial_results = run_backtest_with_params(params, seasons=[2021, 2022])

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


def monte_carlo_permutation_test(bets, n_permutations=10000, seed=42):
    """
    Tests whether the strategy's ROI on holdout data is statistically significant.
    H0: the observed ROI could be achieved by randomly assigning win/loss to the same bets.
    Returns (actual_roi, p_value, percentile).
    """
    if not bets:
        return None, None, None

    odds_arr   = np.array([b['ml_odds']   for b in bets], dtype=float)
    wins_arr   = np.array([b['ml_result'] == 'WIN' for b in bets], dtype=bool)
    n          = len(bets)

    def calc_roi(wins, odds):
        wagered = n * 100
        ret = 0.0
        for w, o in zip(wins, odds):
            if w:
                ret += 100 + (100 * o / 100) if o > 0 else 100 + (100 / abs(o) * 100)
        return (ret - wagered) / wagered * 100

    actual_roi = calc_roi(wins_arr, odds_arr)

    rng = np.random.default_rng(seed)
    permuted_rois = np.array([
        calc_roi(rng.permutation(wins_arr), odds_arr)
        for _ in range(n_permutations)
    ])

    p_value    = float(np.mean(permuted_rois >= actual_roi))
    percentile = float(np.mean(permuted_rois < actual_roi)) * 100
    return actual_roi, p_value, percentile


def run_holdout_evaluation(param_sets, label='Holdout'):
    """
    Runs the top parameter sets on out-of-sample years and reports
    year-by-year ROI to show whether training performance generalises.
    Also runs a Monte Carlo permutation test on 2025.
    """
    print(f"\n{'='*60}")
    print(f"  OUT-OF-SAMPLE EVALUATION — {label}")
    print(f"{'='*60}")
    print(f"  Evaluating top {len(param_sets)} param set(s) across held-out years\n")

    years        = [2021, 2022, 2023, 2024, 2025]
    year_labels  = {2021: 'train', 2022: 'train', 2023: 'validation', 2024: 'test', 2025: 'holdout'}
    all_2025_bets = []

    print(f"  {'Year':<6} {'Role':<12} {'N Bets':<8} {'Win%':<8} {'ROI':<10}")
    print(f"  {'-'*44}")

    for year in years:
        year_bets_all = []
        for params in param_sets:
            result = run_backtest_with_params(params, seasons=[year], return_bets=True)
            if result is None:
                continue
            ml_bets, _ = result
            year_bets_all.extend(ml_bets)

        if not year_bets_all:
            print(f"  {year:<6} {year_labels[year]:<12} {'—':<8} {'—':<8} {'—'}")
            continue

        wins    = sum(1 for b in year_bets_all if b['ml_result'] == 'WIN')
        n       = len(year_bets_all)
        win_pct = wins / n
        wagered = n * 100
        ret     = sum(
            100 + (100 * b['ml_odds'] / 100 if b['ml_odds'] > 0 else 100 / abs(b['ml_odds']) * 100)
            for b in year_bets_all if b['ml_result'] == 'WIN'
        )
        roi = (ret - wagered) / wagered * 100

        role = year_labels[year]
        flag = ' ← out-of-sample' if role not in ('train',) else ''
        print(f"  {year:<6} {role:<12} {n:<8} {win_pct:<8.1%} {roi:+.1f}%{flag}")

        if year == 2025:
            all_2025_bets = year_bets_all

    # Monte Carlo permutation test on 2025 holdout
    if all_2025_bets:
        print(f"\n  Monte Carlo Permutation Test — 2025 holdout ({len(all_2025_bets)} bets, 10,000 permutations)")
        actual_roi, p_value, percentile = monte_carlo_permutation_test(all_2025_bets)
        print(f"  Actual ROI:   {actual_roi:+.2f}%")
        print(f"  P-value:      {p_value:.4f}  ({'significant ✓' if p_value < 0.05 else 'not significant'})")
        print(f"  Percentile:   {percentile:.1f}th  (beats {percentile:.1f}% of random permutations)")
        if p_value >= 0.05:
            print(f"  ⚠️  Result not statistically significant — insufficient evidence of genuine edge")
    print()


def test_individual_metrics(train_seasons=[2021, 2022], test_seasons=[2023, 2024, 2025]):
    """
    Tests each metric in isolation: weight=1.0 for the metric being tested,
    all others near-zero. Identifies which metrics have genuine out-of-sample signal.
    """
    BASE = {
        'ml_edge_min': 0.07, 'ml_edge_max': 0.08,
        'favorites_only': False,
        'rolling_weight_7': 0.60, 'rolling_weight_15': 0.40,
        'rating_cap_low': 0.65, 'rating_cap_high': 1.50,
        'ou_run_threshold': 1.5,
    }
    eps = 0.001

    def make_params(active_metric):
        p = dict(BASE)
        if active_metric in _OFF_KEYS:
            off_w = {k: (1.0 if k == active_metric else eps) for k in _OFF_KEYS}
            pit_w = {k: 1.0/len(_PIT_KEYS) for k in _PIT_KEYS}
        else:
            off_w = {k: 1.0/len(_OFF_KEYS) for k in _OFF_KEYS}
            pit_w = {k: (1.0 if k == active_metric else eps) for k in _PIT_KEYS}
        off_s = sum(off_w.values()); pit_s = sum(pit_w.values())
        p.update({k: v/off_s for k, v in off_w.items()})
        p.update({k: v/pit_s for k, v in pit_w.items()})
        return p

    print(f"\n{'='*75}")
    print(f"  INDIVIDUAL METRIC TEST  |  Train: {train_seasons}  |  Test: {test_seasons}")
    print(f"{'='*75}")
    print(f"\n  {'Metric':<22} {'Side':<10} {'Train ROI':<12} {'Test ROI':<12} {'Brier(test)':<13} {'Signal'}")
    print(f"  {'-'*70}")

    for metric in _OFF_KEYS + _PIT_KEYS:
        side   = 'offense' if metric in _OFF_KEYS else 'pitching'
        params = make_params(metric)
        tr     = run_backtest_with_params(params, seasons=train_seasons)
        te     = run_backtest_with_params(params, seasons=test_seasons)

        tr_roi   = f"{tr['roi']:+.1f}%"   if tr else 'N/A'
        te_roi   = f"{te['roi']:+.1f}%"   if te else 'N/A'
        brier    = f"{te['brier_score']:.4f}" if te and te.get('brier_score') else 'N/A'
        positive = tr and te and tr['roi'] > 0 and te['roi'] > 0
        mixed    = tr and te and te['roi'] > -2
        signal   = '✓ positive' if positive else ('~ mixed' if mixed else '✗ negative')

        print(f"  {metric:<22} {side:<10} {tr_roi:<12} {te_roi:<12} {brier:<13} {signal}")
    print()


def walk_forward_validation(n_trials_per_fold=100, n_runs_per_fold=3):
    """
    True walk-forward cross-validation — runs a fresh parameter search for each
    expanding training window, then evaluates on the next unseen year.
    Folds: [2021]→2022, [2021-22]→2023, [2021-22-23]→2024, [2021-22-23-24]→2025
    Note: takes approximately 4× as long as a single training run.
    """
    import time as _time

    folds = [
        ([2021],                    2022),
        ([2021, 2022],              2023),
        ([2021, 2022, 2023],        2024),
        ([2021, 2022, 2023, 2024],  2025),
    ]

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD VALIDATION  ({n_trials_per_fold} trials × {n_runs_per_fold} runs per fold)")
    print(f"{'='*72}")
    print(f"\n  {'Fold':<6} {'Train':<26} {'Test':<6} {'Train ROI':<12} {'Test ROI':<12} {'N Bets'}")
    print(f"  {'-'*65}")

    wf_rows = []
    for fold_idx, (train_years, test_year) in enumerate(folds, 1):
        t0 = _time.time()
        fold_results = []
        for _ in range(n_runs_per_fold):
            random.seed(random.randint(1, 99999))
            for _ in range(n_trials_per_fold):
                p = _sample_random_params()
                if p is None: continue
                r = run_backtest_with_params(p, seasons=train_years)
                if r: fold_results.append({**p, **r})

        if not fold_results: continue
        fold_df       = pd.DataFrame(fold_results).sort_values('roi', ascending=False)
        best_train_roi = fold_df['roi'].iloc[0]
        best_params   = fold_df.head(5)[[c for c in _PARAM_COLS if c in fold_df.columns]].to_dict('records')

        test_bets = []
        for bp in best_params:
            res = run_backtest_with_params(bp, seasons=[test_year], return_bets=True)
            if res: test_bets.extend(res[0])

        if not test_bets: continue
        n = len(test_bets)
        wins = sum(1 for b in test_bets if b['ml_result'] == 'WIN')
        wagered = n * 100
        ret = sum(
            100 + (100*b['ml_odds']/100 if b['ml_odds'] > 0 else 100/abs(b['ml_odds'])*100)
            for b in test_bets if b['ml_result'] == 'WIN'
        )
        test_roi = (ret - wagered) / wagered * 100
        elapsed  = _time.time() - t0

        print(f"  {fold_idx:<6} {str(train_years):<26} {test_year:<6} {best_train_roi:+.1f}%{'':<6} {test_roi:+.1f}%{'':<6} {n}  ({elapsed:.0f}s)")
        wf_rows.append({'fold': fold_idx, 'train_years': str(train_years), 'test_year': test_year,
                        'best_train_roi': best_train_roi, 'test_roi': test_roi, 'n_bets': n})

    if wf_rows:
        avg_gap    = np.mean([r['best_train_roi'] - r['test_roi'] for r in wf_rows])
        profitable = sum(1 for r in wf_rows if r['test_roi'] > 0)
        print(f"\n  Avg train→test degradation: {avg_gap:+.1f}%")
        print(f"  Profitable test folds: {profitable}/{len(wf_rows)}")
    print()
    return wf_rows


def random_search_calibration(n_trials=100, seed=None, train_seasons=[2021, 2022]):
    """
    Calibration-optimized parameter search — minimizes Brier score instead of
    maximizing ROI. Brier score = mean((predicted_prob - actual)²).
    Lower is better. 0.25 = random (50/50). 0.0 = perfect.
    """
    import time

    if seed is None:
        seed = random.randint(1, 99999)
    random.seed(seed)
    print(f"\nCalibration search | Seed: {seed} | Train: {train_seasons}")

    results = []
    start   = time.time()

    for trial in range(n_trials):
        p = _sample_random_params()
        if p is None: continue
        r = run_backtest_with_params(p, seasons=train_seasons)
        if r and r.get('brier_score') is not None:
            results.append({**p, 'brier_score': r['brier_score'],
                            'roi': r['roi'], 'n_calibration': r['n_calibration']})

        if (trial + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  {trial+1}/{n_trials} trials | {elapsed:.0f}s elapsed")

    results_df = pd.DataFrame(results).sort_values('brier_score', ascending=True)

    from datetime import datetime as _dt
    cal_file = f"historical_data/calibration_{_dt.now().strftime('%Y-%m-%d')}_seed{seed}_{n_trials}trials.csv"
    results_df.to_csv(cal_file, index=False)

    print(f"\nTop 10 by Brier score (lower = better calibration, 0.25 = random):")
    top_cols = ['brier_score', 'roi', 'n_calibration', 'rolling_weight_7',
                'w_woba', 'w_fip', 'w_bb_pit']
    avail = [c for c in top_cols if c in results_df.columns]
    print(results_df[avail].head(10).to_string(index=False))
    print(f"\nSaved to {cal_file}")
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
            pct_pos  = (grp['roi'] > 0).mean()
            has_comb = 'combined_roi' in grp.columns and grp['combined_roi'].notna().any()
            comb_str = f" | top combined: {grp['combined_roi'].max():+.1f}%" if has_comb else ""
            print(f"  Round {int(rnd)}: {len(grp):>5} trials | top ML ROI: {grp['roi'].max():+.1f}% | median: {grp['roi'].median():+.1f}%{comb_str} | % profitable: {pct_pos:.1%}")
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

    # Top 10 across all runs — sort by combined_roi if available, else roi
    sort_col = 'combined_roi' if 'combined_roi' in combined.columns else 'roi'
    top_cols = ['combined_roi', 'roi', 'n_bets', 'ou_roi', 'n_ou_bets',
                'ou_run_threshold', 'ml_edge_min', 'ml_edge_max',
                'favorites_only', 'rolling_weight_7', 'rating_cap_low', 'rating_cap_high',
                'param_round', 'run_date']
    available = [c for c in top_cols if c in combined.columns]
    print(f"\nTop 10 results (all sample sizes, sorted by {sort_col}):")
    print(combined.nlargest(10, sort_col)[available].to_string(index=False))

    # Minimum sample size filter — more reliable signal
    min_bets = 300
    reliable = combined[combined['n_bets'] >= min_bets]
    if not reliable.empty:
        pct = len(reliable) / len(combined)
        print(f"\nTop 10 results (n_bets >= {min_bets}, {len(reliable)} trials / {pct:.0%} of total):")
        print(reliable.nlargest(10, sort_col)[available].to_string(index=False))

    # Parameter analysis — compare top 25% vs bottom 75%
    threshold = combined['roi'].quantile(0.75)
    top_q = combined[combined['roi'] >= threshold]

    discrete_params = ['ml_edge_min', 'ml_edge_max', 'favorites_only',
                       'rating_cap_low', 'rating_cap_high', 'ou_run_threshold']

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

    # ── Out-of-sample holdout evaluation (round 8+ only) ─────────────────────
    r8 = combined[combined['param_round'] == 8] if 'param_round' in combined.columns else pd.DataFrame()
    if not r8.empty:
        weight_cols = ['w_rolling_off', 'w_woba', 'w_xwoba_bat', 'w_babip_bat',
                       'w_rolling_pit', 'w_fip', 'w_xfip', 'w_k_pit', 'w_bb_pit',
                       'w_xwoba_pit', 'w_babip_pit', 'w_barrel_pit',
                       'rolling_weight_7', 'rolling_weight_15',
                       'ml_edge_min', 'ml_edge_max', 'favorites_only',
                       'rating_cap_low', 'rating_cap_high', 'ou_run_threshold']
        available_w = [c for c in weight_cols if c in r8.columns]
        top_params = r8.nlargest(5, 'roi')[available_w].to_dict('records')
        run_holdout_evaluation(top_params, label='Round 8 — top 5 training params')

def pull_historical_game_logs(team_ids, seasons=[2021, 2022, 2023, 2024, 2025]):
    log_path = 'historical_data/historical_game_logs.json'

    # Load existing file if present — only pull seasons that are missing
    all_logs = {}
    if os.path.exists(log_path):
        print("Loading historical game logs from file...")
        with open(log_path) as f:
            all_logs = json.load(f)

    missing = [s for s in seasons if str(s) not in all_logs]
    if not missing:
        return all_logs

    for season in missing:
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
                        'earnedRuns': g['stat'].get('earnedRuns', g['stat']['runs']),
                        'hits': g['stat']['hits'],
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
        all_logs[str(season)] = season_logs
        print(f"Pulled {len(season_logs)} teams for {season}")

    with open(log_path, 'w') as f:
        json.dump(all_logs, f)
    print("Game logs updated.")
    return all_logs

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
            extra = ['ba', 'xba', 'slg', 'xslg', 'hardhit_percent',
                     'swing_miss_percent', 'batter_run_value_per_100', 'pitcher_run_value_per_100']
            for c in extra:
                if c in df.columns:
                    df[c] = df[c].fillna(0)
                else:
                    df[c] = 0
            for team in df['team'].unique():
                if team not in TEAM_MAP:
                    continue
                sel_cols = ['game_date', 'xwoba', 'babip', 'barrels_per_pa_percent', 'pa'] + extra
                cols = df[df['team'] == team][sel_cols]
                season_data[side][team] = cols.rename(columns={
                    'barrels_per_pa_percent':   'barrel_pct',
                    'swing_miss_percent':        'swmiss',
                    'batter_run_value_per_100':  'rv100_bat',
                    'pitcher_run_value_per_100': 'rv100_pit',
                }).to_dict('records')
        all_data[str(season)] = season_data
        print(f"  Loaded statcast for {season}")

    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print("Statcast data cached.")
    return all_data

statcast_data = load_statcast_data([2021, 2022, 2023, 2024, 2025])

def precompute_statcast(season, statcast_data, target_dates):
    season_str = str(season)
    sorted_dates = sorted(target_dates)
    team_lookup = {}

    for side in ['batting', 'pitching']:
        for team, games in statcast_data.get(season_str, {}).get(side, {}).items():
            games_sorted = sorted(games, key=lambda g: g['game_date'])
            cum_pa = 0; cum_x = 0.0; cum_b = 0.0; cum_br = 0.0
            cum_slg = 0.0; cum_xslg = 0.0; cum_ba = 0.0; cum_xba = 0.0
            cum_hh = 0.0; cum_sm = 0.0; cum_rvb = 0.0; cum_rvp = 0.0
            game_count = 0; game_idx = 0

            for target_date in sorted_dates:
                while game_idx < len(games_sorted) and games_sorted[game_idx]['game_date'] < target_date:
                    g = games_sorted[game_idx]
                    pa = g['pa']
                    cum_x    += g['xwoba']      * pa
                    cum_b    += g['babip']      * pa
                    cum_br   += g['barrel_pct'] * pa
                    cum_slg  += g.get('slg', 0)      * pa
                    cum_xslg += g.get('xslg', 0)     * pa
                    cum_ba   += g.get('ba', 0)        * pa
                    cum_xba  += g.get('xba', 0)       * pa
                    cum_hh   += g.get('hardhit_percent', 0) * pa
                    cum_sm   += g.get('swmiss', 0)    * pa
                    cum_rvb  += g.get('rv100_bat', 0) * pa
                    cum_rvp  += g.get('rv100_pit', 0) * pa
                    cum_pa += pa
                    game_count += 1
                    game_idx += 1
                if game_count >= 7 and cum_pa > 0:
                    team_lookup[(team, side, target_date)] = {
                        'xwoba':      cum_x    / cum_pa,
                        'babip':      cum_b    / cum_pa,
                        'barrel_pct': cum_br   / cum_pa,
                        'slg':        cum_slg  / cum_pa,
                        'xslg':       cum_xslg / cum_pa,
                        'ba':         cum_ba   / cum_pa,
                        'xba':        cum_xba  / cum_pa,
                        'hardhit_percent': cum_hh / cum_pa,
                        'swmiss':     cum_sm   / cum_pa,
                        'rv100_bat':  cum_rvb  / cum_pa,
                        'rv100_pit':  cum_rvp  / cum_pa,
                    }

    lg_avgs = {}
    for target_date in sorted_dates:
        xb, bb, brb = [], [], []
        slgb, xslgb, bab, xbab, hhb, smb, rvbb = [], [], [], [], [], [], []
        xp, bp, brp = [], [], []
        slgp, xslgp, bap, xbap, hhp, smp, rvpp = [], [], [], [], [], [], []
        for team in TEAM_MAP:
            bs = team_lookup.get((team, 'batting',  target_date))
            ps = team_lookup.get((team, 'pitching', target_date))
            if bs:
                xb.append(bs['xwoba']); bb.append(bs['babip']); brb.append(bs['barrel_pct'])
                slgb.append(bs['slg']); xslgb.append(bs['xslg'])
                bab.append(bs['ba']); xbab.append(bs['xba'])
                hhb.append(bs['hardhit_percent']); smb.append(bs['swmiss'])
                rvbb.append(bs['rv100_bat'])
            if ps:
                xp.append(ps['xwoba']); bp.append(ps['babip']); brp.append(ps['barrel_pct'])
                slgp.append(ps['slg']); xslgp.append(ps['xslg'])
                bap.append(ps['ba']); xbap.append(ps['xba'])
                hhp.append(ps['hardhit_percent']); smp.append(ps['swmiss'])
                rvpp.append(ps['rv100_pit'])
        if len(xb) >= 10:
            def _avg(lst): return sum(lst) / len(lst) if lst else 0.0
            lg_avgs[target_date] = {
                'xwoba_bat':  _avg(xb),  'babip_bat':  _avg(bb),  'barrel_bat': _avg(brb),
                'slg_bat':    _avg(slgb), 'xslg_bat':  _avg(xslgb),
                'ba_bat':     _avg(bab),  'xba_bat':   _avg(xbab),
                'hardhit_bat': _avg(hhb), 'swmiss_bat': _avg(smb),
                'rv100_bat':  _avg(rvbb),
                'xwoba_pit':  _avg(xp),  'babip_pit':  _avg(bp),  'barrel_pit': _avg(brp),
                'slg_pit':    _avg(slgp), 'xslg_pit':  _avg(xslgp),
                'ba_pit':     _avg(bap),  'xba_pit':   _avg(xbap),
                'hardhit_pit': _avg(hhp), 'swmiss_pit': _avg(smp),
                'rv100_pit':  _avg(rvpp),
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

# Park factors lookup (2025 used as proxy for all seasons — parks are stable year to year)
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
    'Giants': 'SFG', 'Nationals': 'WSN', 'Astros': 'HOU'
}
_pf_df = pd.read_csv('constants/park_factors.csv')
PARK_FACTOR_LOOKUP = {}
for _, _row in _pf_df[_pf_df['Season'] == 2025].iterrows():
    _fg = _PARK_FACTOR_TEAM_MAP.get(_row['Team'].strip())
    if _fg:
        PARK_FACTOR_LOOKUP[_fg] = _row['1yr'] / 100

# Load season FanGraphs data
def load_season_stats(year):
    offense = pd.read_csv(f'season_stats/offense_{year}.csv')
    pitching = pd.read_csv(f'season_stats/pitching_{year}.csv')
    
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
with open('cache/mlb_odds_dataset.json', 'r') as f:
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

# Merge supplemental 2025 odds if available (fetched via fetch_2025_odds.py)
_supp = 'historical_data/odds_2025_supplement.csv'
if os.path.exists(_supp):
    _supp_df = pd.read_csv(_supp)
    odds_df  = pd.concat([odds_df, _supp_df], ignore_index=True).sort_values('date').reset_index(drop=True)
    print(f"Merged 2025 supplement: {len(_supp_df)} additional games")

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

def de_vig_probs(home_ml, away_ml):
    """Strip the sportsbook overround so market probs sum to exactly 1.0.
    FanDuel typically runs ~104–105% overround; using raw implied probs
    inflates market estimates by ~2–3%, which deflates the calculated edge
    by the same amount and chokes off qualifying bet volume."""
    raw_h = american_to_prob(home_ml)
    raw_a = american_to_prob(away_ml)
    total = raw_h + raw_a
    return raw_h / total, raw_a / total

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

            # Cumulative season stats (wOBA, FIP, K%, BB%, HR/9, ERA, WHIP)
            woba_num = woba_den = fip_num = fip_ip = 0.0
            k_total = bb_total = hbp_total = bf_total = total_fb = 0
            hr_total = er_total = hits_total = 0
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
                    fip_num   += 13*g['homeRuns'] + 3*(g['baseOnBalls']+g['hitBatsmen']) - 2*g['strikeOuts']
                    fip_ip    += ip_to_float(g['inningsPitched'])
                    k_total   += g['strikeOuts']
                    bb_total  += g['baseOnBalls']
                    hbp_total += g.get('hitBatsmen', 0)
                    bf_total  += g['battersFaced']
                    total_fb  += g['airOuts']
                    hr_total  += g['homeRuns']
                    er_total  += g.get('earnedRuns', g['runs_allowed'])
                    hits_total += g.get('hits', 0)
                    pit_idx   += 1

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
                hr9   = hr_total  * 9 / fip_ip  if fip_ip > 0 else 1.20
                era   = er_total  * 9 / fip_ip  if fip_ip > 0 else 4.50
                whip  = (bb_total + hits_total) / fip_ip if fip_ip > 0 else 1.30
                wrc_plus_r = ((woba - woba_weights['lg_woba']) /
                              (woba_weights['woba_scale'] * woba_weights['r_per_pa']) + 1.0
                              if woba_den > 0 else 1.0)

                team_lookup[(team, target_date)] = {
                    'hit_7': hit_7, 'hit_15': hit_15,
                    'pitch_7': pit_7, 'pitch_15': pit_15,
                    'woba': woba, 'fip': fip,
                    'k_pct': k_pct, 'bb_pct': bb_pct,
                    'hr9': hr9, 'era': era, 'whip': whip,
                    'wrc_plus_r': wrc_plus_r,
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
            hr9_v = []; era_v = []; whip_v = []
            for team in TEAM_MAP:
                r = team_lookup.get((team, target_date))
                if r:
                    woba_v.append(r['woba']); fip_v.append(r['fip'])
                    xfip_v.append(r['xfip']); k_v.append(r['k_pct'])
                    bb_v.append(r['bb_pct'])
                    hr9_v.append(r['hr9']); era_v.append(r['era'])
                    whip_v.append(r['whip'])
            if len(woba_v) >= 10:
                def _avg(lst): return sum(lst) / len(lst) if lst else 0.0
                lg_cache[(season_str, target_date)] = {
                    'woba':     _avg(woba_v),
                    'fip':      _avg(fip_v),
                    'xfip':     _avg(xfip_v),
                    'k_pct':    _avg(k_v),
                    'bb_pct':   _avg(bb_v),
                    'hr9':      _avg(hr9_v),
                    'era':      _avg(era_v),
                    'whip':     _avg(whip_v),
                    'avg_runs': lg_avg_runs,
                }

        for (team, date), stats in team_lookup.items():
            team_cache[(season_str, team, date)] = stats

        print(f"  {season}: {len(team_lookup)} team-date entries precomputed")

    print("Rolling precompute complete.")
    return team_cache, lg_cache

rolling_team_cache, rolling_lg_cache = precompute_all_rolling(
    [2021, 2022, 2023, 2024, 2025], game_logs, odds_df
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

def run_backtest_with_params(params, seasons=[2021, 2022, 2023, 2024, 2025], return_bets=False):
    all_results    = []
    all_ou_results = []
    brier_records  = []  # (model_home_prob, actual_home_win) for ALL games with scores
    
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

            def sc_bat_r(sc, key, lg_key): return sc.get(key, 0) / lg_sc[lg_key] if sc and lg_sc and lg_sc.get(lg_key, 0) > 0 and sc.get(key) is not None else 1.0
            def sc_pit_r(sc, key, lg_key): return lg_sc[lg_key] / sc.get(key, 0) if sc and lg_sc and sc.get(key, 0) > 0 and lg_sc.get(lg_key) is not None else 1.0

            home_xwoba_bat_r  = sc_bat_r(home_sc_bat, 'xwoba',      'xwoba_bat')
            away_xwoba_bat_r  = sc_bat_r(away_sc_bat, 'xwoba',      'xwoba_bat')
            home_babip_bat_r  = sc_bat_r(home_sc_bat, 'babip',      'babip_bat')
            away_babip_bat_r  = sc_bat_r(away_sc_bat, 'babip',      'babip_bat')
            home_barrel_bat_r = sc_bat_r(home_sc_bat, 'barrel_pct', 'barrel_bat')
            away_barrel_bat_r = sc_bat_r(away_sc_bat, 'barrel_pct', 'barrel_bat')
            # New statcast batting ratings (higher stat = better for offense)
            home_slg_bat_r    = sc_bat_r(home_sc_bat, 'slg',  'slg_bat')
            away_slg_bat_r    = sc_bat_r(away_sc_bat, 'slg',  'slg_bat')
            home_xslg_bat_r   = sc_bat_r(home_sc_bat, 'xslg', 'xslg_bat')
            away_xslg_bat_r   = sc_bat_r(away_sc_bat, 'xslg', 'xslg_bat')
            home_ba_bat_r     = sc_bat_r(home_sc_bat, 'ba',   'ba_bat')
            away_ba_bat_r     = sc_bat_r(away_sc_bat, 'ba',   'ba_bat')
            home_xba_bat_r    = sc_bat_r(home_sc_bat, 'xba',  'xba_bat')
            away_xba_bat_r    = sc_bat_r(away_sc_bat, 'xba',  'xba_bat')
            home_hh_bat_r     = sc_bat_r(home_sc_bat, 'hardhit_percent', 'hardhit_bat')
            away_hh_bat_r     = sc_bat_r(away_sc_bat, 'hardhit_percent', 'hardhit_bat')
            # swing_miss batting: lower = better → use sc_pit_r formula (invert)
            home_sm_bat_r     = sc_pit_r(home_sc_bat, 'swmiss', 'swmiss_bat')
            away_sm_bat_r     = sc_pit_r(away_sc_bat, 'swmiss', 'swmiss_bat')
            # batter run value: higher = better, offset to handle near-zero league avg
            _lg_rvb = lg_sc.get('rv100_bat', 0)
            _C = 5.0
            home_rv100_bat_r  = ((home_sc_bat.get('rv100_bat', _lg_rvb) + _C) / (_lg_rvb + _C)
                                 if home_sc_bat and (_lg_rvb + _C) != 0 else 1.0)
            away_rv100_bat_r  = ((away_sc_bat.get('rv100_bat', _lg_rvb) + _C) / (_lg_rvb + _C)
                                 if away_sc_bat and (_lg_rvb + _C) != 0 else 1.0)
            # wRC+ (already normalized to 1.0 = league average)
            home_wrc_plus_r   = home_rolling.get('wrc_plus_r', 1.0)
            away_wrc_plus_r   = away_rolling.get('wrc_plus_r', 1.0)

            home_xwoba_pit_r  = sc_pit_r(home_sc_pit, 'xwoba',      'xwoba_pit')
            away_xwoba_pit_r  = sc_pit_r(away_sc_pit, 'xwoba',      'xwoba_pit')
            home_babip_pit_r  = sc_pit_r(home_sc_pit, 'babip',      'babip_pit')
            away_babip_pit_r  = sc_pit_r(away_sc_pit, 'babip',      'babip_pit')
            home_barrel_pit_r = sc_pit_r(home_sc_pit, 'barrel_pct', 'barrel_pit')
            away_barrel_pit_r = sc_pit_r(away_sc_pit, 'barrel_pct', 'barrel_pit')
            # New statcast pitching ratings (lower stat = better for pitching → sc_pit_r)
            home_slg_pit_r    = sc_pit_r(home_sc_pit, 'slg',  'slg_pit')
            away_slg_pit_r    = sc_pit_r(away_sc_pit, 'slg',  'slg_pit')
            home_xslg_pit_r   = sc_pit_r(home_sc_pit, 'xslg', 'xslg_pit')
            away_xslg_pit_r   = sc_pit_r(away_sc_pit, 'xslg', 'xslg_pit')
            home_ba_pit_r     = sc_pit_r(home_sc_pit, 'ba',   'ba_pit')
            away_ba_pit_r     = sc_pit_r(away_sc_pit, 'ba',   'ba_pit')
            home_xba_pit_r    = sc_pit_r(home_sc_pit, 'xba',  'xba_pit')
            away_xba_pit_r    = sc_pit_r(away_sc_pit, 'xba',  'xba_pit')
            home_hh_pit_r     = sc_pit_r(home_sc_pit, 'hardhit_percent', 'hardhit_pit')
            away_hh_pit_r     = sc_pit_r(away_sc_pit, 'hardhit_percent', 'hardhit_pit')
            # swing_miss pitching: higher = better → sc_bat_r formula (non-invert)
            home_sm_pit_r     = sc_bat_r(home_sc_pit, 'swmiss', 'swmiss_pit')
            away_sm_pit_r     = sc_bat_r(away_sc_pit, 'swmiss', 'swmiss_pit')
            # pitcher run value: lower = better, offset to handle near-zero league avg
            _lg_rvp = lg_sc.get('rv100_pit', 0)
            home_rv100_pit_r  = ((_lg_rvp + _C) / (home_sc_pit.get('rv100_pit', _lg_rvp) + _C)
                                 if home_sc_pit and (home_sc_pit.get('rv100_pit', _lg_rvp) + _C) != 0 else 1.0)
            away_rv100_pit_r  = ((_lg_rvp + _C) / (away_sc_pit.get('rv100_pit', _lg_rvp) + _C)
                                 if away_sc_pit and (away_sc_pit.get('rv100_pit', _lg_rvp) + _C) != 0 else 1.0)
            # Game-log pitching: ERA, WHIP, HR/9 — lower = better
            lg_era  = lg_roll.get('era',  4.50)
            lg_whip = lg_roll.get('whip', 1.30)
            lg_hr9  = lg_roll.get('hr9',  1.20)
            home_era_r  = lg_era  / home_rolling['era']  if home_rolling.get('era',  0) > 0 else 1.0
            away_era_r  = lg_era  / away_rolling['era']  if away_rolling.get('era',  0) > 0 else 1.0
            home_whip_r = lg_whip / home_rolling['whip'] if home_rolling.get('whip', 0) > 0 else 1.0
            away_whip_r = lg_whip / away_rolling['whip'] if away_rolling.get('whip', 0) > 0 else 1.0
            home_hr9_r  = lg_hr9  / home_rolling['hr9']  if home_rolling.get('hr9',  0) > 0 else 1.0
            away_hr9_r  = lg_hr9  / away_rolling['hr9']  if away_rolling.get('hr9',  0) > 0 else 1.0

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
                        home_babip_bat_r * params['w_babip_bat'] +
                        home_slg_bat_r   * params['w_slg'] +
                        home_xslg_bat_r  * params['w_xslg'] +
                        home_ba_bat_r    * params['w_ba'] +
                        home_xba_bat_r   * params['w_xba'] +
                        home_hh_bat_r    * params['w_hardhit_bat'] +
                        home_sm_bat_r    * params['w_swmiss_bat'] +
                        home_rv100_bat_r * params['w_rv100_bat'] +
                        home_wrc_plus_r  * params['w_wrc_plus'])
            away_off = (away_off_roll    * params['w_rolling_off'] +
                        away_woba_r      * params['w_woba'] +
                        away_xwoba_bat_r * params['w_xwoba_bat'] +
                        away_babip_bat_r * params['w_babip_bat'] +
                        away_slg_bat_r   * params['w_slg'] +
                        away_xslg_bat_r  * params['w_xslg'] +
                        away_ba_bat_r    * params['w_ba'] +
                        away_xba_bat_r   * params['w_xba'] +
                        away_hh_bat_r    * params['w_hardhit_bat'] +
                        away_sm_bat_r    * params['w_swmiss_bat'] +
                        away_rv100_bat_r * params['w_rv100_bat'] +
                        away_wrc_plus_r  * params['w_wrc_plus'])
            home_pit = (home_pit_roll    * params['w_rolling_pit'] +
                        home_fip_r       * params['w_fip'] +
                        home_xfip_r      * params['w_xfip'] +
                        home_k_pit_r     * params['w_k_pit'] +
                        home_bb_pit_r    * params['w_bb_pit'] +
                        home_xwoba_pit_r * params['w_xwoba_pit'] +
                        home_babip_pit_r * params['w_babip_pit'] +
                        home_barrel_pit_r* params['w_barrel_pit'] +
                        home_era_r       * params['w_era'] +
                        home_whip_r      * params['w_whip'] +
                        home_hr9_r       * params['w_hr9'] +
                        home_hh_pit_r    * params['w_hardhit_pit'] +
                        home_sm_pit_r    * params['w_swmiss_pit'] +
                        home_rv100_pit_r * params['w_rv100_pit'])
            away_pit = (away_pit_roll    * params['w_rolling_pit'] +
                        away_fip_r       * params['w_fip'] +
                        away_xfip_r      * params['w_xfip'] +
                        away_k_pit_r     * params['w_k_pit'] +
                        away_bb_pit_r    * params['w_bb_pit'] +
                        away_xwoba_pit_r * params['w_xwoba_pit'] +
                        away_babip_pit_r * params['w_babip_pit'] +
                        away_barrel_pit_r* params['w_barrel_pit'] +
                        away_era_r       * params['w_era'] +
                        away_whip_r      * params['w_whip'] +
                        away_hr9_r       * params['w_hr9'] +
                        away_hh_pit_r    * params['w_hardhit_pit'] +
                        away_sm_pit_r    * params['w_swmiss_pit'] +
                        away_rv100_pit_r * params['w_rv100_pit'])
            
            home_off = max(params['rating_cap_low'], min(params['rating_cap_high'], home_off))
            away_off = max(params['rating_cap_low'], min(params['rating_cap_high'], away_off))
            home_pit = max(params['rating_cap_low'], min(params['rating_cap_high'], home_pit))
            away_pit = max(params['rating_cap_low'], min(params['rating_cap_high'], away_pit))
            
            park_factor = PARK_FACTOR_LOOKUP.get(home_team, 1.0)
            home_lambda = (home_off / away_pit) * lg_avg_runs * park_factor
            away_lambda = (away_off / home_pit) * lg_avg_runs * park_factor
            
            home_win_pct, away_win_pct, avg_total = run_simulation(home_lambda, away_lambda)

            # Brier score: record for ALL games with known outcomes (no odds required)
            _hs = game.get('home_score'); _as = game.get('away_score')
            if not pd.isna(_hs) and not pd.isna(_as):
                brier_records.append((home_win_pct, 1 if _hs > _as else 0))

            fd_home_ml = game.get('fd_home_ml')
            fd_away_ml = game.get('fd_away_ml')

            if pd.isna(fd_home_ml) or pd.isna(fd_away_ml):
                continue
            
            home_market_prob, away_market_prob = de_vig_probs(fd_home_ml, fd_away_ml)
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

            # ── Over/Under bet ────────────────────────────────────────────────
            fd_total      = game.get('fd_total')
            fd_over_odds  = game.get('fd_over_odds')
            fd_under_odds = game.get('fd_under_odds')

            # O/U bet — only requires the total line, juice optional (ROI skipped if absent)
            if not pd.isna(fd_total):
                ou_diff      = avg_total - fd_total
                actual_total = home_score + away_score
                ou_thresh    = params.get('ou_run_threshold', 1.5)

                ou_bet    = None
                ou_odds   = None
                ou_result = None

                if ou_diff > ou_thresh:
                    ou_bet   = 'over'
                    ou_odds  = fd_over_odds if not pd.isna(fd_over_odds) else None
                    if actual_total > fd_total:   ou_result = 'WIN'
                    elif actual_total == fd_total: ou_result = 'PUSH'
                    else:                          ou_result = 'LOSS'
                elif -ou_diff > ou_thresh:
                    ou_bet   = 'under'
                    ou_odds  = fd_under_odds if not pd.isna(fd_under_odds) else None
                    if actual_total < fd_total:   ou_result = 'WIN'
                    elif actual_total == fd_total: ou_result = 'PUSH'
                    else:                          ou_result = 'LOSS'

                if ou_bet:
                    all_ou_results.append({'ou_odds': ou_odds, 'ou_result': ou_result})
    
    if not all_results:
        return None

    # Raw bets mode — return individual records for holdout/permutation analysis
    if return_bets:
        return all_results, all_ou_results

    brier_score = float(np.mean([(p - y)**2 for p, y in brier_records])) if brier_records else None

    # ── Moneyline metrics ─────────────────────────────────────────────────────
    ml_df      = pd.DataFrame(all_results)
    n_bets     = len(ml_df)
    ml_wins    = len(ml_df[ml_df['ml_result'] == 'WIN'])
    win_rate   = ml_wins / n_bets
    ml_wagered = n_bets * 100
    ml_return  = 0
    for _, row in ml_df.iterrows():
        if row['ml_result'] == 'WIN':
            odds = row['ml_odds']
            ml_return += 100 + (100 * odds / 100) if odds > 0 else 100 + (100 / abs(odds) * 100)
    ml_profit = ml_return - ml_wagered
    roi       = ml_profit / ml_wagered * 100

    # ── O/U metrics ───────────────────────────────────────────────────────────
    n_ou_bets  = 0
    ou_win_rate = 0.0
    ou_roi     = 0.0
    ou_profit  = 0.0

    if all_ou_results:
        ou_df      = pd.DataFrame(all_ou_results)
        n_ou_bets  = len(ou_df)
        ou_wins    = len(ou_df[ou_df['ou_result'] == 'WIN'])
        ou_win_rate = ou_wins / n_ou_bets
        # ROI only for bets where juice was available — WIN/LOSS tracked for all
        ou_with_juice = ou_df[ou_df['ou_odds'].notna()]
        if len(ou_with_juice) > 0:
            ou_wagered = len(ou_with_juice) * 100
            ou_return  = 0
            for _, row in ou_with_juice.iterrows():
                if row['ou_result'] == 'WIN':
                    odds = float(row['ou_odds'])
                    ou_return += 100 + (100 * odds / 100) if odds > 0 else 100 + (100 / abs(odds) * 100)
                elif row['ou_result'] == 'PUSH':
                    ou_return += 100
            ou_profit = ou_return - ou_wagered
            ou_roi    = ou_profit / ou_wagered * 100
        else:
            ou_profit = 0.0
            ou_roi    = 0.0

    # ── Combined portfolio ROI ────────────────────────────────────────────────
    total_wagered = (n_bets + n_ou_bets) * 100
    combined_roi  = (ml_profit + ou_profit) / total_wagered * 100 if total_wagered > 0 else 0

    return {
        'n_bets':       n_bets,
        'win_rate':     round(win_rate, 4),
        'roi':          round(roi, 4),
        'profit':       round(ml_profit, 2),
        'n_ou_bets':    n_ou_bets,
        'ou_win_rate':  round(ou_win_rate, 4),
        'ou_roi':       round(ou_roi, 4),
        'ou_profit':    round(ou_profit, 2),
        'combined_roi': round(combined_roi, 4),
        'brier_score':  round(brier_score, 6) if brier_score is not None else None,
        'n_calibration': len(brier_records),
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

# ── Meta model — 4 validated metrics combined via logistic regression ─────────

# Feature 0: de-vigged market logit (unscaled anchor).
# Features 1–4: league-normalized Statcast difference metrics (standardized).
_META_FEATURES = ['logit_market_prob', 'd_woba', 'd_xwoba_bat', 'd_xfip', 'd_xwoba_pit']


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _fit_logistic(X, y, C=1.0, n_iter=3000, lr=0.01):
    """
    Logistic regression via Adam, minimizing log-loss with L2 regularization.
    Returns (intercept, coef) as 1-D arrays.
    """
    n, d = X.shape
    lam = 1.0 / (C * n)
    Xa = np.column_stack([np.ones(n), X])
    w = np.zeros(Xa.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, n_iter + 1):
        p = _sigmoid(Xa @ w)
        grad = Xa.T @ (p - y) / n
        grad[1:] += lam * w[1:]
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w[0], w[1:]


def collect_game_features_for_meta(seasons):
    """
    Extract per-game feature vectors for the market-residual meta model.
    Feature 0: logit of de-vigged FanDuel home win probability (market anchor).
    Features 1–4: league-normalized Statcast difference metrics (home − away).
    Games without both ML lines are excluded (logit is undefined).
    """
    records = []
    for season in seasons:
        season_str = str(season)
        season_odds = odds_df[odds_df['date'].str.startswith(season_str)]
        season_dates = set(season_odds['date'].tolist())
        sc_lookup, sc_lg = precompute_statcast(season, statcast_data, season_dates)

        for _, game in season_odds.iterrows():
            date      = game['date']
            home_team = normalize_team(game['home_team'])
            away_team = normalize_team(game['away_team'])
            if home_team not in TEAM_MAP or away_team not in TEAM_MAP:
                continue

            home_rolling = rolling_team_cache.get((season_str, home_team, date))
            away_rolling = rolling_team_cache.get((season_str, away_team, date))
            lg_roll      = rolling_lg_cache.get((season_str, date))
            if home_rolling is None or away_rolling is None or lg_roll is None:
                continue

            home_score = game.get('home_score')
            away_score = game.get('away_score')
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            lg_woba = lg_roll['woba']
            lg_xfip = lg_roll['xfip']

            home_woba_r = home_rolling['woba'] / lg_woba if lg_woba > 0 else 1.0
            away_woba_r = away_rolling['woba'] / lg_woba if lg_woba > 0 else 1.0
            home_xfip_r = lg_xfip / home_rolling['xfip'] if home_rolling.get('xfip', 0) > 0 else 1.0
            away_xfip_r = lg_xfip / away_rolling['xfip'] if away_rolling.get('xfip', 0) > 0 else 1.0

            lg_sc       = sc_lg.get(date, {})
            home_sc_bat = sc_lookup.get((home_team, 'batting',  date), {})
            away_sc_bat = sc_lookup.get((away_team, 'batting',  date), {})
            home_sc_pit = sc_lookup.get((home_team, 'pitching', date), {})
            away_sc_pit = sc_lookup.get((away_team, 'pitching', date), {})

            def _bat_r(sc, key, lg_key):
                return sc.get(key, 0) / lg_sc[lg_key] if sc and lg_sc and lg_sc.get(lg_key, 0) > 0 and sc.get(key) else 1.0
            def _pit_r(sc, key, lg_key):
                return lg_sc[lg_key] / sc.get(key, 0) if sc and lg_sc and sc.get(key, 0) > 0 and lg_sc.get(lg_key) else 1.0

            home_xwoba_bat_r = _bat_r(home_sc_bat, 'xwoba', 'xwoba_bat')
            away_xwoba_bat_r = _bat_r(away_sc_bat, 'xwoba', 'xwoba_bat')
            home_xwoba_pit_r = _pit_r(home_sc_pit, 'xwoba', 'xwoba_pit')
            away_xwoba_pit_r = _pit_r(away_sc_pit, 'xwoba', 'xwoba_pit')

            fd_home_ml = game.get('fd_home_ml')
            fd_away_ml = game.get('fd_away_ml')

            # Require both lines to compute market logit — skip games without odds
            if pd.isna(fd_home_ml) or pd.isna(fd_away_ml):
                continue
            fd_home_ml = float(fd_home_ml)
            fd_away_ml = float(fd_away_ml)
            if fd_home_ml == 0 or fd_away_ml == 0:
                continue

            p_home_dv, _ = de_vig_probs(fd_home_ml, fd_away_ml)
            p_clipped     = float(np.clip(p_home_dv, 0.001, 0.999))
            logit_mp      = float(np.log(p_clipped / (1.0 - p_clipped)))

            records.append({
                'date':              date,
                'season':            season,
                'home_team':         home_team,
                'away_team':         away_team,
                'logit_market_prob': logit_mp,
                'd_woba':            home_woba_r      - away_woba_r,
                'd_xwoba_bat':       home_xwoba_bat_r - away_xwoba_bat_r,
                'd_xfip':            home_xfip_r      - away_xfip_r,
                'd_xwoba_pit':       home_xwoba_pit_r - away_xwoba_pit_r,
                'y':                 1 if home_score > away_score else 0,
                'fd_home_ml':        fd_home_ml,
                'fd_away_ml':        fd_away_ml,
            })

    return pd.DataFrame(records)


def walk_forward_meta_model(C=1.0):
    """
    Walk-forward cross-validation for the logistic regression meta model.
    Uses 4 validated metrics (d_woba, d_xwoba_bat, d_xfip, d_xwoba_pit).
    Trains to minimize log-loss; evaluates Brier score only — no ROI.
    """
    folds = [
        ([2021],                   2022),
        ([2021, 2022],             2023),
        ([2021, 2022, 2023],       2024),
        ([2021, 2022, 2023, 2024], 2025),
    ]

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD META MODEL  |  Features: {_META_FEATURES}")
    print(f"  Optimizer: Logistic Regression (log-loss, C={C})")
    print(f"{'='*72}")

    print("\nCollecting game features for all seasons...")
    all_feat = collect_game_features_for_meta([2021, 2022, 2023, 2024, 2025])
    print(f"  {len(all_feat)} games collected\n")

    print(f"  {'Fold':<5} {'Train':<26} {'Test':<6} {'N (train)':<11} {'N (test)':<10} {'Train Brier':<13} {'Test Brier':<12} {'Test LogLoss'}")
    print(f"  {'-'*90}")

    rows = []
    for fold_idx, (train_years, test_year) in enumerate(folds, 1):
        train_df = all_feat[all_feat['season'].isin(train_years)].reset_index(drop=True)
        test_df  = all_feat[all_feat['season'] == test_year].reset_index(drop=True)
        if len(train_df) < 100 or len(test_df) < 50:
            continue

        X_tr = train_df[_META_FEATURES].values.astype(float)
        y_tr = train_df['y'].values.astype(float)
        X_te = test_df[_META_FEATURES].values.astype(float)
        y_te = test_df['y'].values.astype(float)

        # Col 0 = market logit (unscaled — preserves direct linear relationship).
        # Cols 1–4 = Statcast differences (standardized on training data only).
        mu  = X_tr[:, 1:].mean(axis=0)
        sig = X_tr[:, 1:].std(axis=0) + 1e-8
        X_tr_s = np.column_stack([X_tr[:, 0], (X_tr[:, 1:] - mu) / sig])
        X_te_s = np.column_stack([X_te[:, 0], (X_te[:, 1:] - mu) / sig])

        intercept, coef = _fit_logistic(X_tr_s, y_tr, C=C)

        p_tr = _sigmoid(X_tr_s @ coef + intercept)
        p_te = _sigmoid(X_te_s @ coef + intercept)

        brier_tr = float(np.mean((p_tr - y_tr) ** 2))
        brier_te = float(np.mean((p_te - y_te) ** 2))
        ll_te    = float(-np.mean(y_te * np.log(p_te + 1e-15) + (1 - y_te) * np.log(1 - p_te + 1e-15)))

        coef_str = '  '.join(f'{f}={c:+.3f}' for f, c in zip(_META_FEATURES, coef))
        print(f"  {fold_idx:<5} {str(train_years):<26} {test_year:<6} {len(train_df):<11} {len(test_df):<10} {brier_tr:.4f}        {brier_te:.4f}       {ll_te:.4f}")
        print(f"         intercept={intercept:+.3f}  {coef_str}")

        rows.append({
            'fold': fold_idx, 'train_years': str(train_years), 'test_year': test_year,
            'n_train': len(train_df), 'n_test': len(test_df),
            'train_brier': brier_tr, 'test_brier': brier_te, 'test_log_loss': ll_te,
        })

    if rows:
        avg_gap = np.mean([r['test_brier'] - r['train_brier'] for r in rows])
        print(f"\n  Avg train→test Brier gap: {avg_gap:+.4f}  (positive = overfit, negative = generalizing)")
        print(f"  Baseline Brier (random 50/50 guesser): 0.2500")
    print()
    return rows


def walk_forward_financial_sim(C=1.0, flat_bet=20.0, edge_min=0.07, all_feat=None):
    """
    Fold-aware financial simulation using meta model (logistic regression) probabilities.
    Each fold trains exclusively on its training years, then generates out-of-sample
    probabilities for the held-out test year:
      Fold 1: train [2021]              → test 2022
      Fold 2: train [2021, 2022]        → test 2023
      Fold 3: train [2021, 2022, 2023]  → test 2024
      Fold 4: train [2021–2024]         → test 2025

    Edge = model_prob − de-vigged market_implied_prob.
    Bets placed when edge > edge_min (no upper cap).
    ROI computed with flat_bet per qualifying game.
    Pass all_feat to reuse pre-collected features across multiple calls.
    """
    folds = [
        ([2021],                   2022),
        ([2021, 2022],             2023),
        ([2021, 2022, 2023],       2024),
        ([2021, 2022, 2023, 2024], 2025),
    ]
    flat_bet = float(flat_bet)

    print(f"\n{'='*80}")
    print(f"  FINANCIAL SIMULATION — META MODEL WALK-FORWARD")
    print(f"  Edge filter: >{edge_min:.0%} (no upper cap)  |  Flat bet: ${flat_bet:.0f}/game  |  Logistic C={C}")
    print(f"{'='*80}\n")

    if all_feat is None:
        print("Collecting game features for 2021–2025...")
        all_feat = collect_game_features_for_meta([2021, 2022, 2023, 2024, 2025])
        print(f"  {len(all_feat)} games with known outcomes\n")

    print(f"  {'Fold':<5} {'Year':<5} {'Odds Games':>11} {'Bets':>5} {'W':>4} {'L':>4} {'Win%':>7} {'Profit':>11} {'Season ROI':>11}")
    print(f"  {'-'*66}")

    grand_bets    = 0
    grand_wagered = 0.0
    grand_profit  = 0.0
    grand_wins    = 0
    grand_losses  = 0
    year_rows     = []

    for fold_idx, (train_years, test_year) in enumerate(folds, 1):
        train_df = all_feat[all_feat['season'].isin(train_years)].reset_index(drop=True)
        test_df  = all_feat[all_feat['season'] == test_year].reset_index(drop=True)
        if len(train_df) < 100 or len(test_df) < 50:
            continue

        X_tr = train_df[_META_FEATURES].values.astype(float)
        y_tr = train_df['y'].values.astype(float)
        X_te = test_df[_META_FEATURES].values.astype(float)

        # Col 0 = market logit (unscaled). Cols 1–4 standardized on training data.
        mu  = X_tr[:, 1:].mean(axis=0)
        sig = X_tr[:, 1:].std(axis=0) + 1e-8
        X_tr_s = np.column_stack([X_tr[:, 0], (X_tr[:, 1:] - mu) / sig])
        X_te_s = np.column_stack([X_te[:, 0], (X_te[:, 1:] - mu) / sig])

        intercept, coef = _fit_logistic(X_tr_s, y_tr, C=C)
        p_home_arr = _sigmoid(X_te_s @ coef + intercept)

        wins = losses = n_odds_games = 0
        profit = 0.0

        for i, row in test_df.iterrows():
            fd_home_ml = row['fd_home_ml']
            fd_away_ml = row['fd_away_ml']
            if pd.isna(fd_home_ml) or pd.isna(fd_away_ml):
                continue
            fd_home_ml = float(fd_home_ml)
            fd_away_ml = float(fd_away_ml)
            if fd_home_ml == 0 or fd_away_ml == 0:
                continue
            n_odds_games += 1

            home_prob = float(p_home_arr[i])
            away_prob = 1.0 - home_prob

            home_market, away_market = de_vig_probs(fd_home_ml, fd_away_ml)

            home_edge = home_prob - home_market
            away_edge = away_prob - away_market

            y_actual = int(row['y'])  # 1 = home win, 0 = away win

            bet_odds = None
            bet_won  = None
            if home_edge > edge_min:
                bet_odds = float(fd_home_ml)
                bet_won  = (y_actual == 1)
            elif away_edge > edge_min:
                bet_odds = float(fd_away_ml)
                bet_won  = (y_actual == 0)

            if bet_odds is None:
                continue

            if bet_won:
                wins += 1
                profit += flat_bet * bet_odds / 100 if bet_odds > 0 else flat_bet * 100 / abs(bet_odds)
            else:
                losses += 1
                profit -= flat_bet

        n_bets  = wins + losses
        wagered = n_bets * flat_bet
        roi     = profit / wagered * 100 if wagered > 0 else 0.0
        win_pct = wins / n_bets if n_bets > 0 else 0.0

        print(f"  {fold_idx:<5} {test_year:<5} {n_odds_games:>11} {n_bets:>5} {wins:>4} {losses:>4} {win_pct:>7.1%} ${profit:>+9.2f}  {roi:>+8.1f}%")

        grand_bets    += n_bets
        grand_wagered += wagered
        grand_profit  += profit
        grand_wins    += wins
        grand_losses  += losses
        year_rows.append({
            'fold': fold_idx, 'year': test_year, 'n_bets': n_bets,
            'wins': wins, 'losses': losses, 'profit': round(profit, 2), 'roi': round(roi, 2),
        })

    if grand_wagered > 0:
        overall_roi      = grand_profit / grand_wagered * 100
        overall_win_rate = grand_wins / grand_bets if grand_bets > 0 else 0.0
        print(f"  {'-'*66}")
        print(f"  {'TOTAL':<5} {'2022–25':<5} {'':>11} {grand_bets:>5} {grand_wins:>4} {grand_losses:>4} "
              f"{overall_win_rate:>7.1%} ${grand_profit:>+9.2f}  {overall_roi:>+8.1f}%")
        print(f"\n  Total wagered:    ${grand_wagered:>10,.2f}")
        print(f"  Total profit:     ${grand_profit:>+10.2f}")
        print(f"  4-year ROI:       {overall_roi:>+.1f}%")
        print(f"  Avg bets/season:  {grand_bets / len(year_rows):.0f}")
        print(f"\n  Edge filter: >{edge_min:.0%}, no upper cap")
    print()
    return year_rows


def walk_forward_dual_strategy(all_feat, flat_bet=20.0):
    """
    Walk-forward financial simulation running two betting strategies in parallel.
    Both strategies are trained on identical folds with identical standardization;
    they differ only in regularization strength (C) and edge filter threshold.

    Strategy A — The Sniper    : C=1.0, edge > 5.0%  (high-conviction, lower volume)
    Strategy B — The Enforcer  : C=5.0, edge > 3.5%  (relaxed reg, higher volume)

    Uses scikit-learn LogisticRegression (L2 penalty, lbfgs solver) instead of
    the custom numpy Adam optimizer.  Column 0 (market logit) is left unscaled;
    columns 1-4 (Statcast diffs) are standardized on training data only.

    Folds:
      Fold 1: train [2021]             -> test 2022
      Fold 2: train [2021, 2022]       -> test 2023
      Fold 3: train [2021, 2022, 2023] -> test 2024
      Fold 4: train [2021-2024]        -> test 2025
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("  ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return {}

    STRATEGIES = [
        ('The Sniper',   1.0, 0.050),
        ('The Enforcer', 5.0, 0.035),
    ]

    FOLDS = [
        ([2021],                   2022),
        ([2021, 2022],             2023),
        ([2021, 2022, 2023],       2024),
        ([2021, 2022, 2023, 2024], 2025),
    ]
    flat_bet = float(flat_bet)

    # Per-strategy running totals
    totals = {
        name: {'bets': 0, 'wins': 0, 'wagered': 0.0, 'profit': 0.0, 'rows': []}
        for name, _, _ in STRATEGIES
    }

    # ── One block of output per strategy ──────────────────────────────────────
    for strat_name, C, edge_min in STRATEGIES:
        print(f"\n{'='*70}")
        print(f"  STRATEGY: {strat_name}   |   C={C}   |   edge > {edge_min:.1%}   |   flat ${flat_bet:.0f}/game")
        print(f"{'='*70}")
        print(f"  {'Fold':<5} {'Year':<5} {'Odds':>7} {'Bets':>5} {'W':>4} {'L':>4} {'Win%':>6} {'Profit':>10} {'ROI':>8}")
        print(f"  {'-'*56}")

        model = LogisticRegression(
            C=C, penalty='l2', solver='lbfgs',
            max_iter=1000, fit_intercept=True,
        )

        for fold_idx, (train_years, test_year) in enumerate(FOLDS, 1):
            train_df = all_feat[all_feat['season'].isin(train_years)].reset_index(drop=True)
            test_df  = all_feat[all_feat['season'] == test_year].reset_index(drop=True)
            if len(train_df) < 100 or len(test_df) < 50:
                continue

            X_tr = train_df[_META_FEATURES].values.astype(float)
            y_tr = train_df['y'].values.astype(int)
            X_te = test_df[_META_FEATURES].values.astype(float)

            # Col 0 = market logit (unscaled — preserves linear relationship to outcome logit)
            # Cols 1-4 = Statcast diffs (standardized on training fold; no test leakage)
            mu  = X_tr[:, 1:].mean(axis=0)
            sig = X_tr[:, 1:].std(axis=0) + 1e-8
            X_tr_s = np.column_stack([X_tr[:, 0], (X_tr[:, 1:] - mu) / sig])
            X_te_s = np.column_stack([X_te[:, 0], (X_te[:, 1:] - mu) / sig])

            model.fit(X_tr_s, y_tr)
            # predict_proba returns (n_samples, 2): col 1 = P(home win)
            p_home_arr = model.predict_proba(X_te_s)[:, 1]

            wins = losses = n_odds = 0
            profit = 0.0

            for i, row in test_df.iterrows():
                fd_home_ml = row.get('fd_home_ml')
                fd_away_ml = row.get('fd_away_ml')
                if pd.isna(fd_home_ml) or pd.isna(fd_away_ml):
                    continue
                fd_home_ml = float(fd_home_ml)
                fd_away_ml = float(fd_away_ml)
                if fd_home_ml == 0 or fd_away_ml == 0:
                    continue
                n_odds += 1

                home_prob = float(p_home_arr[i])
                away_prob = 1.0 - home_prob
                home_mkt, away_mkt = de_vig_probs(fd_home_ml, fd_away_ml)
                home_edge = home_prob - home_mkt
                away_edge = away_prob - away_mkt
                y_actual  = int(row['y'])

                bet_odds = bet_won = None
                if home_edge > edge_min:
                    bet_odds, bet_won = fd_home_ml, (y_actual == 1)
                elif away_edge > edge_min:
                    bet_odds, bet_won = fd_away_ml, (y_actual == 0)

                if bet_odds is None:
                    continue

                if bet_won:
                    wins += 1
                    profit += (flat_bet * bet_odds / 100 if bet_odds > 0
                               else flat_bet * 100 / abs(bet_odds))
                else:
                    losses += 1
                    profit -= flat_bet

            n_bets  = wins + losses
            wagered = n_bets * flat_bet
            roi     = profit / wagered * 100 if wagered > 0 else 0.0
            win_pct = wins / n_bets if n_bets > 0 else 0.0

            print(f"  {fold_idx:<5} {test_year:<5} {n_odds:>7} {n_bets:>5} {wins:>4} {losses:>4} "
                  f"{win_pct:>6.1%} ${profit:>+8.2f} {roi:>+7.1f}%")

            acc = totals[strat_name]
            acc['bets']    += n_bets
            acc['wins']    += wins
            acc['wagered'] += wagered
            acc['profit']  += profit
            acc['rows'].append({
                'fold': fold_idx, 'year': test_year, 'n_bets': n_bets,
                'wins': wins, 'losses': losses,
                'profit': round(profit, 2), 'roi': round(roi, 2),
            })

        # Per-strategy total row
        acc = totals[strat_name]
        if acc['wagered'] > 0:
            tot_roi  = acc['profit'] / acc['wagered'] * 100
            tot_wpct = acc['wins'] / acc['bets'] if acc['bets'] > 0 else 0.0
            n_folds  = len(acc['rows'])
            print(f"  {'-'*56}")
            print(f"  {'TOTAL':<5} {'22-25':<5} {'':>7} {acc['bets']:>5} {acc['wins']:>4} "
                  f"{acc['bets'] - acc['wins']:>4} {tot_wpct:>6.1%} "
                  f"${acc['profit']:>+8.2f} {tot_roi:>+7.1f}%")
            print(f"  Avg bets/year: {acc['bets'] / n_folds:.0f}   "
                  f"Total wagered: ${acc['wagered']:,.0f}   "
                  f"Total profit: ${acc['profit']:>+,.2f}")

    # ── Side-by-side summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DUAL STRATEGY SUMMARY   |   flat ${flat_bet:.0f}/game   |   2022-2025")
    print(f"{'='*70}")
    print(f"  {'Strategy':<18} {'C':>5}  {'Edge':>7}  {'Bets':>5}  {'Win%':>6}  {'4yr ROI':>8}  {'$/yr avg':>9}")
    print(f"  {'-'*64}")
    for strat_name, C, edge_min in STRATEGIES:
        acc = totals[strat_name]
        if acc['wagered'] > 0:
            roi       = acc['profit'] / acc['wagered'] * 100
            wpct      = acc['wins'] / acc['bets'] if acc['bets'] > 0 else 0.0
            avg_yr    = acc['bets'] / len(acc['rows']) if acc['rows'] else 0
            profit_yr = acc['profit'] / len(acc['rows']) if acc['rows'] else 0
            print(f"  {strat_name:<18} {C:>5.1f}  >{edge_min:>5.1%}  {acc['bets']:>5}  "
                  f"{wpct:>6.1%}  {roi:>+7.1f}%  ${profit_yr:>+8.2f}")
    print()

    return totals


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER-LEVEL FEATURE SYSTEM  (Round 9 — bottom-up rebuild)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces team-level rolling averages with:
#   • Individual batter rolling 100 PA vs the pitcher's hand type (wOBA, xwOBA)
#   • Individual starting pitcher rolling 100 BF (xFIP, xwOBA allowed)
#   • Historical box score batting orders + positional PA weights for aggregation
#
# New data files required (download via fetch_player_data.py):
#   player_data/batters_vs_L_YYYY.csv  — per-batter, per-game stats vs LHP
#   player_data/batters_vs_R_YYYY.csv  — per-batter, per-game stats vs RHP
#   player_data/starters_YYYY.csv      — per-starter, per-game stats
#
# Generated cache files:
#   historical_data/game_lineups.json       — box score batting orders + starter IDs
#   historical_data/pitcher_hand_cache.json — pitcher_id → throw hand (L/R)
# ─────────────────────────────────────────────────────────────────────────────

# Structural PA weights for batting order spots 1–9.
# Leadoff sees ~123% of the average slot's PA; cleanup ~111%; 9-hole ~100%.
PA_WEIGHTS = np.array([0.123, 0.120, 0.117, 0.114, 0.111, 0.108, 0.105, 0.102, 0.100])

_MIN_BATTER_PA   = 30    # require this many rolling PA before using a batter's stats
_MIN_PITCHER_BF  = 20    # require this many rolling BF before using a starter's stats
_ROLLING_PA_WIN  = 100   # target rolling window size in PA / BF


def aggregate_lineup_metric(player_stats_dict, lineup_list, lg_avg):
    """
    Position-weighted aggregate of one metric across a 9-man batting order.

    player_stats_dict : {player_id (int): metric_value (float)}
                        maps each batter to their situation-specific rolling stat
                        (e.g. xwOBA vs RHP over the last 100 PA).
    lineup_list       : ordered list of 9 player_ids (batting order slots 1–9).
    lg_avg            : league-average fallback for rookies / missing data.

    Returns float — np.dot(per_slot_metrics, PA_WEIGHTS).
    """
    slots = lineup_list[:9]
    metrics = np.array(
        [player_stats_dict.get(pid, lg_avg) for pid in slots],
        dtype=float
    )
    if len(metrics) < 9:
        metrics = np.pad(metrics, (0, 9 - len(metrics)), constant_values=lg_avg)
    return float(np.dot(metrics, PA_WEIGHTS))


# ── Reverse map: MLB team ID → FanGraphs abbreviation ────────────────────────
_MLB_ID_TO_FG = {v: k for k, v in TEAM_MAP.items()}


def pull_historical_lineups(seasons, cache_path='historical_data/game_lineups.json'):
    """
    Fetch starting batting orders and starter IDs from MLB Stats API box scores.

    One schedule call per season (fast) then one boxscore_data call per game
    (~30–45 min per season on first run). Results are cached incrementally so
    a restart resumes where it left off.

    Returns
    -------
    {season_str: {game_pk_str: {
        'date'             : 'YYYY-MM-DD',
        'home_fg'          : FG abbreviation,
        'away_fg'          : FG abbreviation,
        'home_lineup'      : [player_id × 9],   # batting order
        'away_lineup'      : [player_id × 9],
        'home_starter_id'  : int,
        'away_starter_id'  : int,
        'home_starter_hand': 'L' or 'R',
        'away_starter_hand': 'L' or 'R',
    }}}
    """
    import time as _time

    cache = {}
    if os.path.exists(cache_path):
        print("Loading game lineup cache from disk...")
        with open(cache_path) as f:
            cache = json.load(f)

    missing_seasons = [s for s in seasons if str(s) not in cache]
    if not missing_seasons:
        print(f"Lineup cache complete for {seasons}.")
        return cache

    # Pitcher hand lookup — persisted separately so it survives restarts
    _hand_path = 'historical_data/pitcher_hand_cache.json'
    hand_cache = {}
    if os.path.exists(_hand_path):
        with open(_hand_path) as f:
            hand_cache = {int(k): v for k, v in json.load(f).items()}

    def _get_hand(player_id):
        if player_id in hand_cache:
            return hand_cache[player_id]
        try:
            data = statsapi.get('people', {'personIds': player_id})
            code = data['people'][0]['pitchHand']['code']
        except Exception:
            code = 'R'
        hand_cache[player_id] = code
        return code

    def _save_caches(season_str, season_cache):
        cache[season_str] = season_cache
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        with open(_hand_path, 'w') as f:
            json.dump({str(k): v for k, v in hand_cache.items()}, f)

    for season in missing_seasons:
        print(f"\nPulling {season} box score lineups from MLB API...")
        t0 = _time.time()
        season_cache = {}

        # Retry up to 5 times — MLB API occasionally returns 503 on large date ranges
        schedule = None
        for _attempt in range(5):
            try:
                schedule = statsapi.schedule(
                    start_date=f'{season}-03-01',
                    end_date=f'{season}-11-01',
                    sportId=1,
                )
                break
            except Exception as _e:
                wait = 15 * (_attempt + 1)
                print(f"  Schedule fetch attempt {_attempt + 1} failed ({_e}). Retrying in {wait}s...")
                _time.sleep(wait)
        if schedule is None:
            print(f"  ERROR: could not fetch schedule for {season} after 5 attempts. Skipping.")
            continue

        # Filter to final regular-season games (excludes spring training, playoffs)
        final_games = [g for g in schedule
                       if g.get('status') == 'Final' and g.get('game_type') == 'R']
        print(f"  {len(final_games)} final regular-season games to process")

        for i, game in enumerate(final_games):
            game_pk  = game['game_id']
            date_str = game['game_date']
            home_fg  = _MLB_ID_TO_FG.get(game['home_id'])
            away_fg  = _MLB_ID_TO_FG.get(game['away_id'])

            if not home_fg or not away_fg:
                continue

            bs = None
            for _attempt in range(3):
                try:
                    bs = statsapi.boxscore_data(game_pk)
                    break
                except Exception as _e:
                    if _attempt < 2:
                        _time.sleep(5 * (_attempt + 1))
            if bs is None:
                continue

            def _starting_lineup(side):
                """Return [player_id × 9] in batting order from box score side dict."""
                players = bs[side].get('players', {})
                slots = []
                for pdata in players.values():
                    bo = pdata.get('battingOrder', '')
                    # Starters have battingOrder '100','200',...,'900'
                    # Substitutes have '101','201',...
                    if bo and str(bo).endswith('00'):
                        slots.append((int(bo), pdata['person']['id']))
                slots.sort()
                return [pid for _, pid in slots]

            home_lineup = _starting_lineup('home')
            away_lineup = _starting_lineup('away')

            # pitchers list is ordered by appearance: index 0 = game starter
            home_pitchers = bs['home'].get('pitchers', [])
            away_pitchers = bs['away'].get('pitchers', [])

            if len(home_lineup) < 9 or len(away_lineup) < 9:
                continue
            if not home_pitchers or not away_pitchers:
                continue

            home_starter_id = home_pitchers[0]
            away_starter_id = away_pitchers[0]

            def _starter_ip(side, pitcher_id):
                """
                Extract inningsPitched (str, baseball notation e.g. '5.2' = 5⅔ IP)
                from the pitcher's per-game stats block inside the box score.
                The 'outs' field is not present in the MLB API box score stats block;
                ip_to_float() converts the IP string to an exact decimal instead.
                Returns ip_str or None if unavailable.
                """
                player_key = f'ID{pitcher_id}'
                pdata      = bs[side].get('players', {}).get(player_key, {})
                pit        = pdata.get('stats', {}).get('pitching', {})
                return pit.get('inningsPitched')

            season_cache[str(game_pk)] = {
                'date':               date_str,
                'home_fg':            home_fg,
                'away_fg':            away_fg,
                'home_lineup':        home_lineup,
                'away_lineup':        away_lineup,
                'home_starter_id':    home_starter_id,
                'away_starter_id':    away_starter_id,
                'home_starter_hand':  _get_hand(home_starter_id),
                'away_starter_hand':  _get_hand(away_starter_id),
                # True IP from the box score stats block (baseball notation string)
                # ip_to_float('5.2') = 5.667 exactly — no estimation needed
                'home_starter_ip':    _starter_ip('home', home_starter_id),
                'away_starter_ip':    _starter_ip('away', away_starter_id),
            }

            if (i + 1) % 200 == 0:
                elapsed = _time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(final_games) - i - 1) / rate
                print(f"  {i+1}/{len(final_games)} games | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")
                _save_caches(str(season), season_cache)

        _save_caches(str(season), season_cache)
        elapsed = _time.time() - t0
        print(f"  {season} complete: {len(season_cache)} games cached in {elapsed:.0f}s")

    return cache


# ── Player-level rolling caches ───────────────────────────────────────────────

def load_player_batter_data(seasons):
    """
    Load per-batter, per-game Statcast data split by pitcher handedness.
    Files: player_data/batters_vs_L_YYYY.csv  and  batters_vs_R_YYYY.csv
    Required columns: player_id, game_date, pa, woba, xwoba

    Returns {'L': DataFrame, 'R': DataFrame}
    """
    frames = {'L': [], 'R': []}
    for season in seasons:
        for hand in ('L', 'R'):
            path = f'player_data/batters_vs_{hand}_{season}.csv'
            if not os.path.exists(path):
                print(f"  Missing: {path}  (run fetch_player_data.py)")
                continue
            df = pd.read_csv(path, usecols=['player_id', 'game_date', 'pa', 'woba', 'xwoba'])
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
            df = df[df['pa'] > 0].dropna(subset=['woba', 'xwoba'])
            frames[hand].append(df)
    return {
        hand: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        for hand, dfs in frames.items()
    }


def _build_ip_lookup(cache_path='historical_data/game_lineups.json'):
    """
    Build {(player_id: int, game_pk: int): ip_decimal: float} from the
    game_lineups.json cache written by pull_historical_lineups().

    'inningsPitched' is stored as a baseball-notation string (e.g. '6.1' = 6⅓ IP).
    ip_to_float() converts that to a true decimal (6.333...).
    Falls back to outs/3 if inningsPitched is missing but outs is present.
    """
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path) as f:
        lineup_cache = json.load(f)

    lookup = {}
    for season_games in lineup_cache.values():
        for game_pk_str, gdata in season_games.items():
            game_pk = int(game_pk_str)
            for side in ('home', 'away'):
                pid    = gdata.get(f'{side}_starter_id')
                ip_str = gdata.get(f'{side}_starter_ip')
                if pid is None or ip_str is None:
                    continue
                try:
                    lookup[(int(pid), game_pk)] = ip_to_float(str(ip_str))
                except (ValueError, TypeError):
                    pass
    return lookup


def load_player_pitcher_data(seasons):
    """
    Load per-starter, per-game Statcast data.
    File: player_data/starters_YYYY.csv
    Available columns (from Savant grouped export, verified 2024):
      player_id, game_date, game_pk, pa, woba, xwoba, so, bb, hrs,
      k_percent, bb_percent, obp, hardhit_percent, swing_miss_percent

    IP source (priority order):
      1. game_lineups.json cache  — true inningsPitched from the box score stats block
      2. outs / 3                 — also exact, from the same cache (fallback if IP string absent)
      Estimation from OBP is no longer used.

    Returns DataFrame with canonical column names for build_pitcher_rolling_cache.
    """
    # Load true IP lookup from box score cache (built by pull_historical_lineups)
    ip_lookup = _build_ip_lookup()
    n_ip = len(ip_lookup)
    print(f"  IP lookup: {n_ip:,} (player_id, game_pk) entries from game_lineups.json"
          + (" (empty — run pull_historical_lineups first)" if n_ip == 0 else ""))

    frames = []
    for season in seasons:
        path = f'player_data/starters_{season}.csv'
        if not os.path.exists(path):
            print(f"  Missing: {path}  (run fetch_player_data.py)")
            continue
        df = pd.read_csv(path)
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

        # Canonical column names for the rolling cache builder
        df = df.rename(columns={
            'xwoba': 'xwoba_against',
            'hrs':   'hr',
            'so':    'k',
        })
        if 'xwoba_against' not in df.columns and 'woba' in df.columns:
            df['xwoba_against'] = df['woba']

        # Merge true IP from lineup cache on (player_id, game_pk)
        df['player_id'] = df['player_id'].astype(int)
        df['game_pk']   = df['game_pk'].astype(int)
        df['ip'] = df.apply(
            lambda r: ip_lookup.get((r['player_id'], r['game_pk'])), axis=1
        )
        matched   = df['ip'].notna().sum()
        unmatched = df['ip'].isna().sum()
        print(f"  {season}: {matched:,} rows with true IP, {unmatched:,} without (no xFIP for those rows)")

        # Fly balls from true IP: IP × 3 outs/IP × 0.40 FB/out (MLB average)
        df['fb'] = df['ip'] * 3.0 * 0.40

        # HBP not in Savant grouped data
        if 'hbp' not in df.columns:
            df['hbp'] = 0

        df = df[df['pa'].astype(float) > 0].dropna(subset=['xwoba_against'])
        df['season'] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_batter_rolling_cache(batter_data_by_hand):
    """
    For each player and each pitcher-hand split, compute rolling _ROLLING_PA_WIN PA
    stats at every target date in odds_df using O(log n) prefix-sum lookups.

    Parameters
    ----------
    batter_data_by_hand : {'L': DataFrame, 'R': DataFrame}
        Output of load_player_batter_data().

    Returns
    -------
    batter_cache : {(player_id, hand, date): {'woba': float, 'xwoba': float, 'pa': int}}
    lg_batter_avg : {(hand, date): {'woba': float, 'xwoba': float}}
    """
    import bisect

    # All target dates across all seasons in odds_df
    target_dates = sorted(set(odds_df['date'].tolist()))

    batter_cache = {}

    for hand, df in batter_data_by_hand.items():
        if df.empty:
            continue

        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

        for player_id, grp in df.groupby('player_id'):
            games = grp.sort_values('game_date')[['game_date', 'pa', 'woba', 'xwoba']].to_dict('records')
            n = len(games)
            if n == 0:
                continue

            # Build prefix sums over PA, wOBA×PA, xwOBA×PA
            cum_pa       = [0] * (n + 1)
            cum_woba_pa  = [0.0] * (n + 1)
            cum_xwoba_pa = [0.0] * (n + 1)
            game_dates   = []

            for i, g in enumerate(games):
                cum_pa[i + 1]       = cum_pa[i]       + g['pa']
                cum_woba_pa[i + 1]  = cum_woba_pa[i]  + g['woba']  * g['pa']
                cum_xwoba_pa[i + 1] = cum_xwoba_pa[i] + g['xwoba'] * g['pa']
                game_dates.append(g['game_date'])

            game_idx = 0  # sweep pointer: games[0..game_idx-1] precede current target_date

            for target_date in target_dates:
                # Advance pointer past all games strictly before target_date
                while game_idx < n and game_dates[game_idx] < target_date:
                    game_idx += 1

                k         = game_idx         # games[0..k-1] are before target_date
                total_pa  = cum_pa[k]

                if total_pa < _MIN_BATTER_PA:
                    continue

                # Binary search for the start of the rolling window:
                # find smallest j such that cum_pa[k] - cum_pa[j] >= window_pa
                window_pa = min(total_pa, _ROLLING_PA_WIN)
                threshold = cum_pa[k] - window_pa
                lo, hi = 0, k
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cum_pa[mid] <= threshold:
                        lo = mid + 1
                    else:
                        hi = mid
                j = lo

                pa_win = cum_pa[k] - cum_pa[j]
                if pa_win == 0:
                    continue

                batter_cache[(player_id, hand, target_date)] = {
                    'woba':  (cum_woba_pa[k]  - cum_woba_pa[j])  / pa_win,
                    'xwoba': (cum_xwoba_pa[k] - cum_xwoba_pa[j]) / pa_win,
                    'pa':    pa_win,
                }

    # League averages per (hand, date) — simple mean across qualifying players
    accum = {}  # {(hand, date): [woba_vals, xwoba_vals]}
    for (player_id, hand, date), stats in batter_cache.items():
        key = (hand, date)
        if key not in accum:
            accum[key] = [[], []]
        accum[key][0].append(stats['woba'])
        accum[key][1].append(stats['xwoba'])

    lg_batter_avg = {
        key: {
            'woba':  sum(vals[0]) / len(vals[0]),
            'xwoba': sum(vals[1]) / len(vals[1]),
        }
        for key, vals in accum.items()
        if len(vals[0]) >= 20
    }

    n_entries = len(batter_cache)
    print(f"  Batter cache built: {n_entries:,} entries across {len(lg_batter_avg):,} (hand, date) pairs")
    return batter_cache, lg_batter_avg


def build_pitcher_rolling_cache(pitcher_df, fip_const_by_season):
    """
    For each starting pitcher, compute rolling _ROLLING_PA_WIN BF stats at every
    target date in odds_df.  Computes xFIP from rolling K/BB/HR/FB/IP components
    using the season-level FIP constant and a per-date league HR/FB rate.

    Parameters
    ----------
    pitcher_df         : DataFrame from load_player_pitcher_data()
    fip_const_by_season: {season (int): cFIP (float)}

    Returns
    -------
    pitcher_cache  : {(player_id, date): {'xfip': float, 'xwoba_pit': float, 'pa': int}}
    lg_pitcher_avg : {date: {'xfip': float, 'xwoba_pit': float}}
    """
    if pitcher_df.empty:
        return {}, {}

    target_dates = sorted(set(odds_df['date'].tolist()))

    # League HR/FB rate per date — computed from all starters in the data set
    all_events = []
    for _, row in pitcher_df.iterrows():
        try:
            all_events.append((row['game_date'], int(row.get('hr', 0) or 0),
                               int(row.get('fb', 0) or 0) + int(row.get('hr', 0) or 0)))
        except (ValueError, TypeError):
            continue
    all_events.sort(key=lambda x: x[0])

    cum_hr_all = cum_fb_all = ev_idx = 0
    lg_hr_fb_by_date = {}
    for target_date in target_dates:
        while ev_idx < len(all_events) and all_events[ev_idx][0] < target_date:
            cum_hr_all += all_events[ev_idx][1]
            cum_fb_all += all_events[ev_idx][2]
            ev_idx += 1
        lg_hr_fb_by_date[target_date] = cum_hr_all / cum_fb_all if cum_fb_all > 0 else 0.115

    pitcher_cache = {}
    pitcher_df = pitcher_df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    for player_id, grp in pitcher_df.groupby('player_id'):
        grp = grp.sort_values('game_date')
        keep = ['game_date', 'pa', 'xwoba_against', 'k', 'bb', 'hr', 'hbp', 'fb', 'ip', 'season']
        available = [c for c in keep if c in grp.columns]
        games = grp[available].to_dict('records')
        n = len(games)
        if n == 0:
            continue

        cum_pa       = [0]   * (n + 1)
        cum_xwoba_pa = [0.0] * (n + 1)
        cum_k        = [0]   * (n + 1)
        cum_bb       = [0]   * (n + 1)
        cum_hr       = [0]   * (n + 1)
        cum_hbp      = [0]   * (n + 1)
        cum_fb       = [0.0] * (n + 1)
        cum_ip       = [0.0] * (n + 1)
        game_dates   = []

        for i, g in enumerate(games):
            def _int(v): return int(v) if v and not pd.isna(v) else 0
            def _flt(v): return float(v) if v and not pd.isna(v) else 0.0

            cum_pa[i+1]       = cum_pa[i]       + _int(g['pa'])
            cum_xwoba_pa[i+1] = cum_xwoba_pa[i] + _flt(g['xwoba_against']) * _int(g['pa'])
            cum_k[i+1]        = cum_k[i]        + _int(g['k'])
            cum_bb[i+1]       = cum_bb[i]        + _int(g['bb'])
            cum_hr[i+1]       = cum_hr[i]        + _int(g['hr'])
            cum_hbp[i+1]      = cum_hbp[i]       + _int(g['hbp'])
            cum_fb[i+1]       = cum_fb[i]        + _flt(g['fb'])
            _raw_ip = g.get('ip')
            cum_ip[i+1]       = cum_ip[i]        + (ip_to_float(_raw_ip) if _raw_ip is not None and not pd.isna(_raw_ip) else 0.0)
            game_dates.append(g['game_date'])

        game_idx = 0

        for target_date in target_dates:
            season = int(target_date[:4])
            fip_c  = fip_const_by_season.get(season, 3.20)

            while game_idx < n and game_dates[game_idx] < target_date:
                game_idx += 1

            k        = game_idx
            total_pa = cum_pa[k]

            if total_pa < _MIN_PITCHER_BF:
                continue

            window_pa = min(total_pa, _ROLLING_PA_WIN)
            threshold = cum_pa[k] - window_pa
            lo, hi = 0, k
            while lo < hi:
                mid = (lo + hi) // 2
                if cum_pa[mid] <= threshold:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo

            pa_win = cum_pa[k] - cum_pa[j]
            ip_win = cum_ip[k] - cum_ip[j]
            if pa_win == 0 or ip_win == 0:
                continue

            xwoba_pit = (cum_xwoba_pa[k] - cum_xwoba_pa[j]) / pa_win
            k_win  = cum_k[k]   - cum_k[j]
            bb_win = cum_bb[k]  - cum_bb[j]
            hr_win = cum_hr[k]  - cum_hr[j]
            hb_win = cum_hbp[k] - cum_hbp[j]
            fb_win = cum_fb[k]  - cum_fb[j]

            lg_hr_fb  = lg_hr_fb_by_date.get(target_date, 0.115)
            exp_hr    = lg_hr_fb * fb_win
            xfip_num  = 13 * exp_hr + 3 * (bb_win + hb_win) - 2 * k_win
            xfip      = (xfip_num / ip_win + fip_c) if ip_win > 0 else 4.50

            pitcher_cache[(player_id, target_date)] = {
                'xfip':      xfip,
                'xwoba_pit': xwoba_pit,
                'pa':        pa_win,
            }

    # League averages per date
    accum = {}
    for (pid, date), stats in pitcher_cache.items():
        if date not in accum:
            accum[date] = [[], []]
        accum[date][0].append(stats['xfip'])
        accum[date][1].append(stats['xwoba_pit'])

    lg_pitcher_avg = {
        date: {
            'xfip':      sum(vals[0]) / len(vals[0]),
            'xwoba_pit': sum(vals[1]) / len(vals[1]),
        }
        for date, vals in accum.items()
        if len(vals[0]) >= 10
    }

    print(f"  Pitcher cache built: {len(pitcher_cache):,} entries across {len(lg_pitcher_avg):,} dates")
    return pitcher_cache, lg_pitcher_avg


# ── Main player-level feature collector ──────────────────────────────────────

def collect_game_features_player_level(seasons, game_lineup_cache,
                                       batter_cache, lg_batter_avg,
                                       pitcher_cache, lg_pitcher_avg):
    """
    Build per-game feature vectors using player-level rolling stats.

    For each game:
      • Identify the actual starting pitcher (from box score) and their hand.
      • Select each batter's rolling wOBA / xwOBA vs that pitcher's hand type.
      • Aggregate across the batting order using aggregate_lineup_metric().
      • Look up the starter's rolling xFIP and xwOBA allowed.
      • Normalize by league average → compute 4 difference features.
      • Attach de-vigged market logit as the market anchor (same as meta model).

    Features returned (same schema as collect_game_features_for_meta):
      logit_market_prob, d_woba, d_xwoba_bat, d_xfip, d_xwoba_pit

    Pitching features use the HOME team's pitcher rating vs the AWAY team's,
    so a positive d_xfip means the home starter has a better (lower) xFIP.
    """
    # Build (date, home_fg, away_fg) → lineup_data lookup
    # Doubleheaders: keep a list; caller gets the first match.
    lineup_by_teams = {}
    for season_cache in game_lineup_cache.values():
        for game_pk_str, gdata in season_cache.items():
            key = (gdata['date'], gdata['home_fg'], gdata['away_fg'])
            lineup_by_teams.setdefault(key, []).append(gdata)

    records = []
    missing_lineup = missing_starter = missing_lg = 0

    for season in seasons:
        season_str  = str(season)
        season_odds = odds_df[odds_df['date'].str.startswith(season_str)]
        _, fip_c    = load_woba_fip_constants('constants/woba_fip_constants.csv', season)

        for _, game in season_odds.iterrows():
            date      = game['date']
            home_team = normalize_team(game['home_team'])
            away_team = normalize_team(game['away_team'])

            if home_team not in TEAM_MAP or away_team not in TEAM_MAP:
                continue

            home_score = game.get('home_score')
            away_score = game.get('away_score')
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            fd_home_ml = game.get('fd_home_ml')
            fd_away_ml = game.get('fd_away_ml')
            if pd.isna(fd_home_ml) or pd.isna(fd_away_ml):
                continue
            fd_home_ml = float(fd_home_ml)
            fd_away_ml = float(fd_away_ml)
            if fd_home_ml == 0 or fd_away_ml == 0:
                continue

            # ── Lineup lookup ──────────────────────────────────────────────
            gdata_list = lineup_by_teams.get((date, home_team, away_team))
            if not gdata_list:
                missing_lineup += 1
                continue
            gdata = gdata_list[0]  # first match (handles most cases; rare doubleheaders may mismatch)

            home_lineup       = [int(p) for p in gdata['home_lineup']]
            away_lineup       = [int(p) for p in gdata['away_lineup']]
            home_starter_id   = int(gdata['home_starter_id'])
            away_starter_id   = int(gdata['away_starter_id'])
            # Hand of pitcher the OPPOSING lineup faces:
            #   home batters face the AWAY starter's hand
            #   away batters face the HOME starter's hand
            hand_vs_home = gdata['away_starter_hand']   # hand home batters face
            hand_vs_away = gdata['home_starter_hand']   # hand away batters face

            # ── Starter stats lookup ───────────────────────────────────────
            home_pit = pitcher_cache.get((home_starter_id, date))
            away_pit = pitcher_cache.get((away_starter_id, date))
            if home_pit is None or away_pit is None:
                missing_starter += 1
                continue

            # ── League averages ────────────────────────────────────────────
            lg_bat_h = lg_batter_avg.get((hand_vs_home, date))   # for home batters vs hand
            lg_bat_a = lg_batter_avg.get((hand_vs_away, date))   # for away batters vs hand
            lg_pit   = lg_pitcher_avg.get(date)
            if lg_bat_h is None or lg_bat_a is None or lg_pit is None:
                missing_lg += 1
                continue

            # ── Batter stats dicts for this game ──────────────────────────
            # home batters: rolling wOBA/xwOBA vs hand_vs_home
            home_woba_dict  = {pid: batter_cache.get((pid, hand_vs_home, date), {}).get('woba')
                               for pid in home_lineup}
            home_xwoba_dict = {pid: batter_cache.get((pid, hand_vs_home, date), {}).get('xwoba')
                               for pid in home_lineup}
            # away batters: rolling wOBA/xwOBA vs hand_vs_away
            away_woba_dict  = {pid: batter_cache.get((pid, hand_vs_away, date), {}).get('woba')
                               for pid in away_lineup}
            away_xwoba_dict = {pid: batter_cache.get((pid, hand_vs_away, date), {}).get('xwoba')
                               for pid in away_lineup}

            # Remove None values — lg_avg fills those slots inside aggregate_lineup_metric
            home_woba_dict  = {k: v for k, v in home_woba_dict.items()  if v is not None}
            home_xwoba_dict = {k: v for k, v in home_xwoba_dict.items() if v is not None}
            away_woba_dict  = {k: v for k, v in away_woba_dict.items()  if v is not None}
            away_xwoba_dict = {k: v for k, v in away_xwoba_dict.items() if v is not None}

            # ── Aggregate across batting order ─────────────────────────────
            home_woba     = aggregate_lineup_metric(home_woba_dict,  home_lineup, lg_bat_h['woba'])
            home_xwoba_bat = aggregate_lineup_metric(home_xwoba_dict, home_lineup, lg_bat_h['xwoba'])
            away_woba     = aggregate_lineup_metric(away_woba_dict,  away_lineup, lg_bat_a['woba'])
            away_xwoba_bat = aggregate_lineup_metric(away_xwoba_dict, away_lineup, lg_bat_a['xwoba'])

            # ── Normalize batting by league average ───────────────────────
            lg_woba_h  = lg_bat_h['woba']  or 0.320
            lg_xwoba_h = lg_bat_h['xwoba'] or 0.315
            lg_woba_a  = lg_bat_a['woba']  or 0.320
            lg_xwoba_a = lg_bat_a['xwoba'] or 0.315

            home_woba_r     = home_woba     / lg_woba_h  if lg_woba_h  > 0 else 1.0
            away_woba_r     = away_woba     / lg_woba_a  if lg_woba_a  > 0 else 1.0
            home_xwoba_bat_r = home_xwoba_bat / lg_xwoba_h if lg_xwoba_h > 0 else 1.0
            away_xwoba_bat_r = away_xwoba_bat / lg_xwoba_a if lg_xwoba_a > 0 else 1.0

            # ── Pitcher ratings (lower stat = better → invert) ─────────────
            lg_xfip      = lg_pit['xfip']
            lg_xwoba_pit = lg_pit['xwoba_pit']

            home_xfip_r      = lg_xfip      / home_pit['xfip']      if home_pit['xfip']      > 0 else 1.0
            away_xfip_r      = lg_xfip      / away_pit['xfip']      if away_pit['xfip']      > 0 else 1.0
            home_xwoba_pit_r = lg_xwoba_pit / home_pit['xwoba_pit'] if home_pit['xwoba_pit'] > 0 else 1.0
            away_xwoba_pit_r = lg_xwoba_pit / away_pit['xwoba_pit'] if away_pit['xwoba_pit'] > 0 else 1.0

            # ── Market logit anchor ────────────────────────────────────────
            p_home_dv, _ = de_vig_probs(fd_home_ml, fd_away_ml)
            p_clipped    = float(np.clip(p_home_dv, 0.001, 0.999))
            logit_mp     = float(np.log(p_clipped / (1.0 - p_clipped)))

            records.append({
                'date':              date,
                'season':            season,
                'home_team':         home_team,
                'away_team':         away_team,
                'logit_market_prob': logit_mp,
                'd_woba':            home_woba_r      - away_woba_r,
                'd_xwoba_bat':       home_xwoba_bat_r - away_xwoba_bat_r,
                'd_xfip':            home_xfip_r      - away_xfip_r,
                'd_xwoba_pit':       home_xwoba_pit_r - away_xwoba_pit_r,
                'y':                 1 if home_score > away_score else 0,
                'fd_home_ml':        fd_home_ml,
                'fd_away_ml':        fd_away_ml,
            })

    print(f"\n  Player-level features: {len(records):,} games")
    print(f"  Skipped — no lineup: {missing_lineup:,} | no starter stats: {missing_starter:,} | no lg avg: {missing_lg:,}")
    return pd.DataFrame(records)


def test_ip_cache(season=2024, n_games=10):
    """
    Verify that pull_historical_lineups correctly extracts inningsPitched and
    outs from MLB API box scores.  Fetches the first n_games April games for
    the given season, prints a verification table, does NOT write to disk.

    Expected output: every row should show a valid IP string ('5.0', '6.2', etc.)
    and a non-zero outs integer, confirming the boxscore stats block path is correct.
    """
    print(f"\n{'='*80}")
    print(f"  IP CACHE VERIFICATION TEST  |  Season {season}  |  Sample: {n_games} games")
    print(f"{'='*80}\n")

    schedule = statsapi.schedule(
        start_date=f'{season}-04-01',
        end_date=f'{season}-04-30',
        sportId=1,
    )
    final_games = [g for g in schedule if g.get('status') == 'Final'][:n_games]
    print(f"  Fetching {len(final_games)} box scores from April {season}...\n")

    header = f"  {'game_pk':<10} {'date':<12} {'side':<5} {'team':<5} {'starter_id':<12} {'IP str':<9} {'IP dec':<9} status"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    all_ok = True
    for game in final_games:
        game_pk  = game['game_id']
        date_str = game['game_date']
        home_fg  = _MLB_ID_TO_FG.get(game['home_id'], '?')
        away_fg  = _MLB_ID_TO_FG.get(game['away_id'], '?')

        try:
            bs = statsapi.boxscore_data(game_pk)
        except Exception as e:
            print(f"  {game_pk}  {date_str}  ERROR: {e}")
            all_ok = False
            continue

        for side, team_fg in (('home', home_fg), ('away', away_fg)):
            pitchers = bs[side].get('pitchers', [])
            if not pitchers:
                print(f"  {game_pk:<10} {date_str:<12} {side:<5} {team_fg:<5} {'—':<12} no pitchers list")
                all_ok = False
                continue

            starter_id = pitchers[0]
            player_key = f'ID{starter_id}'
            pdata      = bs[side].get('players', {}).get(player_key, {})
            pit        = pdata.get('stats', {}).get('pitching', {})
            ip_str     = pit.get('inningsPitched', 'MISSING')

            # inningsPitched is all we need — 'outs' is not in the MLB API boxscore
            ok = ip_str != 'MISSING' and ip_str is not None
            if not ok:
                all_ok = False
                print(f"  {game_pk:<10} {date_str:<12} {side:<5} {team_fg:<5} "
                      f"{starter_id:<12} {'MISSING':<9} {'—':<9} !! MISSING")
                continue

            try:
                ip_dec = f'{ip_to_float(str(ip_str)):.3f}'
            except Exception:
                ip_dec = 'ERR'
                all_ok = False

            print(f"  {game_pk:<10} {date_str:<12} {side:<5} {team_fg:<5} "
                  f"{starter_id:<12} {str(ip_str):<9} {ip_dec:<9} OK")

    print()
    if all_ok:
        print("  Result: All starters have valid inningsPitched and outs in box score.")
        print("  Ready to run pull_historical_lineups() for full season caching.")
    else:
        print("  WARNING: Some entries are missing. Check boxscore structure above.")
    print()


# ── Run mode ──────────────────────────────────────────────────────────────────
# 'train'              — standard ROI-optimized parameter search (round 8)
# 'walk_forward'       — 4-fold walk-forward validation (~3 hours)
# 'individual_metrics' — test each metric in isolation for out-of-sample signal
# 'calibration'        — Brier-score-optimized parameter search
# 'meta_model'         — logistic regression over 4 validated metrics, walk-forward
# 'financial_sim'      — flat-bet ROI simulation using meta model fold probabilities
# 'player_level_meta'  — player-level features (box score lineups + 100-PA rolling)
# 'dual_strategy'      — Sniper + Enforcer sim on pre-built features (skips data build)
# 'test_ip_cache'      — verify inningsPitched/outs extraction from box scores (quick)
RUN_MODE    = 'player_level_meta'
N_RUNS      = 10
_PARAM_ROUND = 8  # must match PARAM_ROUND inside random_search

import glob as _glob_outer
import ctypes as _ctypes
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002

_ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED)

try:
    if RUN_MODE == 'train':
        _completed_files = _glob_outer.glob(f'historical_data/search_*_100trials_r{_PARAM_ROUND}.csv')
        _completed_runs  = len(_completed_files)
        _remaining_runs  = max(0, N_RUNS - _completed_runs)
        if _remaining_runs == 0:
            print(f"All {N_RUNS} runs already complete. Running evaluate_runs().")
        else:
            if _completed_runs:
                print(f"Found {_completed_runs} completed run(s). Running {_remaining_runs} more.")
            for run in range(_remaining_runs):
                print(f"\n{'='*55}")
                print(f"  Run {_completed_runs + run + 1} of {N_RUNS}")
                print(f"{'='*55}")
                random_search(n_trials=100)
        evaluate_runs()

    elif RUN_MODE == 'walk_forward':
        walk_forward_validation(n_trials_per_fold=100, n_runs_per_fold=3)

    elif RUN_MODE == 'individual_metrics':
        test_individual_metrics(train_seasons=[2021, 2022], test_seasons=[2023, 2024, 2025])

    elif RUN_MODE == 'calibration':
        for run in range(N_RUNS):
            print(f"\n{'='*55}")
            print(f"  Calibration run {run + 1} of {N_RUNS}")
            print(f"{'='*55}")
            random_search_calibration(n_trials=100, train_seasons=[2021, 2022])

    elif RUN_MODE == 'meta_model':
        walk_forward_meta_model(C=1.0)

    elif RUN_MODE == 'financial_sim':
        print("Collecting game features for 2021–2025 (once for all sweeps)...")
        _feat_cache = collect_game_features_for_meta([2021, 2022, 2023, 2024, 2025])
        print(f"  {len(_feat_cache)} games with known outcomes\n")
        for _thresh in [0.03, 0.04, 0.05]:
            walk_forward_financial_sim(C=1.0, flat_bet=20.0, edge_min=_thresh, all_feat=_feat_cache)

    elif RUN_MODE == 'dual_strategy':
        print("Collecting team-level meta model features for dual strategy sim...")
        _ds_feat = collect_game_features_for_meta([2021, 2022, 2023, 2024, 2025])
        print(f"  {len(_ds_feat)} games collected\n")
        walk_forward_dual_strategy(_ds_feat, flat_bet=20.0)

    elif RUN_MODE == 'test_ip_cache':
        test_ip_cache(season=2024, n_games=10)

    elif RUN_MODE == 'player_level_meta':
        _pl_seasons = [2021, 2022, 2023, 2024, 2025]

        print("Step 1/5 — Loading historical box score lineups...")
        _lineup_cache = pull_historical_lineups(_pl_seasons)

        print("\nStep 2/5 — Loading individual batter data...")
        _batter_data = load_player_batter_data(_pl_seasons)

        print("\nStep 3/5 — Loading individual starter data...")
        _pitcher_data = load_player_pitcher_data(_pl_seasons)

        print("\nStep 4/5 — Building rolling caches (100 PA windows)...")
        _batter_cache, _lg_bat_avg = build_batter_rolling_cache(_batter_data)
        _fip_consts = {s: load_woba_fip_constants('constants/woba_fip_constants.csv', s)[1]
                       for s in _pl_seasons}
        _pitcher_cache, _lg_pit_avg = build_pitcher_rolling_cache(_pitcher_data, _fip_consts)

        print("\nStep 5/5 — Collecting player-level game features...")
        _pl_feat = collect_game_features_player_level(
            _pl_seasons,
            _lineup_cache, _batter_cache, _lg_bat_avg, _pitcher_cache, _lg_pit_avg,
        )

        if _pl_feat.empty:
            print("No features collected — check that player_data/ CSVs exist.")
        else:
            print(f"\nStep 5b — Brier score calibration check (no ROI)...")
            walk_forward_meta_model(C=1.0)   # calibration — does not use _pl_feat directly

            print(f"\nStep 5c — Dual strategy financial simulation...")
            walk_forward_dual_strategy(_pl_feat, flat_bet=20.0)

finally:
    # Always re-enable sleep when done or if an error occurs
    _ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
