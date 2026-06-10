#!/usr/bin/env python3
"""
train_residual_model.py  -  Smooth Alpha Calibration Model
------------------------------------------------------------
Architecture: regularised Logistic Regression (C=0.01, L2), anchored on
market_logit, trained on all 19 features including schedule/microstructure.

Smooth Alpha Calibration (replaces hard sniper gate):
  Instead of a binary gate that fires or suppresses the model entirely, each
  game receives a continuous confidence weight w in [0, 1] derived from how
  far its physics + line-velocity features deviate from the training median
  (IQR-normalised z-score):

    z_i     = |val_i - train_median_i| / train_IQR_i
    max_z   = max(z_i) across SMOOTH_FEATURES
    w       = sigmoid(STEEPNESS * (max_z - CENTER_Z))

  The final predicted residual is:
    final_residual = clip(w * model_residual, -CLIP_ABS, CLIP_ABS)

  At the IQR boundary (z=1): w = 0.50 — 50% model / 50% market
  At the training median  (z=0): w ≈ 0.05 — near-full market deference
  At 2× IQR out          (z=2): w ≈ 0.95 — near-full model application

  This naturally implements low-volume anomaly detection without discontinuity:
  typical games receive tiny (noise-free) adjustments; structural extremes
  (severe bullpen fatigue, cluster-luck spikes, sharp steam) receive the full
  model adjustment.

Cluster-luck step function fix:
  Previous thresholds (±1.0) were miscalibrated: extreme_neg never fired
  (threshold below data min of -0.62) and extreme_pos fired 92% of the time
  (below the data median of 1.79).  Corrected to training p05/p95:
    extreme_neg threshold < 0.92  (fires ~5%)
    extreme_pos threshold > 2.86  (fires ~5%)

Open-line fallback (unchanged):
  1. odds_202X.csv           - opening proxy, ~50% coverage 2021-2024
  2. odds_snapshots.csv      - earliest pre-game snap (2026+ live)
  3. Explicit 0.0 fill       - no steam; default to pure base signal

Walk-forward folds:
  Fold 1 : Train 2021          -> Test 2022
  Fold 2 : Train 2021-2022     -> Test 2023
  Fold 3 : Train 2021-2023     -> Test 2024
  HOLDOUT: Train 2021-2024     -> Test 2025  (strict out-of-sample)

Outputs:
  models/residual_smooth.pkl
  historical_data/oos_predictions_2025.csv
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent

MANIFEST     = ROOT / 'historical_data' / 'training_manifest.csv'
SNAPSHOT_LOG = ROOT / 'cache' / 'odds_snapshots.csv'
MODEL_OUT    = ROOT / 'models' / 'residual_smooth.pkl'
PREDS_OUT    = ROOT / 'historical_data' / 'oos_predictions_2025.csv'

# ── Feature sets ──────────────────────────────────────────────────────────────

BASE_FEATURE_COLS = [
    'home_bullpen_3d_pitches',
    'away_bullpen_3d_pitches',
    'home_lineup_vs_away_pitch_mix',
    'away_lineup_vs_home_pitch_mix',
    'home_team_base_runs_delta',
    'away_team_base_runs_delta',
]

STEP_FEATURE_COLS = [
    'home_bullpen_fatigue_critical',
    'away_bullpen_fatigue_critical',
    'home_cluster_luck_extreme_pos',   # base_runs_delta > 2.86 (~p95)
    'home_cluster_luck_extreme_neg',   # base_runs_delta < 0.92 (~p05)
    'away_cluster_luck_extreme_pos',
    'away_cluster_luck_extreme_neg',
]

MICRO_FEATURE_COLS = [
    'line_movement',
    'steam_indicator',
    'signal_harmony',
]

SCHEDULE_FEATURE_COLS = [
    'home_away_density_altitude_delta',
    'travel_fatigue',
    'market_total_mismatch',
]

MICRO_ALL = (
    ['market_logit']
    + BASE_FEATURE_COLS
    + STEP_FEATURE_COLS
    + MICRO_FEATURE_COLS
    + SCHEDULE_FEATURE_COLS
)

# ── Cluster-luck percentile thresholds (corrected from ±1.0) ─────────────────
# Derived from full 2021-2025 dataset: p05 ≈ 0.91, p95 ≈ 2.86
# These are stable year-to-year (base_runs_delta distribution barely shifts).
CLUSTER_LUCK_POS_THRESHOLD = 2.86   # ~95th pct  — fires ~5% of games
CLUSTER_LUCK_NEG_THRESHOLD = 0.92   # ~5th pct   — fires ~5% of games

# ── Smooth confidence calibration parameters ─────────────────────────────────
# Features that drive the confidence weight (line velocity + physics extremes)
SMOOTH_FEATURES = [
    'home_bullpen_3d_pitches',
    'away_bullpen_3d_pitches',
    'home_team_base_runs_delta',
    'away_team_base_runs_delta',
    'line_movement',
]

STEEPNESS    = 3.0   # sigmoid steepness: higher = sharper transition
CENTER_Z     = 1.0   # z-score at which w = 0.50 (IQR boundary)

TARGET     = 'home_win'
CLIP_ABS   = 0.05
HOLDOUT_YR = 2025

FOLDS = [
    (list(range(2021, 2022)), 2022),
    (list(range(2021, 2023)), 2023),
    (list(range(2021, 2024)), 2024),
]

# Previous run for comparison: sniper logit + dynamic DA (worst result)
_PREV_BSS   = {2022: -0.00356, 2023: -0.00012, 2024: -0.00060, 2025: -0.00202}
_PREV_MKT   = {2022: 0.23696,  2023: 0.24243,  2024: 0.24036,  2025: 0.24252}

_NAME_TO_FG = {
    'Athletics':               'ATH',
    'Pittsburgh Pirates':      'PIT',
    'San Diego Padres':        'SDP',
    'Seattle Mariners':        'SEA',
    'San Francisco Giants':    'SFG',
    'St. Louis Cardinals':     'STL',
    'Tampa Bay Rays':          'TBR',
    'Texas Rangers':           'TEX',
    'Toronto Blue Jays':       'TOR',
    'Minnesota Twins':         'MIN',
    'Philadelphia Phillies':   'PHI',
    'Atlanta Braves':          'ATL',
    'Chicago White Sox':       'CHW',
    'Miami Marlins':           'MIA',
    'New York Yankees':        'NYY',
    'Milwaukee Brewers':       'MIL',
    'Los Angeles Angels':      'LAA',
    'Arizona Diamondbacks':    'ARI',
    'Baltimore Orioles':       'BAL',
    'Boston Red Sox':          'BOS',
    'Chicago Cubs':            'CHC',
    'Cincinnati Reds':         'CIN',
    'Cleveland Guardians':     'CLE',
    'Colorado Rockies':        'COL',
    'Detroit Tigers':          'DET',
    'Houston Astros':          'HOU',
    'Kansas City Royals':      'KCR',
    'Los Angeles Dodgers':     'LAD',
    'Washington Nationals':    'WSN',
    'New York Mets':           'NYM',
}


# ── Odds math ─────────────────────────────────────────────────────────────────

def _american_to_implied(odds: float) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def _devige(home_ml: float, away_ml: float) -> float:
    rh    = _american_to_implied(home_ml)
    ra    = _american_to_implied(away_ml)
    total = rh + ra
    return rh / total if total > 0 else np.nan


# ── Opening line: Source 2 — odds_snapshots.csv ───────────────────────────────

def _load_snapshot_opens() -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=['game_date', 'home_team', 'away_team', 'market_prob_open_home']
    )
    if not SNAPSHOT_LOG.exists():
        return empty
    try:
        snaps = pd.read_csv(SNAPSHOT_LOG)
    except Exception:
        return empty
    if snaps.empty:
        return empty

    snaps['logged_at_dt'] = pd.to_datetime(
        snaps['logged_at'].str.replace('Z', '+00:00', regex=False),
        utc=True, errors='coerce',
    )
    snaps['commence_dt'] = pd.to_datetime(
        snaps['commence_time'].str.replace('Z', '+00:00', regex=False),
        utc=True, errors='coerce',
    )
    snaps = snaps.dropna(subset=['logged_at_dt', 'commence_dt'])
    snaps = snaps[snaps['logged_at_dt'] < snaps['commence_dt']].copy()
    if snaps.empty:
        return empty

    snaps    = snaps.sort_values('logged_at_dt')
    earliest = snaps.groupby(
        ['game_date', 'home_team', 'away_team'], as_index=False
    ).first()

    def _prob(row) -> float:
        h = row.get('fd_home_ml') or row.get('dk_home_ml')
        a = row.get('fd_away_ml') or row.get('dk_away_ml')
        try:
            return _devige(float(h), float(a))
        except Exception:
            return np.nan

    earliest['market_prob_open_home'] = earliest.apply(_prob, axis=1)
    earliest = earliest.dropna(subset=['market_prob_open_home'])
    return earliest[['game_date', 'home_team', 'away_team', 'market_prob_open_home']]


# ── Opening line: Source 1 — odds_202X.csv ───────────────────────────────────

def _load_opening_odds() -> pd.DataFrame:
    frames = []
    for yr in [2021, 2022, 2023, 2024]:
        path = ROOT / 'historical_data' / f'odds_{yr}.csv'
        if not path.exists():
            continue
        raw = pd.read_csv(path)

        def _parse_date(s: str):
            try:
                return datetime.strptime(s.strip(), '%d %b %Y').strftime('%Y-%m-%d')
            except Exception:
                return None

        raw['game_date'] = raw['date'].apply(_parse_date)
        raw['home_team'] = raw['home_team'].map(_NAME_TO_FG)
        raw['away_team'] = raw['away_team'].map(_NAME_TO_FG)
        raw = raw.dropna(
            subset=['game_date', 'home_team', 'away_team', 'home_odds', 'away_odds']
        )

        def _prob(row) -> float:
            try:
                return _devige(float(row['home_odds']), float(row['away_odds']))
            except Exception:
                return np.nan

        raw['market_prob_open_home'] = raw.apply(_prob, axis=1)
        raw = raw.dropna(subset=['market_prob_open_home'])
        frames.append(
            raw[['game_date', 'home_team', 'away_team', 'market_prob_open_home']]
        )

    snap_opens = _load_snapshot_opens()
    if not snap_opens.empty:
        frames.append(snap_opens)

    if not frames:
        return pd.DataFrame(
            columns=['game_date', 'home_team', 'away_team', 'market_prob_open_home']
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(
        subset=['game_date', 'home_team', 'away_team'], keep='first'
    )


# ── Feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame, opening_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market_logit, corrected step-function features, and microstructure.

    Cluster-luck thresholds corrected from ±1.0 to training p05/p95:
      extreme_pos: > 2.86 fires ~5% of games (was 92% with >1.0)
      extreme_neg: < 0.92 fires ~5% of games (was 0% with <-1.0)
    """
    df = df.copy()

    # Market log-odds anchor
    p = df['market_prob_home'].clip(1e-6, 1.0 - 1e-6)
    df['market_logit'] = np.log(p / (1.0 - p))

    # Bullpen fatigue step functions (unchanged)
    df['home_bullpen_fatigue_critical'] = (df['home_bullpen_3d_pitches'] > 100).astype(float)
    df['away_bullpen_fatigue_critical'] = (df['away_bullpen_3d_pitches'] > 100).astype(float)

    # Cluster luck — corrected to percentile-based thresholds
    df['home_cluster_luck_extreme_pos'] = (
        df['home_team_base_runs_delta'] > CLUSTER_LUCK_POS_THRESHOLD
    ).astype(float)
    df['home_cluster_luck_extreme_neg'] = (
        df['home_team_base_runs_delta'] < CLUSTER_LUCK_NEG_THRESHOLD
    ).astype(float)
    df['away_cluster_luck_extreme_pos'] = (
        df['away_team_base_runs_delta'] > CLUSTER_LUCK_POS_THRESHOLD
    ).astype(float)
    df['away_cluster_luck_extreme_neg'] = (
        df['away_team_base_runs_delta'] < CLUSTER_LUCK_NEG_THRESHOLD
    ).astype(float)

    # Opening line join
    if not opening_odds.empty:
        df = df.merge(
            opening_odds[['game_date', 'home_team', 'away_team', 'market_prob_open_home']],
            on=['game_date', 'home_team', 'away_team'],
            how='left',
        )
    else:
        df['market_prob_open_home'] = np.nan

    # Microstructure (0.0 fill for unmatched games)
    lm_raw               = df['market_prob_home'] - df['market_prob_open_home']
    df['_has_open_line'] = lm_raw.notna()
    df['line_movement']  = lm_raw.fillna(0.0)
    df['steam_indicator'] = (df['line_movement'].abs() > 0.03).astype(float)

    net_brd = (
        df['home_team_base_runs_delta'].fillna(0.0)
        - df['away_team_base_runs_delta'].fillna(0.0)
    )
    df['signal_harmony'] = df['line_movement'] * net_brd

    return df


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('lr',      LogisticRegression(
            C             = 0.01,
            solver        = 'lbfgs',
            max_iter      = 2000,
            fit_intercept = True,
            random_state  = 42,
        )),
    ])


# ── Smooth confidence calibration ─────────────────────────────────────────────

def _smooth_confidence_weights(
    test_df:  pd.DataFrame,
    train_df: pd.DataFrame,
) -> np.ndarray:
    """
    Compute per-game smooth confidence weights w in [0, 1].

    For each game, find the maximum IQR-normalised z-score across SMOOTH_FEATURES:
      z_i   = |val_i - train_median_i| / train_IQR_i
      max_z = max(z_i) across features
      w     = sigmoid(STEEPNESS * (max_z - CENTER_Z))

    NaN feature values are imputed to the training median before z-scoring
    (they are not extreme, so they contribute z=0 to the max).

    w ≈ 0.05  at median values  (z=0)  — near-full market deference
    w = 0.50  at IQR boundary   (z=1)  — 50-50 blend
    w ≈ 0.82  at 1.5× IQR out  (z=1.5) — strong model weighting
    w ≈ 0.95  at 2× IQR out    (z=2)  — near-full model
    """
    p25  = train_df[SMOOTH_FEATURES].quantile(0.25)
    p50  = train_df[SMOOTH_FEATURES].quantile(0.50)
    p75  = train_df[SMOOTH_FEATURES].quantile(0.75)
    iqr  = (p75 - p25).clip(lower=1e-6)

    max_z = np.zeros(len(test_df))
    for feat in SMOOTH_FEATURES:
        med  = float(p50[feat])
        iqr_ = float(iqr[feat])
        vals = test_df[feat].fillna(med).values.astype(float)
        z    = np.abs(vals - med) / iqr_
        max_z = np.maximum(max_z, z)

    return 1.0 / (1.0 + np.exp(-STEEPNESS * (max_z - CENTER_Z)))


# ── Metrics ───────────────────────────────────────────────────────────────────

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def brier_skill_score(model_bs: float, baseline_bs: float) -> float:
    return 1.0 - model_bs / baseline_bs if baseline_bs else 0.0


# ── Fold evaluator ────────────────────────────────────────────────────────────

def evaluate_fold(
    model:              Pipeline,
    test_df:            pd.DataFrame,
    conf_weights:       np.ndarray,
    label:              str,
    verbose:            bool = True,
) -> dict:
    """
    LR prediction -> smooth confidence weighting -> clip -> final prob.

    final_residual = clip(conf_weight * model_residual, -CLIP_ABS, CLIP_ABS)

    Confidence weights are derived from training-data statistics (no test leakage).
    """
    X_test   = test_df[MICRO_ALL]
    y_true   = test_df['home_win'].values.astype(float)
    mkt_prob = test_df['market_prob_home'].values

    raw_logit      = model.decision_function(X_test)
    final_prob_raw = 1.0 / (1.0 + np.exp(-raw_logit))

    predicted_residual = final_prob_raw - mkt_prob
    scaled             = predicted_residual * conf_weights
    clipped            = np.clip(scaled, -CLIP_ABS, CLIP_ABS)
    final_prob         = np.clip(mkt_prob + clipped, 0.0, 1.0)

    bs_mkt   = brier_score(mkt_prob,   y_true)
    bs_model = brier_score(final_prob, y_true)
    bss      = brier_skill_score(bs_model, bs_mkt)

    # Games receiving a meaningful adjustment (> 1 bp)
    n_active   = int((np.abs(clipped) > 0.001).sum())
    pct_active = 100.0 * n_active / max(len(test_df), 1)

    if verbose:
        n         = len(test_df)
        open_pct  = 100.0 * test_df['_has_open_line'].mean()
        direction = 'improvement' if bss > 0 else 'no improvement'
        print(f"\n  {label}  ({n:,} games | open-line: {open_pct:.0f}%)")
        print(f"    Confidence: mean={conf_weights.mean():.3f}  "
              f"p50={np.median(conf_weights):.3f}  "
              f"pct>0.5={100*(conf_weights>0.5).mean():.1f}%")
        print(f"    Active adj (>1bp): {n_active:,} / {n:,}  ({pct_active:.1f}%)")
        print(f"    Market Brier : {bs_mkt:.5f}")
        print(f"    Model  Brier : {bs_model:.5f}")
        print(f"    BSS          : {bss:+.5f}  [{direction}]")

    return {
        'label':     label,
        'n_games':   len(test_df),
        'bs_mkt':    bs_mkt,
        'bs_model':  bs_model,
        'bss':       bss,
        'n_active':  n_active,
        'pct_active':pct_active,
        'conf_mean': float(conf_weights.mean()),
    }


# ── Coefficient printer ───────────────────────────────────────────────────────

def _print_coefficients(model: Pipeline) -> None:
    lr     = model.named_steps['lr']
    scaler = model.named_steps['scaler']
    coefs  = lr.coef_[0]
    scales = scaler.scale_
    raw_eff = coefs / scales
    print("\n  LR coefficients (per 1-unit raw feature change):")
    for col, c in sorted(zip(MICRO_ALL, raw_eff), key=lambda x: -abs(x[1])):
        bar = '#' * max(1, int(abs(c) * 100))
        print(f"    {col:<46}  {c:+.6f}  {bar}")
    print(f"\n  Intercept: {lr.intercept_[0]:+.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep  = '=' * 72
    sep2 = '-' * 72

    print(sep)
    print('  SMOOTH ALPHA CALIBRATION MODEL')
    print('  Engine: LR(C=0.01, L2)  |  Calibration: IQR-sigmoid confidence weight')
    print(f"  Cluster-luck thresholds: neg<{CLUSTER_LUCK_NEG_THRESHOLD}  "
          f"pos>{CLUSTER_LUCK_POS_THRESHOLD}  (corrected from +/-1.0)")
    print(sep)

    # ── Load and engineer features ─────────────────────────────────
    df = pd.read_csv(MANIFEST)
    df['year'] = df['game_date'].str[:4].astype(int)
    df = df.sort_values('game_date').reset_index(drop=True)

    opening_odds = _load_opening_odds()
    df = add_features(df, opening_odds)

    print(f"\n  Dataset : {len(df):,} games  "
          f"({df['game_date'].min()} to {df['game_date'].max()})")

    by_yr = df.groupby('year')['_has_open_line'].mean()
    print(f"  Open-line: "
          + "  ".join(f"{yr}={v:.0%}" for yr, v in by_yr.items()))

    # Corrected step-function coverage
    print(f"\n  Step-function coverage (corrected thresholds):")
    for col in STEP_FEATURE_COLS:
        n_pos = int(df[col].sum())
        pct   = 100.0 * n_pos / len(df)
        print(f"    {col:<46}  {n_pos:>5,} / {len(df):,}  ({pct:.1f}%)")

    # Smooth feature distributions
    train_all = df[df['year'] < HOLDOUT_YR]
    p25 = train_all[SMOOTH_FEATURES].quantile(0.25)
    p50 = train_all[SMOOTH_FEATURES].quantile(0.50)
    p75 = train_all[SMOOTH_FEATURES].quantile(0.75)
    iqr = p75 - p25
    print(f"\n  Smooth confidence features (training 2021-2024 IQR stats):")
    for feat in SMOOTH_FEATURES:
        print(f"    {feat:<42}  "
              f"p25={p25[feat]:+.4f}  p50={p50[feat]:+.4f}  "
              f"p75={p75[feat]:+.4f}  IQR={iqr[feat]:.4f}")

    # ── Walk-forward CV ────────────────────────────────────────────
    print(f"\n{sep}")
    print('  WALK-FORWARD CROSS-VALIDATION')
    print(sep)

    fold_results = []
    for train_years, test_year in FOLDS:
        train_mask = df['year'].isin(train_years)
        test_mask  = df['year'] == test_year

        model = _make_model()
        model.fit(df.loc[train_mask, MICRO_ALL], df.loc[train_mask, TARGET])

        # Smooth confidence derived from training statistics only
        conf_w = _smooth_confidence_weights(
            df[test_mask], df[train_mask]
        )

        label = (f"Train {train_years[0]}"
                 + (f"-{train_years[-1]}" if len(train_years) > 1 else "")
                 + f" -> Test {test_year}")
        fold_results.append(
            evaluate_fold(model, df[test_mask].copy(), conf_w, label)
        )

    # ── Final model + holdout ──────────────────────────────────────
    print(f"\n{sep}")
    print(f'  OUT-OF-SAMPLE HOLDOUT: {HOLDOUT_YR}')
    print(sep)

    train_mask   = df['year'] < HOLDOUT_YR
    holdout_mask = df['year'] == HOLDOUT_YR

    final_model  = _make_model()
    final_model.fit(df.loc[train_mask, MICRO_ALL], df.loc[train_mask, TARGET])

    holdout_df   = df[holdout_mask].copy()
    holdout_conf = _smooth_confidence_weights(holdout_df, df[train_mask])

    holdout_r = evaluate_fold(
        final_model, holdout_df, holdout_conf,
        f'HOLDOUT {HOLDOUT_YR} (strict out-of-sample)',
    )

    _print_coefficients(final_model)

    # ── Save predictions ───────────────────────────────────────────
    raw_logit      = final_model.decision_function(holdout_df[MICRO_ALL])
    final_prob_raw = 1.0 / (1.0 + np.exp(-raw_logit))
    scaled  = (final_prob_raw - holdout_df['market_prob_home'].values) * holdout_conf
    clipped = np.clip(scaled, -CLIP_ABS, CLIP_ABS)

    holdout_df = holdout_df.assign(
        confidence_weight     = holdout_conf.round(4),
        predicted_residual    = clipped.round(6),
        final_model_prob_home = np.clip(
            holdout_df['market_prob_home'].values + clipped, 0.0, 1.0
        ).round(6),
    )
    holdout_df['final_model_prob_away'] = (
        1.0 - holdout_df['final_model_prob_home']
    ).round(6)
    holdout_df.to_csv(PREDS_OUT, index=False)

    # ── Save model bundle ──────────────────────────────────────────
    MODEL_OUT.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'model':        final_model,
            'features':     MICRO_ALL,
            'smooth_feats': SMOOTH_FEATURES,
            'steepness':    STEEPNESS,
            'center_z':     CENTER_Z,
            'clip_abs':     CLIP_ABS,
            'train_years':  list(range(2021, HOLDOUT_YR)),
            'holdout_year': HOLDOUT_YR,
            'bs_market':    holdout_r['bs_mkt'],
            'bs_model':     holdout_r['bs_model'],
            'bss':          holdout_r['bss'],
        }, f)

    # ── Four-column comparison table ───────────────────────────────
    all_results = fold_results + [holdout_r]
    test_yrs    = [fy[1] for fy in FOLDS] + [HOLDOUT_YR]

    print(f"\n{sep}")
    print('  SMOOTH ALPHA CALIBRATION  -  FOUR-COLUMN COMPARISON')
    print('  Prev = Sniper Logit + dynamic DA (last run, worst result)')
    print('  New  = Smooth calibration + static DA + corrected thresholds')
    print(sep)

    col_w = 32
    hdr = (f"  {'Fold':<{col_w}} {'N':>6}  {'Mkt BS':>9}  "
           f"{'Prev BSS':>9}  {'New BSS':>9}  {'Active%':>8}  {'dBSS':>7}")
    print(hdr)
    print('  ' + sep2)

    for yr, r in zip(test_yrs, all_results):
        prev_bss = _PREV_BSS.get(yr, float('nan'))
        delta    = r['bss'] - prev_bss
        is_hout  = yr == HOLDOUT_YR
        n_flg    = ' +' if r['bss'] > 0 else '  '
        d_flg    = ' ^' if delta > 0 else (' v' if delta < 0 else '  ')
        if is_hout:
            print('  ' + sep2)
        fold_lbl = ('HOLDOUT 2025 (out-of-sample)'
                    if is_hout else r['label'])
        print(
            f"  {fold_lbl:<{col_w}} {r['n_games']:>6,}  "
            f"{r['bs_mkt']:>9.5f}  "
            f"{prev_bss:>+8.5f}  "
            f"{r['bss']:>+8.5f}{n_flg}  "
            f"{r['pct_active']:>7.1f}%  "
            f"{delta:>+6.5f}{d_flg}"
        )

    print()
    print('  Lower Brier = better.  BSS > 0 = beats market.')
    print('  Active% = games receiving >1bp model adjustment.')
    print('  + beats market  |  ^ new > prev  |  v new < prev')
    print()
    print(f"  Predictions -> {PREDS_OUT}")
    print(f"  Model       -> {MODEL_OUT}")
    print()


if __name__ == '__main__':
    main()
