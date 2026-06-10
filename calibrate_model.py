"""
calibrate_model.py
──────────────────────────────────────────────────────────────────────────────
Platt Scaling calibration layer for the origination model.

The player-only Log5 model produces probabilities clustered tightly around
53% (std ~4%).  Platt scaling on those logits alone cannot move the Brier
Score meaningfully — there is no information to re-arrange.

This script fits a two-input logistic calibrator:
    X = [logit(p_player_only), logit_market_prob]
    y = actual home-win outcome (1/0)

The market logit carries strong real-world signal.  The player logit adds
whatever marginal lineup-vs-pitcher signal exists.  The calibrator learns
the optimal blend, regularised to avoid over-fitting on 671 games.

Usage:
    python calibrate_model.py

Output:
    models/platt_calibrator.pkl  (joblib dict with calibrator + metadata)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pickle
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ── helpers ───────────────────────────────────────────────────────────────────

def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))

def reliability_summary(p, y, bins=None):
    """Print a compact reliability table."""
    bins = bins or np.arange(0.35, 0.80, 0.05)
    print(f"  {'Bin':>10}  {'N':>5}  {'Pred':>6}  {'Actual':>6}  {'Gap':>6}")
    for lo in bins:
        hi   = lo + 0.05
        mask = (p >= lo) & (p < hi)
        nb   = int(mask.sum())
        if nb == 0:
            continue
        pred = float(p[mask].mean())
        act  = float(y[mask].mean())
        gap  = act - pred
        flag = ' OVER' if gap < -0.04 else (' UNDER' if gap > 0.04 else '')
        print(f"  [{lo:.0%}-{hi:.0%})  {nb:>5}  {pred:.3f}  {act:.3f}  {gap:+.3f}{flag}")


# ── 1. load cache ─────────────────────────────────────────────────────────────

CACHE_PATH = 'cache/calib_2026.pkl'
SAVE_PATH  = 'models/platt_calibrator.pkl'
TARGET_BRIER = 0.2450

if not os.path.exists(CACHE_PATH):
    print(f"ERROR: {CACHE_PATH} not found.")
    print("Run backtest.py with RUN_MODE = '2026_validation' first.")
    sys.exit(1)

with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

df    = cache['log5_2026']
p_raw = np.asarray(cache['p_home_raw'], float)   # player-only Log5 predictions
y     = df['y'].values.astype(int)                # 1 = home win

n        = len(y)
n_wins   = int(y.sum())
date_min = df['date'].min()
date_max = df['date'].max()

print(f"\n{'='*70}")
print(f"  calibrate_model.py  —  Platt Scaling  |  2026 origination model")
print(f"{'='*70}")
print(f"  Games : {n}  ({date_min} -> {date_max})")
print(f"  Wins  : {n_wins}  ({n_wins/n:.1%} home win rate)")

assert 'logit_market_prob' in df.columns, \
    "logit_market_prob missing from calib_2026.pkl — re-run backtest.py"


# ── 2. describe raw predictions ───────────────────────────────────────────────

print(f"\n  Raw player-only model  (before calibration)")
print(f"  Mean : {p_raw.mean():.4f}   Std : {p_raw.std():.4f}")
print(f"  Min  : {p_raw.min():.4f}   Max : {p_raw.max():.4f}")

brier_raw = brier(p_raw, y)
print(f"  Brier: {brier_raw:.4f}  (coin-flip baseline = 0.2500)")


# ── 3. player-only Platt (1-D) — diagnostic ──────────────────────────────────
# Shows why 1-D Platt on the player model is insufficient on its own.

print(f"\n  [A] Platt Scaling — player-only logit  (1 input)")
logits_player = logit(p_raw).reshape(-1, 1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

p_platt_1d = np.zeros(n)
for tr, va in skf.split(logits_player, y):
    clf = LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000)
    clf.fit(logits_player[tr], y[tr])
    p_platt_1d[va] = clf.predict_proba(logits_player[va])[:, 1]

brier_platt_1d = brier(p_platt_1d, y)
print(f"  Brier (5-fold CV) : {brier_platt_1d:.4f}  (improvement: {brier_raw - brier_platt_1d:+.4f})")
if brier_platt_1d >= TARGET_BRIER:
    print(f"  ** Does NOT reach target {TARGET_BRIER} — player signal too narrow (std {p_raw.std():.4f})")


# ── 4. market-only Platt (1-D) — reference ceiling ───────────────────────────

print(f"\n  [B] Platt Scaling — market logit only  (1 input, reference)")
logits_market = df['logit_market_prob'].values.reshape(-1, 1)

p_platt_mkt = np.zeros(n)
for tr, va in skf.split(logits_market, y):
    clf = LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000)
    clf.fit(logits_market[tr], y[tr])
    p_platt_mkt[va] = clf.predict_proba(logits_market[va])[:, 1]

brier_mkt_cv = brier(p_platt_mkt, y)
print(f"  Brier (5-fold CV) : {brier_mkt_cv:.4f}  (improvement: {brier_raw - brier_mkt_cv:+.4f})")


# ── 5. blended Platt (2-D) — production calibrator ───────────────────────────
# Two inputs: [logit(p_player), logit_market].
# C=3.5 matches the production model regularisation strength; prevents
# over-fitting the player logit (which has very small variance).

print(f"\n  [C] Blended Platt — [player logit, market logit]  (2 inputs)  *PRODUCTION*")
X_blend = np.column_stack([logits_player, logits_market])

# Standardise so the small-variance player column isn't drowned out
scaler  = StandardScaler()
X_blend_s = scaler.fit_transform(X_blend)

p_platt_blend = np.zeros(n)
for tr, va in skf.split(X_blend_s, y):
    scl_cv = StandardScaler().fit(X_blend[tr])
    clf = LogisticRegression(C=3.5, solver='lbfgs', max_iter=2000)
    clf.fit(scl_cv.transform(X_blend[tr]), y[tr])
    p_platt_blend[va] = clf.predict_proba(scl_cv.transform(X_blend[va]))[:, 1]

brier_blend_cv = brier(p_platt_blend, y)
print(f"  Brier (5-fold CV) : {brier_blend_cv:.4f}  (improvement: {brier_raw - brier_blend_cv:+.4f})")

if brier_blend_cv < TARGET_BRIER:
    print(f"  ** Meets target (< {TARGET_BRIER})")
else:
    print(f"  ** Does NOT meet target {TARGET_BRIER}")

print()
reliability_summary(p_platt_blend, y)


# ── 6. summary table ──────────────────────────────────────────────────────────

print(f"\n  {'Approach':<42}  {'Brier (CV)':>10}  {'vs raw':>8}")
print(f"  {'-'*64}")
rows = [
    ("Raw player-only (no calibration)",   brier_raw,        0.0),
    ("[A] Platt — player logit only",       brier_platt_1d,   brier_raw - brier_platt_1d),
    ("[B] Platt — market logit only",       brier_mkt_cv,     brier_raw - brier_mkt_cv),
    ("[C] Blended Platt — player + market", brier_blend_cv,   brier_raw - brier_blend_cv),
]
for label, b, imp in rows:
    target_flag = '  <-- TARGET MET' if b < TARGET_BRIER else ''
    print(f"  {label:<42}  {b:>10.4f}  {imp:>+8.4f}{target_flag}")


# ── 7. fit final in-sample calibrator (whichever won) & save ─────────────────

# Determine best approach
best_label, best_brier = min(
    [('player', brier_platt_1d), ('market', brier_mkt_cv), ('blend', brier_blend_cv)],
    key=lambda x: x[1]
)

print(f"\n  Best approach: [{best_label}]  Brier {best_brier:.4f}")
print(f"  Fitting final calibrator on all {n} games ...")

if best_label == 'player':
    X_final = logits_player
    scaler_final = None
elif best_label == 'market':
    X_final = logits_market
    scaler_final = None
else:
    scaler_final = StandardScaler().fit(X_blend)
    X_final = scaler_final.transform(X_blend)

clf_final = LogisticRegression(C=3.5, solver='lbfgs', max_iter=2000)
clf_final.fit(X_final, y)
p_is = clf_final.predict_proba(X_final)[:, 1]
print(f"  In-sample Brier   : {brier(p_is, y):.4f}")
print(f"  Coefficients      : {clf_final.coef_[0]}   intercept: {clf_final.intercept_[0]:.4f}")

os.makedirs('models', exist_ok=True)
payload = {
    'calibrator':    clf_final,
    'scaler':        scaler_final,   # None for 1-D approaches
    'input_type':    best_label,     # 'player' | 'market' | 'blend'
    'brier_raw':     brier_raw,
    'brier_cal_cv':  best_brier,
    'brier_cal_is':  brier(p_is, y),
    'n_games':       n,
    'date_range':    f"{date_min} -> {date_max}",
}
joblib.dump(payload, SAVE_PATH)
print(f"\n  Saved -> {SAVE_PATH}")
print(f"{'='*70}\n")
