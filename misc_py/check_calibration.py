"""
Calibration diagnostic for 2026 out-of-sample predictions.

Requires cache/calib_2026.pkl, which is written automatically when
backtest.py runs with RUN_MODE='2026_validation'.

Outputs:
  - Brier Score (raw model and post-Platt calibration)
  - Reliability curve in 5% probability buckets
  - Feature variance comparison: training scaler vs 2026 data
  - C-parameter sweep: Brier + Sniper ROI for C in [0.3, 0.5, 1.0, 1.5, 2.0, 3.5]
"""

import os
import sys
import pickle
import numpy as np

CACHE_PATH    = 'cache/calib_2026.pkl'
LOG5_FEATURES = ['logit_market_prob', 'd_log5_woba', 'd_log5_xwoba', 'd_xfip']
EDGE_MIN      = 0.045   # Sniper threshold


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _de_vig(hml, aml):
    raw_h = (100 / (hml + 100)) if hml > 0 else (abs(hml) / (abs(hml) + 100))
    raw_a = (100 / (aml + 100)) if aml > 0 else (abs(aml) / (abs(aml) + 100))
    tot   = raw_h + raw_a
    return raw_h / tot, raw_a / tot


def _kelly_stake(bankroll, ml, prob, fraction=0.5, cap=0.15):
    dec = (1 + ml / 100) if ml > 0 else (1 + 100 / abs(ml))
    b   = dec - 1
    if b <= 0:
        return 0.0
    fk = (b * prob - (1 - prob)) / b
    stk = bankroll * min(fraction * fk, cap)
    return round(max(stk, 0.0), 2)


def sniper_roi(feat_df, p_arr):
    """Run Sniper simulation; return (n_bets, n_wins, wagered, profit)."""
    bets = wins = 0
    wagered = profit = 0.0
    bankroll = 1000.0
    for i in range(len(feat_df)):
        row  = feat_df.iloc[i]
        hml  = float(row['fd_home_ml'])
        aml  = float(row['fd_away_ml'])
        hp   = float(p_arr[i])
        ap   = 1.0 - hp
        hm, am = _de_vig(hml, aml)
        he = hp - hm
        ae = ap - am
        bt_ml = bt_p = bt_won = None
        if he > EDGE_MIN:
            bt_ml, bt_p, bt_won = hml, hp, (int(row['y']) == 1)
        elif ae > EDGE_MIN:
            bt_ml, bt_p, bt_won = aml, ap, (int(row['y']) == 0)
        if bt_ml is None:
            continue
        stk = _kelly_stake(bankroll, bt_ml, bt_p)
        if stk <= 0:
            continue
        bets    += 1
        wagered += stk
        if bt_won:
            wins += 1
            net   = stk * (bt_ml / 100) if bt_ml > 0 else stk * (100 / abs(bt_ml))
            profit   += net
            bankroll += net
        else:
            profit   -= stk
            bankroll  = max(bankroll - stk, 1.0)
    return bets, wins, wagered, profit


def reliability_curve(p_pred, y_true, label=''):
    combined_p = np.concatenate([p_pred, 1.0 - p_pred])
    combined_y = np.concatenate([y_true.astype(float), 1.0 - y_true.astype(float)])
    tag = f' ({label})' if label else ''
    print(f"\n  Reliability Curve{tag}  -- symmetric, both sides combined")
    print(f"  {'Bin':>12}  {'N':>6}  {'Avg Pred':>9}  {'Actual%':>9}  {'Gap':>8}")
    print(f"  {'-'*50}")
    for lo in np.arange(0.45, 0.80, 0.05):
        hi   = lo + 0.05
        mask = (combined_p >= lo) & (combined_p < hi)
        n    = int(mask.sum())
        if n == 0:
            continue
        pred = float(combined_p[mask].mean())
        act  = float(combined_y[mask].mean())
        diff = act - pred
        flag = '  OVER-CONF' if diff < -0.04 else ('  under-conf' if diff > 0.04 else '')
        print(f"  [{lo:.0%}-{hi:.0%})  {n:>6}  {pred:>9.3f}  {act:>9.3f}  {diff:>+8.3f}{flag}")


def main():
    if not os.path.exists(CACHE_PATH):
        print(f"\nERROR: {CACHE_PATH} not found.")
        print("Generate it by running:  PYTHONIOENCODING=utf-8 python backtest.py")
        print("(backtest.py must be set to RUN_MODE='2026_validation')")
        sys.exit(1)

    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)

    feat_df  = cache['log5_2026']
    p_raw    = cache['p_home_raw']
    mu_mdl   = np.array(cache['mu'])     # training scaler: mean of non-market features
    sig_mdl  = np.array(cache['sig'])    # training scaler: std of non-market features
    y_true   = feat_df['y'].values.astype(int)
    n_games  = len(feat_df)
    C_base   = cache.get('C_base', 3.5)

    print(f"\n{'='*70}")
    print(f"  CALIBRATION DIAGNOSTIC  —  2026 out-of-sample")
    print(f"  {n_games} games  |  C_base={C_base}")
    print(f"{'='*70}")

    # ── Raw Brier score ────────────────────────────────────────────────────────
    brier_raw = float(np.mean((p_raw - y_true) ** 2))
    print(f"\n  Brier Score (raw C={C_base}): {brier_raw:.4f}")
    print(f"  Reference: coin flip = 0.2500  |  typical MLB model = 0.23-0.24")

    reliability_curve(p_raw, y_true, label=f'raw C={C_base}')

    # ── Feature variance: training scaler vs 2026 data ────────────────────────
    X_26 = feat_df[LOG5_FEATURES].values.astype(float)

    print(f"\n  Feature Distribution — Training Scaler vs 2026 Test Data")
    print(f"  {'Feature':>22}  {'Train mu':>9}  {'Train std':>10}  "
          f"{'2026 mu':>9}  {'2026 std':>10}  {'Std Ratio':>10}")
    print(f"  {'-'*76}")
    for i, feat in enumerate(LOG5_FEATURES):
        vals = X_26[:, i]
        if i == 0:
            # logit_market_prob is NOT standardized (col 0)
            print(f"  {feat:>22}  {'(anchor, not scaled)':>35}  "
                  f"{vals.mean():>9.4f}  {vals.std():>10.4f}  {'—':>10}")
        else:
            mu_tr  = mu_mdl[i - 1]
            sig_tr = sig_mdl[i - 1]
            mu_26  = vals.mean()
            sig_26 = vals.std()
            ratio  = sig_26 / sig_tr if sig_tr > 0 else float('nan')
            flag   = '  ! HIGH' if ratio > 1.3 else ('  ! LOW' if ratio < 0.7 else '')
            print(f"  {feat:>22}  {mu_tr:>9.4f}  {sig_tr:>10.4f}  "
                  f"{mu_26:>9.4f}  {sig_26:>10.4f}  {ratio:>10.2f}x{flag}")

    # ── Platt calibration + C sweep (requires training features in cache) ─────
    if 'log5_train' not in cache:
        print("\n  NOTE: training features not in cache — re-run backtest.py to enable")
        print("  Platt calibration and C-sweep analysis.\n")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    log5_train  = cache['log5_train']
    X_tr_raw    = log5_train[LOG5_FEATURES].values.astype(float)
    y_tr        = log5_train['y'].values.astype(int)
    mu_tr       = np.array(cache.get('mu_train', X_tr_raw[:, 1:].mean(axis=0)))
    sig_tr      = np.array(cache.get('sig_train', X_tr_raw[:, 1:].std(axis=0) + 1e-8))
    X_tr_s      = np.column_stack([X_tr_raw[:, 0], (X_tr_raw[:, 1:] - mu_tr) / sig_tr])
    X_26_s      = np.column_stack([X_26[:, 0], (X_26[:, 1:] - mu_tr) / sig_tr])

    print(f"\n  Training set: {len(log5_train):,} games (2021-2025)")

    # 5-fold OOF for Platt calibration at C=3.5
    print("  Fitting Platt calibrator (5-fold OOF on training data)...")
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_tr))
    for tr_i, va_i in skf.split(X_tr_s, y_tr):
        cv = LogisticRegression(C=3.5, solver='lbfgs', max_iter=1000)
        cv.fit(X_tr_s[tr_i], y_tr[tr_i])
        oof_probs[va_i] = cv.predict_proba(X_tr_s[va_i])[:, 1]

    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(_logit(oof_probs).reshape(-1, 1), y_tr)

    p_cal     = platt.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
    brier_cal = float(np.mean((p_cal - y_true) ** 2))
    print(f"\n  Brier Score (C=3.5 + Platt):  {brier_cal:.4f}")
    reliability_curve(p_cal, y_true, label='C=3.5 + Platt')

    # ── C parameter sweep ─────────────────────────────────────────────────────
    print(f"\n  C-Parameter Sweep  (train 2021-2025 → test 2026,  Sniper edge > 4.5%)")
    print(f"  {'C':>5}  {'Brier':>8}  {'Bets':>6}  {'W/L':>8}  {'ROI':>9}  {'P&L':>10}")
    print(f"  {'-'*56}")

    for C_val in [0.3, 0.5, 1.0, 1.5, 2.0, 3.5]:
        sw_m = LogisticRegression(C=C_val, solver='lbfgs', max_iter=1000)
        sw_m.fit(X_tr_s, y_tr)
        p_sw      = sw_m.predict_proba(X_26_s)[:, 1]
        brier_sw  = float(np.mean((p_sw - y_true) ** 2))
        nb, nw, wag, pnl = sniper_roi(feat_df, p_sw)
        roi = pnl / wag * 100 if wag > 0 else 0.0
        wl  = f"{nw}/{nb - nw}"
        mark = '  <-- current' if C_val == C_base else ''
        print(f"  {C_val:>5.1f}  {brier_sw:.4f}  {nb:>6}  {wl:>8}  {roi:>+9.1f}%  ${pnl:>+9.2f}{mark}")

    print()


if __name__ == '__main__':
    main()
