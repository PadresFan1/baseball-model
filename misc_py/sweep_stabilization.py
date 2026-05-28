#!/usr/bin/env python3
"""
misc_py/sweep_stabilization.py

Walk-forward stabilization sweep to prevent overfitting player-stabilization
parameters to the current 2026 season.

Architecture
------------
  Step 1 (Optimize)  : Tune on 2025 fold — train 2021-2024, test 2025.
  Step 2 (Validate)  : Top 1-2 configs run blindly on all usable 2026 games.

Approaches tested
-----------------
  Baseline   : No stabilization (current production behavior).
  Approach A : Hard PA/IP cutoff — 50% deviation compression below threshold.
  Approach B : Dynamic Bayesian blending — W * rolling_100pa + (1-W) * baseline.
               Baseline = cumulative season stats (or prior-year snapshot if < 50 PA).

Usage
-----
  PYTHONIOENCODING=utf-8 python misc_py/sweep_stabilization.py
  Runtime: ~10 min first run (builds caches), ~5 min on subsequent runs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# Approach A: Hard PA/IP cutoff — 50% deviation compression below floor
APPROACH_A_CONFIGS = [
    {'name': 'A-100/15',  'approach': 'A', 'pa_floor': 100, 'ip_floor': 15},
    {'name': 'A-150/25',  'approach': 'A', 'pa_floor': 150, 'ip_floor': 25},
    {'name': 'A-200/45',  'approach': 'A', 'pa_floor': 200, 'ip_floor': 45},
    {'name': 'A-250/60',  'approach': 'A', 'pa_floor': 250, 'ip_floor': 60},
]

# Approach B: Dynamic Bayesian weight schedules
# bat_tiers / pit_tiers: [(vol_lo, vol_hi_exclusive, recent_form_weight), ...]
APPROACH_B_CONFIGS = [
    {
        'name': 'B-Conservative',
        'approach': 'B',
        'bat_tiers': [(0, 50, 0.05), (50, 100, 0.15), (100, 200, 0.35), (200, 9999, 0.50)],
        'pit_tiers': [(0, 10, 0.05), (10, 20, 0.15), (20, 40, 0.35), (40, 9999, 0.50)],
    },
    {
        'name': 'B-Balanced',
        'approach': 'B',
        'bat_tiers': [(0, 50, 0.10), (50, 100, 0.30), (100, 200, 0.60), (200, 9999, 1.00)],
        'pit_tiers': [(0, 10, 0.10), (10, 20, 0.30), (20, 40, 0.60), (40, 9999, 1.00)],
    },
    {
        'name': 'B-Aggressive',
        'approach': 'B',
        'bat_tiers': [(0, 50, 0.25), (50, 100, 0.50), (100, 200, 0.75), (200, 9999, 1.00)],
        'pit_tiers': [(0, 10, 0.25), (10, 20, 0.50), (20, 40, 0.75), (40, 9999, 1.00)],
    },
]

ALL_CONFIGS = (
    [{'name': 'Baseline', 'approach': 'none'}]
    + APPROACH_A_CONFIGS
    + APPROACH_B_CONFIGS
)

# ─────────────────────────────────────────────────────────────────────────────
# STABILIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _tier_weight(vol, tiers):
    """Return recent-form weight for player volume (PA or IP)."""
    for lo, hi, w in tiers:
        if lo <= vol < hi:
            return w
    return tiers[-1][2]


def _stabilize_bat(raw, lg_avg, ext_entry, metric, config):
    """
    Apply stabilization to one batter rolling stat before the Log5 formula.
    raw      : value from batter_cache (rolling 100-PA window)
    lg_avg   : league average for this date/hand
    ext_entry: dict from extended batter lookup, may be None
    metric   : 'woba' or 'xwoba'
    config   : entry from ALL_CONFIGS
    """
    approach = config['approach']
    if approach == 'none' or ext_entry is None:
        return raw

    season_pa = ext_entry.get('season_pa', 9999)

    if approach == 'A':
        if season_pa < config['pa_floor']:
            return lg_avg + 0.5 * (raw - lg_avg)
        return raw

    if approach == 'B':
        w = _tier_weight(season_pa, config['bat_tiers'])
        if season_pa >= 50:
            baseline = ext_entry.get(f'season_{metric}')
        else:
            baseline = ext_entry.get(f'prior_{metric}')
        if baseline is None:
            baseline = lg_avg
        return w * raw + (1.0 - w) * baseline

    return raw


def _stabilize_pit_xfip(raw, lg_xfip, ext_entry, config):
    """Apply stabilization to pitcher rolling xFIP."""
    approach = config['approach']
    if approach == 'none' or ext_entry is None:
        return raw

    season_ip = ext_entry.get('season_ip', 9999)

    if approach == 'A':
        if season_ip < config['ip_floor']:
            return lg_xfip + 0.5 * (raw - lg_xfip)
        return raw

    if approach == 'B':
        w = _tier_weight(season_ip, config['pit_tiers'])
        if season_ip >= 10:
            baseline = ext_entry.get('season_xfip')
        else:
            baseline = ext_entry.get('prior_xfip')
        if baseline is None:
            baseline = lg_xfip
        return w * raw + (1.0 - w) * baseline

    return raw


def _stabilize_pit_xwoba(raw, lg_xwoba_pit, ext_entry, config):
    """Apply stabilization to pitcher rolling xwOBA allowed."""
    approach = config['approach']
    if approach == 'none' or ext_entry is None:
        return raw

    season_ip = ext_entry.get('season_ip', 9999)

    if approach == 'A':
        if season_ip < config['ip_floor']:
            return lg_xwoba_pit + 0.5 * (raw - lg_xwoba_pit)
        return raw

    if approach == 'B':
        w = _tier_weight(season_ip, config['pit_tiers'])
        if season_ip >= 10:
            baseline = ext_entry.get('season_xwoba_pit')
        else:
            baseline = ext_entry.get('prior_xwoba_pit')
        if baseline is None:
            baseline = lg_xwoba_pit
        return w * raw + (1.0 - w) * baseline

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED CACHE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_extended_batter_lookup(batter_data_by_hand, target_dates):
    """
    For every (player_id, hand, date) build season-level stats alongside
    the standard rolling 100-PA cache.

    Returns
    -------
    ext_bat : {(player_id, hand, date): {
        'season_pa'   : int   — PA accumulated in the current calendar year
        'season_woba' : float — PA-weighted cumulative wOBA this season
        'season_xwoba': float — PA-weighted cumulative xwOBA this season
        'prior_woba'  : float|None — rolling 100-PA snapshot at end of prior season
        'prior_xwoba' : float|None — same
    }}
    """
    ext = {}
    target_dates = sorted(target_dates)

    for hand, df in batter_data_by_hand.items():
        if df.empty:
            continue

        df = df.sort_values(['player_id', 'game_date'])

        for player_id, grp in df.groupby('player_id'):
            games = (grp.sort_values('game_date')
                       [['game_date', 'pa', 'woba', 'xwoba']]
                       .to_dict('records'))
            n = len(games)
            if n == 0:
                continue

            gd = [g['game_date'] for g in games]

            # Global prefix sums (across all seasons)
            cum_pa   = [0] * (n + 1)
            cum_wpa  = [0.0] * (n + 1)
            cum_xwpa = [0.0] * (n + 1)
            for i, g in enumerate(games):
                pa_i = int(g['pa'])
                cum_pa[i+1]   = cum_pa[i]   + pa_i
                cum_wpa[i+1]  = cum_wpa[i]  + float(g['woba'])  * pa_i
                cum_xwpa[i+1] = cum_xwpa[i] + float(g['xwoba']) * pa_i

            game_idx = 0  # sweep pointer

            for target_date in target_dates:
                while game_idx < n and gd[game_idx] < target_date:
                    game_idx += 1
                k = game_idx
                if k == 0:
                    continue

                # ── Season start (Jan 1 of target year) ──────────────────
                season_start = target_date[:4] + '-01-01'
                lo, hi = 0, k
                while lo < hi:
                    mid = (lo + hi) // 2
                    if gd[mid] < season_start:
                        lo = mid + 1
                    else:
                        hi = mid
                s_idx = lo  # index of first game in current season

                season_pa = cum_pa[k] - cum_pa[s_idx]
                if season_pa == 0:
                    continue

                season_woba  = (cum_wpa[k]  - cum_wpa[s_idx])  / season_pa
                season_xwoba = (cum_xwpa[k] - cum_xwpa[s_idx]) / season_pa

                # ── Prior-year rolling snapshot (last 100 PA before season) ─
                prior_woba = prior_xwoba = None
                if s_idx > 0:
                    prior_total = cum_pa[s_idx]
                    if prior_total >= 30:
                        win = min(prior_total, 100)
                        thresh = prior_total - win
                        lo2, hi2 = 0, s_idx
                        while lo2 < hi2:
                            mid2 = (lo2 + hi2) // 2
                            if cum_pa[mid2] <= thresh:
                                lo2 = mid2 + 1
                            else:
                                hi2 = mid2
                        j = lo2
                        pa_w = cum_pa[s_idx] - cum_pa[j]
                        if pa_w > 0:
                            prior_woba  = (cum_wpa[s_idx]  - cum_wpa[j])  / pa_w
                            prior_xwoba = (cum_xwpa[s_idx] - cum_xwpa[j]) / pa_w

                ext[(player_id, hand, target_date)] = {
                    'season_pa':    season_pa,
                    'season_woba':  season_woba,
                    'season_xwoba': season_xwoba,
                    'prior_woba':   prior_woba,
                    'prior_xwoba':  prior_xwoba,
                }

    print(f"  Extended batter lookup: {len(ext):,} entries")
    return ext


def build_extended_pitcher_lookup(pitcher_df, fip_const_by_season, target_dates):
    """
    For every (player_id, date) build cumulative season xFIP/xwOBA alongside
    the standard rolling 100-BF cache.

    Returns
    -------
    ext_pit : {(player_id, date): {
        'season_ip':        float — IP accumulated in the current calendar year
        'season_xfip':      float|None — season-to-date xFIP
        'season_xwoba_pit': float|None — season-to-date xwOBA allowed
        'prior_xfip':       float|None — rolling 100-BF xFIP at end of prior season
        'prior_xwoba_pit':  float|None — same
    }}
    """
    if pitcher_df.empty:
        return {}

    target_dates = sorted(target_dates)

    # ── League HR/FB rate per date (cumulative across all events) ─────────────
    events = []
    for _, row in pitcher_df.iterrows():
        try:
            hr = int(row.get('hr', 0) or 0)
            fb = int(row.get('fb', 0) or 0)
            events.append((row['game_date'], hr, fb + hr))
        except (ValueError, TypeError):
            continue
    events.sort(key=lambda x: x[0])

    cum_hr_l = cum_fb_l = ev_i = 0
    lg_hr_fb_by_date = {}
    for td in target_dates:
        while ev_i < len(events) and events[ev_i][0] < td:
            cum_hr_l += events[ev_i][1]
            cum_fb_l += events[ev_i][2]
            ev_i += 1
        lg_hr_fb_by_date[td] = cum_hr_l / cum_fb_l if cum_fb_l > 0 else 0.115

    ext = {}
    pitcher_df = pitcher_df.sort_values(['player_id', 'game_date'])

    def _i(v):
        return int(v) if v is not None and not pd.isna(v) else 0

    def _f(v):
        return float(v) if v is not None and not pd.isna(v) else 0.0

    for player_id, grp in pitcher_df.groupby('player_id'):
        grp  = grp.sort_values('game_date')
        keep = ['game_date', 'pa', 'xwoba_against', 'k', 'bb', 'hr', 'hbp', 'fb', 'ip']
        avail = [c for c in keep if c in grp.columns]
        games = grp[avail].to_dict('records')
        n = len(games)
        if n == 0:
            continue

        gd = [g['game_date'] for g in games]

        cum_pa   = [0]   * (n + 1)
        cum_xwpa = [0.0] * (n + 1)
        cum_k    = [0]   * (n + 1)
        cum_bb   = [0]   * (n + 1)
        cum_hr   = [0]   * (n + 1)
        cum_hbp  = [0]   * (n + 1)
        cum_fb   = [0.0] * (n + 1)
        cum_ip   = [0.0] * (n + 1)

        from backtest import ip_to_float as _ip2f
        for i, g in enumerate(games):
            pa_i = _i(g['pa'])
            cum_pa[i+1]   = cum_pa[i]   + pa_i
            cum_xwpa[i+1] = cum_xwpa[i] + _f(g['xwoba_against']) * pa_i
            cum_k[i+1]    = cum_k[i]    + _i(g['k'])
            cum_bb[i+1]   = cum_bb[i]   + _i(g['bb'])
            cum_hr[i+1]   = cum_hr[i]   + _i(g['hr'])
            cum_hbp[i+1]  = cum_hbp[i]  + _i(g['hbp'])
            cum_fb[i+1]   = cum_fb[i]   + _f(g['fb'])
            raw_ip = g.get('ip')
            cum_ip[i+1]   = cum_ip[i]   + (_ip2f(raw_ip)
                                            if raw_ip is not None and not pd.isna(raw_ip)
                                            else 0.0)

        game_idx = 0

        for td in target_dates:
            while game_idx < n and gd[game_idx] < td:
                game_idx += 1
            k = game_idx
            if k == 0:
                continue

            season_yr    = td[:4]
            season_start = season_yr + '-01-01'
            fip_c        = fip_const_by_season.get(int(season_yr), 3.20)

            lo, hi = 0, k
            while lo < hi:
                mid = (lo + hi) // 2
                if gd[mid] < season_start:
                    lo = mid + 1
                else:
                    hi = mid
            s_idx = lo

            season_pa = cum_pa[k] - cum_pa[s_idx]
            season_ip = cum_ip[k] - cum_ip[s_idx]
            if season_pa < 10 or season_ip <= 0:
                continue

            # Season-to-date xwOBA allowed
            season_xwoba = (cum_xwpa[k] - cum_xwpa[s_idx]) / season_pa

            # Season-to-date xFIP
            k_s   = cum_k[k]   - cum_k[s_idx]
            bb_s  = cum_bb[k]  - cum_bb[s_idx]
            hr_s  = cum_hr[k]  - cum_hr[s_idx]
            hbp_s = cum_hbp[k] - cum_hbp[s_idx]
            fb_s  = cum_fb[k]  - cum_fb[s_idx]
            lg_hf = lg_hr_fb_by_date.get(td, 0.115)
            xfip_num = 13 * (lg_hf * fb_s) + 3 * (bb_s + hbp_s) - 2 * k_s
            season_xfip = (xfip_num / season_ip + fip_c) if season_ip > 0 else None

            # Prior-year rolling snapshot (last 100 BF before this season)
            prior_xfip = prior_xwoba_pit = None
            if s_idx > 0:
                prior_total = cum_pa[s_idx]
                if prior_total >= 20:
                    win = min(prior_total, 100)
                    thresh = prior_total - win
                    lo2, hi2 = 0, s_idx
                    while lo2 < hi2:
                        mid2 = (lo2 + hi2) // 2
                        if cum_pa[mid2] <= thresh:
                            lo2 = mid2 + 1
                        else:
                            hi2 = mid2
                    j = lo2
                    pa_w = cum_pa[s_idx] - cum_pa[j]
                    ip_w = cum_ip[s_idx] - cum_ip[j]
                    if pa_w > 0:
                        prior_xwoba_pit = (cum_xwpa[s_idx] - cum_xwpa[j]) / pa_w
                    if ip_w > 0:
                        prior_yr  = gd[s_idx - 1][:4]
                        fip_c_p   = fip_const_by_season.get(int(prior_yr), 3.20)
                        k_p   = cum_k[s_idx]   - cum_k[j]
                        bb_p  = cum_bb[s_idx]  - cum_bb[j]
                        hr_p  = cum_hr[s_idx]  - cum_hr[j]
                        hbp_p = cum_hbp[s_idx] - cum_hbp[j]
                        fb_p  = cum_fb[s_idx]  - cum_fb[j]
                        xfip_p = (13 * (0.115 * fb_p) + 3 * (bb_p + hbp_p) - 2 * k_p)
                        prior_xfip = (xfip_p / ip_w + fip_c_p) if ip_w > 0 else None

            ext[(player_id, td)] = {
                'season_ip':        season_ip,
                'season_xfip':      season_xfip,
                'season_xwoba_pit': season_xwoba,
                'prior_xfip':       prior_xfip,
                'prior_xwoba_pit':  prior_xwoba_pit,
            }

    print(f"  Extended pitcher lookup: {len(ext):,} entries")
    return ext


# ─────────────────────────────────────────────────────────────────────────────
# STABILIZED FEATURE COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

def collect_log5_features_stabilized(
    seasons, lineup_cache,
    batter_cache, lg_bat_avg,
    pitcher_cache, lg_pit_avg,
    ext_bat, ext_pit,
    config,
    odds_df_ref,
):
    """
    Mirrors collect_game_features_log5 from backtest.py but applies the
    chosen stabilization transform to every player stat before the Log5
    matchup formula.
    """
    from backtest import (
        normalize_team, TEAM_MAP, de_vig_probs,
        log5_matchup, aggregate_lineup_metric,
    )

    # Build (date, home_fg, away_fg) → lineup data index
    lineup_by_teams = {}
    for sc in lineup_cache.values():
        for gdata in sc.values():
            key = (gdata['date'], gdata['home_fg'], gdata['away_fg'])
            lineup_by_teams.setdefault(key, []).append(gdata)

    records = []
    miss_lu = miss_st = miss_lg = 0

    for season in seasons:
        s_odds = odds_df_ref[odds_df_ref['date'].str.startswith(str(season))]

        for _, game in s_odds.iterrows():
            date      = game['date']
            home_team = normalize_team(game['home_team'])
            away_team = normalize_team(game['away_team'])
            if home_team not in TEAM_MAP or away_team not in TEAM_MAP:
                continue

            hs = game.get('home_score')
            as_ = game.get('away_score')
            if pd.isna(hs) or pd.isna(as_):
                continue

            hml = game.get('fd_home_ml')
            aml = game.get('fd_away_ml')
            if pd.isna(hml) or pd.isna(aml):
                continue
            hml, aml = float(hml), float(aml)
            if hml == 0 or aml == 0:
                continue

            gdl = lineup_by_teams.get((date, home_team, away_team))
            if not gdl:
                miss_lu += 1
                continue
            gd = gdl[0]

            hl   = [int(p) for p in gd['home_lineup']]
            al   = [int(p) for p in gd['away_lineup']]
            h_sp = int(gd['home_starter_id'])
            a_sp = int(gd['away_starter_id'])
            hvh  = gd['away_starter_hand']   # hand home lineup faces
            hva  = gd['home_starter_hand']   # hand away lineup faces

            h_pit_raw = pitcher_cache.get((h_sp, date))
            a_pit_raw = pitcher_cache.get((a_sp, date))
            if h_pit_raw is None or a_pit_raw is None:
                miss_st += 1
                continue

            lg_h = lg_bat_avg.get((hvh, date))
            lg_a = lg_bat_avg.get((hva, date))
            lg_p = lg_pit_avg.get(date)
            if lg_h is None or lg_a is None or lg_p is None:
                miss_lg += 1
                continue

            lg_woba_h  = lg_h['woba']  or 0.320
            lg_xwoba_h = lg_h['xwoba'] or 0.315
            lg_woba_a  = lg_a['woba']  or 0.320
            lg_xwoba_a = lg_a['xwoba'] or 0.315
            lg_xfip      = lg_p['xfip']
            lg_xwoba_pit = lg_p['xwoba_pit']

            # ── Stabilize pitcher stats ────────────────────────────────────
            h_ext_p = ext_pit.get((h_sp, date))
            a_ext_p = ext_pit.get((a_sp, date))

            h_xfip  = _stabilize_pit_xfip(h_pit_raw['xfip'],      lg_xfip,      h_ext_p, config)
            a_xfip  = _stabilize_pit_xfip(a_pit_raw['xfip'],      lg_xfip,      a_ext_p, config)
            h_xwoba = _stabilize_pit_xwoba(h_pit_raw['xwoba_pit'], lg_xwoba_pit, h_ext_p, config)
            a_xwoba = _stabilize_pit_xwoba(a_pit_raw['xwoba_pit'], lg_xwoba_pit, a_ext_p, config)

            # ── Home batting vs away pitcher — Log5 per batter ────────────
            h_l5w = {}; h_l5x = {}
            for pid in hl:
                entry = batter_cache.get((pid, hvh, date), {})
                bw  = entry.get('woba')
                bx  = entry.get('xwoba')
                ext = ext_bat.get((pid, hvh, date))
                if bw is not None:
                    bw = _stabilize_bat(bw, lg_woba_h,  ext, 'woba',  config)
                    h_l5w[pid] = log5_matchup(bw, a_xwoba, lg_woba_h)
                if bx is not None:
                    bx = _stabilize_bat(bx, lg_xwoba_h, ext, 'xwoba', config)
                    h_l5x[pid] = log5_matchup(bx, a_xwoba, lg_xwoba_h)

            # ── Away batting vs home pitcher — Log5 per batter ────────────
            a_l5w = {}; a_l5x = {}
            for pid in al:
                entry = batter_cache.get((pid, hva, date), {})
                bw  = entry.get('woba')
                bx  = entry.get('xwoba')
                ext = ext_bat.get((pid, hva, date))
                if bw is not None:
                    bw = _stabilize_bat(bw, lg_woba_a,  ext, 'woba',  config)
                    a_l5w[pid] = log5_matchup(bw, h_xwoba, lg_woba_a)
                if bx is not None:
                    bx = _stabilize_bat(bx, lg_xwoba_a, ext, 'xwoba', config)
                    a_l5x[pid] = log5_matchup(bx, h_xwoba, lg_xwoba_a)

            # Fallbacks for batters with no data
            fb_hw  = log5_matchup(lg_woba_h,  a_xwoba, lg_woba_h)
            fb_hx  = log5_matchup(lg_xwoba_h, a_xwoba, lg_xwoba_h)
            fb_aw  = log5_matchup(lg_woba_a,  h_xwoba, lg_woba_a)
            fb_ax  = log5_matchup(lg_xwoba_a, h_xwoba, lg_xwoba_a)

            h_log5_w = aggregate_lineup_metric(h_l5w, hl, fb_hw)
            h_log5_x = aggregate_lineup_metric(h_l5x, hl, fb_hx)
            a_log5_w = aggregate_lineup_metric(a_l5w, al, fb_aw)
            a_log5_x = aggregate_lineup_metric(a_l5x, al, fb_ax)

            h_xfip_r = lg_xfip / h_xfip if h_xfip > 0 else 1.0
            a_xfip_r = lg_xfip / a_xfip if a_xfip > 0 else 1.0

            p_dv, _ = de_vig_probs(hml, aml)
            p_clip  = float(np.clip(p_dv, 0.001, 0.999))
            logit_m = float(np.log(p_clip / (1.0 - p_clip)))

            records.append({
                'date':              date,
                'season':            season,
                'home_team':         home_team,
                'away_team':         away_team,
                'logit_market_prob': logit_m,
                'd_log5_woba':       h_log5_w - a_log5_w,
                'd_log5_xwoba':      h_log5_x - a_log5_x,
                'd_xfip':            h_xfip_r - a_xfip_r,
                'y':                 1 if float(hs) > float(as_) else 0,
                'fd_home_ml':        hml,
                'fd_away_ml':        aml,
            })

    print(f"    {len(records):,} features  "
          f"(miss: lineup={miss_lu}, starter={miss_st}, lg_avg={miss_lg})")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# KELLY SIMULATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _sim_strategy(p_home_arr, feat_df, edge_min, start_br=1000.0):
    """Half-Kelly simulation. Returns metrics dict."""
    from backtest import de_vig_probs, _kelly_stake_bt
    bets     = []
    bankroll = start_br
    for i in range(len(feat_df)):
        row = feat_df.iloc[i]
        hp  = float(p_home_arr[i])
        ap  = 1.0 - hp
        hml = float(row['fd_home_ml'])
        aml = float(row['fd_away_ml'])
        hm, am = de_vig_probs(hml, aml)
        he, ae = hp - hm, ap - am
        y = int(row['y'])

        bet_ml = bet_p = bet_won = None
        if he > edge_min:
            bet_ml, bet_p, bet_won = hml, hp, (y == 1)
        elif ae > edge_min:
            bet_ml, bet_p, bet_won = aml, ap, (y == 0)

        if bet_ml is None:
            continue

        stake = _kelly_stake_bt(bankroll, bet_ml, bet_p, 0.5, 0.15)
        if stake <= 0:
            continue

        if bet_won:
            net = stake * (bet_ml / 100.0) if bet_ml > 0 else stake * (100.0 / abs(bet_ml))
            bankroll += net
        else:
            net = -stake
            bankroll = max(bankroll - stake, 1.0)

        bets.append({'stake': stake, 'net': net, 'won': bet_won})

    if not bets:
        return {'n': 0, 'win_pct': 0.0, 'roi': 0.0, 'profit': 0.0}

    wins    = sum(1 for b in bets if b['won'])
    wagered = sum(b['stake'] for b in bets)
    profit  = sum(b['net']   for b in bets)
    return {
        'n':       len(bets),
        'win_pct': wins / len(bets) * 100,
        'roi':     profit / wagered * 100 if wagered > 0 else 0.0,
        'profit':  profit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_stabilization_sweep():
    from sklearn.linear_model import LogisticRegression
    import backtest as bt
    from backtest import (
        odds_df,
        pull_historical_lineups,
        load_player_batter_data, load_player_pitcher_data,
        build_batter_rolling_cache, build_pitcher_rolling_cache,
        load_woba_fip_constants, eval_2026_validation,
        _LOG5_FEATURES,
    )

    ALL_SEASONS  = [2021, 2022, 2023, 2024, 2025, 2026]
    TRAIN_25     = [2021, 2022, 2023, 2024]
    C_PROD       = 3.5

    # ── Step 1: Load and build all caches ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STABILIZATION SWEEP — Loading data and building caches")
    print("=" * 72)

    print("\nStep 1/6 — Loading box score lineups (2021-2026)...")
    lineup_cache = pull_historical_lineups(ALL_SEASONS)

    print("\nStep 2/6 — Loading batter data (2021-2026)...")
    batter_data = load_player_batter_data(ALL_SEASONS)

    print("\nStep 3/6 — Loading starter data (2021-2026)...")
    pitcher_data = load_player_pitcher_data(ALL_SEASONS)

    print("\nStep 4/6 — Building standard 100-PA rolling caches...")
    batter_cache, lg_bat_avg = build_batter_rolling_cache(batter_data)
    fip_consts = {s: load_woba_fip_constants('constants/woba_fip_constants.csv', s)[1]
                  for s in ALL_SEASONS}
    pitcher_cache, lg_pit_avg = build_pitcher_rolling_cache(pitcher_data, fip_consts)

    print("\nStep 5/6 — Building extended season-level lookups...")
    all_dates = sorted(set(odds_df['date'].tolist()))
    ext_bat = build_extended_batter_lookup(batter_data, all_dates)
    ext_pit = build_extended_pitcher_lookup(pitcher_data, fip_consts, all_dates)

    # ── Step 2: Collect features for every config ─────────────────────────────
    print("\n" + "=" * 72)
    print("  Step 6/6 — Collecting Log5 features for each stabilization config")
    print("=" * 72)

    feat_cache = {}
    for cfg in ALL_CONFIGS:
        name = cfg['name']
        print(f"\n  [{name}]")
        feat_df = collect_log5_features_stabilized(
            ALL_SEASONS, lineup_cache,
            batter_cache, lg_bat_avg,
            pitcher_cache, lg_pit_avg,
            ext_bat, ext_pit,
            cfg,
            odds_df_ref=odds_df,
        )
        feat_cache[name] = feat_df

    # ── Step 3: 2025 optimization fold ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 1 — 2025 OPTIMIZATION FOLD  (Train: 2021-2024 | Test: 2025)")
    print("=" * 72 + "\n")

    rows_25 = []
    for cfg in ALL_CONFIGS:
        name    = cfg['name']
        feat_df = feat_cache[name]

        tr = feat_df[feat_df['season'].isin(TRAIN_25)].reset_index(drop=True)
        te = feat_df[feat_df['season'] == 2025].reset_index(drop=True)

        if len(tr) < 100 or len(te) < 50:
            print(f"  {name}: skipped — insufficient data (tr={len(tr)}, te={len(te)})")
            continue

        X_tr = tr[_LOG5_FEATURES].values.astype(float)
        y_tr = tr['y'].values.astype(int)
        X_te = te[_LOG5_FEATURES].values.astype(float)

        mu  = X_tr[:, 1:].mean(axis=0)
        sig = X_tr[:, 1:].std(axis=0) + 1e-8
        Xtr_s = np.column_stack([X_tr[:, 0], (X_tr[:, 1:] - mu) / sig])
        Xte_s = np.column_stack([X_te[:, 0], (X_te[:, 1:] - mu) / sig])

        mdl = LogisticRegression(C=C_PROD, solver='lbfgs', max_iter=1000)
        mdl.fit(Xtr_s, y_tr)
        p25 = mdl.predict_proba(Xte_s)[:, 1]

        brier = float(np.mean((p25 - te['y'].values) ** 2))
        snp   = _sim_strategy(p25, te, 0.045)
        enf   = _sim_strategy(p25, te, 0.040)

        rows_25.append({
            'config':   name,
            'approach': cfg['approach'],
            'n_test':   len(te),
            'brier':    round(brier, 4),
            'snp_n':    snp['n'],
            'snp_w':    round(snp['win_pct'], 1),
            'snp_roi':  round(snp['roi'],     2),
            'snp_pnl':  round(snp['profit'],  2),
            'enf_n':    enf['n'],
            'enf_w':    round(enf['win_pct'], 1),
            'enf_roi':  round(enf['roi'],     2),
            'enf_pnl':  round(enf['profit'],  2),
        })

    # Sort: Sniper ROI primary, Enforcer ROI secondary
    rows_25.sort(key=lambda r: (r['snp_roi'], r['enf_roi']), reverse=True)

    # ── Print 2025 leaderboard ─────────────────────────────────────────────────
    W = [20, 5, 7, 9, 7, 9, 10, 10, 7, 9, 10]
    hdr = (f"  {'Config':<{W[0]}} {'Appr':<{W[1]}} {'Brier':>{W[2]}} | "
           f"{'Snp N':>{W[3]}} {'Win%':>{W[4]}} {'ROI':>{W[5]}} {'P&L':>{W[6]}} | "
           f"{'Enf N':>{W[7]}} {'Win%':>{W[8]}} {'ROI':>{W[9]}} {'P&L':>{W[10]}}")
    sep = "  " + "-" * (len(hdr) - 2)

    print(f"\n{'='*len(hdr)}")
    print(f"  2025 LEADERBOARD  (C={C_PROD}, ½ Kelly, $1k bankroll)")
    print(f"  Sorted by Sniper ROI ↓   |   Sniper edge >4.5%   |   Enforcer edge >4.0%")
    print('=' * len(hdr))
    print(hdr)
    print(sep)
    for r in rows_25:
        print(
            f"  {r['config']:<{W[0]}} {r['approach']:<{W[1]}} {r['brier']:>{W[2]}.4f} | "
            f"{r['snp_n']:>{W[3]}} {r['snp_w']:>{W[4]}.1f} {r['snp_roi']:>+{W[5]}.2f} ${r['snp_pnl']:>+{W[6]-1}.2f} | "
            f"{r['enf_n']:>{W[7]}} {r['enf_w']:>{W[8]}.1f} {r['enf_roi']:>+{W[9]}.2f} ${r['enf_pnl']:>+{W[10]-1}.2f}"
        )
    print(sep)
    print(f"  Baseline = current production behavior (no stabilization applied)")
    print()

    # ── Step 4: 2026 blind validation for top 2 + Baseline ────────────────────
    # Select top 2 non-Baseline configs, always include Baseline for comparison
    top_names = [r['config'] for r in rows_25 if r['config'] != 'Baseline'][:2]
    if 'Baseline' not in top_names:
        top_names.append('Baseline')

    print(f"\n{'='*72}")
    print(f"  STEP 2 — 2026 BLIND VALIDATION")
    print(f"  Configs selected: {', '.join(top_names)}")
    print(f"  Train: 2021-2025 (C={C_PROD})   |   Test: 2026 out-of-sample")
    print('=' * 72)

    for name in top_names:
        feat_df = feat_cache[name]
        tr_26 = feat_df[feat_df['season'].isin([2021, 2022, 2023, 2024, 2025])].reset_index(drop=True)
        te_26 = feat_df[feat_df['season'] == 2026].reset_index(drop=True)

        if te_26.empty:
            print(f"\n  {name}: No 2026 features — check lineups cache and player CSVs.")
            continue

        X_tr = tr_26[_LOG5_FEATURES].values.astype(float)
        y_tr = tr_26['y'].values.astype(int)
        X_te = te_26[_LOG5_FEATURES].values.astype(float)

        mu  = X_tr[:, 1:].mean(axis=0)
        sig = X_tr[:, 1:].std(axis=0) + 1e-8
        Xtr_s = np.column_stack([X_tr[:, 0], (X_tr[:, 1:] - mu) / sig])
        Xte_s = np.column_stack([X_te[:, 0], (X_te[:, 1:] - mu) / sig])

        mdl = LogisticRegression(C=C_PROD, solver='lbfgs', max_iter=1000)
        mdl.fit(Xtr_s, y_tr)
        p26 = mdl.predict_proba(Xte_s)[:, 1]

        eval_2026_validation(
            te_26, p26,
            title=f'2026 BLIND VALIDATION — {name}',
            model_desc=(f'LogisticRegression C={C_PROD}, trained on '
                        f'{len(tr_26):,} games (2021-2025) '
                        f'[{name} stabilization]'),
        )


if __name__ == '__main__':
    print("=" * 72)
    print("  STABILIZATION PARAMETER SWEEP")
    print("  Walk-forward: optimize on 2025, validate blindly on 2026")
    print("=" * 72)
    print("\nImporting backtest infrastructure (loads cached data)...")

    run_stabilization_sweep()
