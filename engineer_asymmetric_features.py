#!/usr/bin/env python3
"""
engineer_asymmetric_features.py
---------------------------------
Builds three asymmetric interaction features, then merges them with
historical_data/market_residuals.csv to produce
historical_data/training_manifest.csv.

Features produced
-----------------
1. Bullpen Fatigue Index  (3-day trailing pitch count, top-4 high-leverage relievers)
     home_bullpen_3d_pitches, away_bullpen_3d_pitches

2. Pitch-Mix Matchup Score  (30-day PA-weighted run value, lineup vs starter hand)
     home_lineup_vs_away_pitch_mix, away_lineup_vs_home_pitch_mix
   NOTE: true pitch-type splits (FB%, SL%, etc.) are not in the player_data files.
   This feature uses each batter's Statcast run-value-per-100 vs the opposing
   starter's handedness (L/R) as the best available proxy for pitch-mix matchup.

3. Cluster Luck Regression - Base Runs Delta  (14-day trailing average)
     home_team_base_runs_delta, away_team_base_runs_delta
   Delta = actual_runs_scored - base_runs_expected per game;
   trailing mean captures positive/negative cluster luck for regression.

4. Density Altitude Delta
     home_away_density_altitude_delta
   (home_park_elevation - away_team_home_park_elevation) / 1000 feet.
   Positive = thinner air at home than what visitors are used to.
   Elevation drives ~80% of air density variance across MLB parks.
   NOTE: Dynamic weather (temperature, pressure, humidity) not yet collected.
   Upgrade path: DA_ft = elevation + 120*(OAT_degF - ISA_temp_degF)

5. Travel Fatigue Score
     travel_fatigue
   rest_delta + 0.5 * tz_advantage; positive = home team structural advantage.
   rest_delta = home_rest_days - away_rest_days (capped at MAX_REST_DAYS).
   tz_advantage = |away_home_tz - home_tz| (static proxy for time zones crossed).
   NOTE: DGANG detection requires per-game start times (not in current data).

6. Market Total Mismatch
     market_total_mismatch
   density_altitude_delta * fd_total.  Sourced from historical_odds_clean.csv.
"""

import glob
import json
import os
import time
import urllib.request
from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ── Team-abbreviation normalisation (odds CSV -> FanGraphs) ───────────────────
# Matches backtest.py's ODDS_TO_FG so the same games are joined correctly.
_ODDS_TO_FG: Dict[str, str] = {
    'KC':  'KCR',
    'SD':  'SDP',
    'SF':  'SFG',
    'TB':  'TBR',
    'WAS': 'WSN',
    'OAK': 'ATH',
}

def _norm(team: str) -> str:
    return _ODDS_TO_FG.get(team, team)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESIDUALS_CSV    = ROOT / 'historical_data' / 'market_residuals.csv'
HIST_ODDS_CSV    = ROOT / 'historical_data' / 'historical_odds_clean.csv'
PLAYER_TEAM_MAP  = ROOT / 'historical_data' / 'player_team_map.json'
GAME_LINEUPS     = ROOT / 'historical_data' / 'game_lineups.json'
GAME_LOGS        = ROOT / 'historical_data' / 'historical_game_logs.json'
RP_GLOB          = str(ROOT / 'rp' / '*.csv')
BATTER_L_GLOB    = str(ROOT / 'player_data' / 'batters_vs_L_*.csv')
BATTER_R_GLOB    = str(ROOT / 'player_data' / 'batters_vs_R_*.csv')
MANIFEST_CSV     = ROOT / 'historical_data' / 'training_manifest.csv'

# ── Stadium elevations (feet above sea level) — all 30 MLB franchises ─────────
# Elevation is the dominant driver of air density variation across MLB parks.
# Higher elevation = thinner air = ball carries further = run-scoring environment.
# Source: publicly known stadium data; static across the dataset years.
STADIUM_ELEVATION_FT: Dict[str, float] = {
    'COL': 5280.0,   # Coors Field, Denver — extreme outlier
    'ARI': 1082.0,   # Chase Field, Phoenix
    'ATL': 1050.0,   # Truist Park, Cumberland, GA
    'MIN':  837.0,   # Target Field, Minneapolis
    'KCR':  741.0,   # Kauffman Stadium, Kansas City
    'PIT':  722.0,   # PNC Park, Pittsburgh
    'MIL':  630.0,   # American Family Field, Milwaukee
    'DET':  600.0,   # Comerica Park, Detroit
    'CHW':  595.0,   # Guaranteed Rate Field, Chicago
    'CHC':  595.0,   # Wrigley Field, Chicago
    'CIN':  490.0,   # Great American Ball Park, Cincinnati
    'STL':  440.0,   # Busch Stadium, St. Louis
    'TEX':  551.0,   # Globe Life Field, Arlington
    'CLE':  653.0,   # Progressive Field, Cleveland
    'LAD':  510.0,   # Dodger Stadium, Los Angeles
    'TOR':  250.0,   # Rogers Centre, Toronto
    'LAA':  154.0,   # Angel Stadium, Anaheim
    'HOU':   43.0,   # Minute Maid Park, Houston
    'NYY':   55.0,   # Yankee Stadium, Bronx
    'PHI':   40.0,   # Citizens Bank Park, Philadelphia
    'SFG':   10.0,   # Oracle Park, San Francisco
    'SEA':   10.0,   # T-Mobile Park, Seattle
    'ATH':   10.0,   # Oakland Coliseum (transitioning to Las Vegas)
    'NYM':   10.0,   # Citi Field, Queens
    'BOS':   10.0,   # Fenway Park, Boston
    'TBR':   10.0,   # Tropicana Field, St. Petersburg
    'BAL':   10.0,   # Camden Yards, Baltimore
    'WSN':   10.0,   # Nationals Park, Washington D.C.
    'MIA':   10.0,   # loanDepot Park, Miami
    'SDP':   10.0,   # Petco Park, San Diego
}

# ── Time zone UTC offsets during baseball season (EDT/CDT/MDT/PDT) ────────────
# Used as a static proxy for time zones an away team crosses traveling to this park.
TEAM_TZ_OFFSET: Dict[str, int] = {
    'ARI': -7, 'ATH': -7, 'LAD': -7, 'LAA': -7, 'SEA': -7, 'SFG': -7, 'SDP': -7,
    'COL': -6,
    'CHC': -5, 'CHW': -5, 'HOU': -5, 'KCR': -5, 'MIL': -5, 'MIN': -5, 'STL': -5,
    'TEX': -5,
    'ATL': -4, 'BAL': -4, 'BOS': -4, 'CIN': -4, 'CLE': -4, 'DET': -4, 'MIA': -4,
    'NYM': -4, 'NYY': -4, 'PHI': -4, 'PIT': -4, 'TBR': -4, 'TOR': -4, 'WSN': -4,
}

MAX_REST_DAYS = 6   # cap rest days at 6 (covers All-Star break, off-days, etc.)

# ── Stadium coordinates and IANA time zones (all 30 MLB franchises) ───────────
# Used to fetch per-game hourly weather from Open-Meteo and assign local time.
STADIUM_GEO: Dict[str, dict] = {
    'NYY': {'lat': 40.8296, 'lon': -73.9262, 'tz': 'America/New_York'},
    'NYM': {'lat': 40.7571, 'lon': -73.8458, 'tz': 'America/New_York'},
    'BOS': {'lat': 42.3467, 'lon': -71.0972, 'tz': 'America/New_York'},
    'TBR': {'lat': 27.7683, 'lon': -82.6534, 'tz': 'America/New_York'},
    'BAL': {'lat': 39.2838, 'lon': -76.6218, 'tz': 'America/New_York'},
    'TOR': {'lat': 43.6414, 'lon': -79.3894, 'tz': 'America/Toronto'},
    'PHI': {'lat': 39.9061, 'lon': -75.1665, 'tz': 'America/New_York'},
    'WSN': {'lat': 38.8730, 'lon': -77.0074, 'tz': 'America/New_York'},
    'ATL': {'lat': 33.8908, 'lon': -84.4677, 'tz': 'America/New_York'},
    'MIA': {'lat': 25.7781, 'lon': -80.2197, 'tz': 'America/New_York'},
    'CIN': {'lat': 39.0979, 'lon': -84.5089, 'tz': 'America/New_York'},
    'PIT': {'lat': 40.4469, 'lon': -80.0057, 'tz': 'America/New_York'},
    'CLE': {'lat': 41.4962, 'lon': -81.6852, 'tz': 'America/New_York'},
    'DET': {'lat': 42.3390, 'lon': -83.0485, 'tz': 'America/New_York'},
    'CHW': {'lat': 41.8299, 'lon': -87.6338, 'tz': 'America/Chicago'},
    'CHC': {'lat': 41.9484, 'lon': -87.6553, 'tz': 'America/Chicago'},
    'MIL': {'lat': 43.0280, 'lon': -87.9712, 'tz': 'America/Chicago'},
    'STL': {'lat': 38.6226, 'lon': -90.1928, 'tz': 'America/Chicago'},
    'MIN': {'lat': 44.9817, 'lon': -93.2783, 'tz': 'America/Chicago'},
    'KCR': {'lat': 39.0517, 'lon': -94.4803, 'tz': 'America/Chicago'},
    'HOU': {'lat': 29.7573, 'lon': -95.3555, 'tz': 'America/Chicago'},
    'TEX': {'lat': 32.7473, 'lon': -97.0824, 'tz': 'America/Chicago'},
    'COL': {'lat': 39.7559, 'lon': -104.9942, 'tz': 'America/Denver'},
    'ARI': {'lat': 33.4453, 'lon': -112.0667, 'tz': 'America/Phoenix'},
    'LAD': {'lat': 34.0739, 'lon': -118.2400, 'tz': 'America/Los_Angeles'},
    'LAA': {'lat': 33.8003, 'lon': -117.8827, 'tz': 'America/Los_Angeles'},
    'SFG': {'lat': 37.7786, 'lon': -122.3893, 'tz': 'America/Los_Angeles'},
    'SDP': {'lat': 32.7076, 'lon': -117.1570, 'tz': 'America/Los_Angeles'},
    'SEA': {'lat': 47.5914, 'lon': -122.3325, 'tz': 'America/Los_Angeles'},
    'ATH': {'lat': 37.7516, 'lon': -122.2005, 'tz': 'America/Los_Angeles'},
}

WEATHER_CACHE   = ROOT / 'cache' / 'weather_stadium.csv'
WEATHER_START   = '2021-01-01'
WEATHER_END     = '2025-12-31'
GAME_LOCAL_HOUR = 19   # 7 PM local — proxy for typical MLB first pitch

# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_add(date_str: str, days: int) -> str:
    return (pd.Timestamp(date_str) + pd.Timedelta(days=days)).strftime('%Y-%m-%d')


def _date_sub(date_str: str, days: int) -> str:
    return (pd.Timestamp(date_str) - pd.Timedelta(days=days)).strftime('%Y-%m-%d')


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1  —  Bullpen Fatigue Index
# ══════════════════════════════════════════════════════════════════════════════

def load_rp_data() -> pd.DataFrame:
    """Load all rp/*.csv files and return a single DataFrame."""
    cols = ['player_id', 'game_date', 'total_pitches', 'pa']
    frames = [pd.read_csv(f, usecols=cols) for f in glob.glob(RP_GLOB)]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['pa', 'total_pitches'])
    df['player_id'] = df['player_id'].astype(str)
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    df['season']    = df['game_date'].str[:4]
    df['pa']        = pd.to_numeric(df['pa'], errors='coerce').fillna(0)
    df['total_pitches'] = pd.to_numeric(df['total_pitches'], errors='coerce').fillna(0)
    return df


def build_bullpen_fatigue_cache(
    rp_df: pd.DataFrame,
    player_team_map: dict,
    target_pairs: List[Tuple[str, str]],   # [(team, date), ...]
) -> Dict[Tuple[str, str], int]:
    """
    For each (team, date) in target_pairs, compute the total pitch count thrown
    by that team's top-4 high-leverage relievers (ranked by season-to-date PA)
    in the 3 calendar days before `date`.

    Returns {(team, date): pitch_count}.
    """
    if rp_df.empty:
        return {}

    # Map player_id -> team using the season-specific player_team_map
    def get_team(row):
        return player_team_map.get(row['season'], {}).get(row['player_id'])

    rp_df = rp_df.copy()
    rp_df['team'] = rp_df.apply(get_team, axis=1)
    rp_df = rp_df.dropna(subset=['team'])
    rp_df = rp_df.sort_values(['team', 'season', 'game_date'])

    cache: Dict[Tuple[str, str], int] = {}

    # Organize target dates per (team, season)
    team_season_targets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for team, date in target_pairs:
        season = date[:4]
        team_season_targets[(team, season)].append(date)

    for (team, season), target_dates in team_season_targets.items():
        subset = rp_df[(rp_df['team'] == team) & (rp_df['season'] == season)]
        if subset.empty:
            continue

        # Records sorted by date: list of (date, player_id, pa, pitches)
        records = list(zip(
            subset['game_date'].tolist(),
            subset['player_id'].tolist(),
            subset['pa'].tolist(),
            subset['total_pitches'].tolist(),
        ))

        # Per-pitcher history for efficient 3-day window scan
        pitcher_dates:   Dict[str, List[str]]  = defaultdict(list)
        pitcher_pitches: Dict[str, List[float]] = defaultdict(list)
        for d, pid, _, pitches in records:
            pitcher_dates[pid].append(d)
            pitcher_pitches[pid].append(pitches)

        sorted_targets = sorted(set(target_dates))
        ptr   = 0                         # pointer into records (sorted by date)
        sto_pa: Dict[str, float] = {}     # season-to-date PA per pitcher

        for target_date in sorted_targets:
            # Advance cumulative STO PA up to (but not including) target_date
            while ptr < len(records) and records[ptr][0] < target_date:
                _, pid, pa, _ = records[ptr]
                sto_pa[pid] = sto_pa.get(pid, 0.0) + pa
                ptr += 1

            if not sto_pa:
                cache[(team, target_date)] = 0
                continue

            # Top-4 relievers by season-to-date PA
            top4 = sorted(sto_pa, key=lambda p: -sto_pa[p])[:4]

            # Sum their pitches in the trailing 3 calendar days
            window_start = _date_sub(target_date, 3)
            total = 0
            for pid in top4:
                dates_pid = pitcher_dates.get(pid, [])
                pitches_pid = pitcher_pitches.get(pid, [])
                for d, p in zip(dates_pid, pitches_pid):
                    if window_start <= d < target_date:
                        total += int(p)
            cache[(team, target_date)] = total

    return cache


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2  —  Pitch-Mix Matchup Score
# ══════════════════════════════════════════════════════════════════════════════

def load_batter_history() -> Dict[Tuple[int, str], Tuple[List[str], List[float], List[float]]]:
    """
    Load batters_vs_L_*.csv and batters_vs_R_*.csv.
    Returns {(player_id, hand): (dates, pa_list, rv100_list)} with each list sorted
    by date for binary-search lookups.
    """
    all_frames = []
    for hand, pattern in [('L', BATTER_L_GLOB), ('R', BATTER_R_GLOB)]:
        for path in glob.glob(pattern):
            df = pd.read_csv(
                path,
                usecols=['player_id', 'game_date', 'pa', 'batter_run_value_per_100'],
            )
            df['hand'] = hand
            all_frames.append(df)

    if not all_frames:
        return {}

    df = pd.concat(all_frames, ignore_index=True)
    df = df.dropna(subset=['pa', 'batter_run_value_per_100'])
    df['player_id']              = pd.to_numeric(df['player_id'], errors='coerce').dropna().astype(int)
    df['pa']                     = pd.to_numeric(df['pa'], errors='coerce').fillna(0)
    df['batter_run_value_per_100'] = pd.to_numeric(df['batter_run_value_per_100'], errors='coerce').fillna(0)
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    df = df[df['pa'] > 0].sort_values(['player_id', 'hand', 'game_date'])

    index: Dict[Tuple[int, str], Tuple[List[str], List[float], List[float]]] = {}
    for (pid, hand), grp in df.groupby(['player_id', 'hand'], sort=False):
        index[(int(pid), hand)] = (
            grp['game_date'].tolist(),
            grp['pa'].tolist(),
            grp['batter_run_value_per_100'].tolist(),
        )
    return index


def _batter_rv100_30d(
    player_id: int,
    target_date: str,
    hand: str,
    history: Dict[Tuple[int, str], Tuple[List[str], List[float], List[float]]],
) -> Tuple[float, float]:
    """
    PA-weighted 30-day rolling batter_run_value_per_100 ending before target_date.
    Returns (rv100, total_pa). Returns (0.0, 0.0) if no data.
    """
    entry = history.get((player_id, hand))
    if entry is None:
        return 0.0, 0.0

    dates, pa_list, rv100_list = entry
    window_start = _date_sub(target_date, 30)

    lo = bisect_left(dates, window_start)
    hi = bisect_left(dates, target_date)

    if lo >= hi:
        return 0.0, 0.0

    total_pa     = sum(pa_list[lo:hi])
    total_rv_raw = sum(rv100_list[i] * pa_list[i] for i in range(lo, hi))

    if total_pa == 0:
        return 0.0, 0.0

    return total_rv_raw / total_pa, total_pa


def compute_pitch_mix_scores(
    residuals: pd.DataFrame,
    lineup_map: Dict[Tuple[str, str, str], dict],
    batter_history: Dict[Tuple[int, str], Tuple[List[str], List[float], List[float]]],
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    For each game in residuals, compute:
      home_lineup_vs_away_pitch_mix : PA-weighted rv100 of home lineup vs away starter hand
      away_lineup_vs_home_pitch_mix : PA-weighted rv100 of away lineup vs home starter hand

    Returns two parallel lists (one value per row, or None if lineup data missing).
    """
    home_scores: List[Optional[float]] = []
    away_scores: List[Optional[float]] = []

    for _, row in residuals.iterrows():
        key = (row['game_date'], _norm(row['home_team']), _norm(row['away_team']))
        game = lineup_map.get(key)

        if game is None:
            home_scores.append(None)
            away_scores.append(None)
            continue

        home_lineup    = game.get('home_lineup', [])
        away_lineup    = game.get('away_lineup', [])
        away_sp_hand   = game.get('away_starter_hand', 'R')
        home_sp_hand   = game.get('home_starter_hand', 'R')
        target_date    = row['game_date']

        def lineup_score(batters, hand):
            if not batters:
                return None
            total_rv  = 0.0
            total_pa  = 0.0
            for pid in batters:
                rv, pa = _batter_rv100_30d(int(pid), target_date, hand, batter_history)
                total_rv += rv * pa
                total_pa += pa
            # Return PA-weighted average; fall back to 0 when no batter data at all
            return total_rv / total_pa if total_pa > 0 else 0.0

        home_scores.append(lineup_score(home_lineup, away_sp_hand))
        away_scores.append(lineup_score(away_lineup, home_sp_hand))

    return home_scores, away_scores


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3  —  Cluster Luck Regression (Base Runs Delta)
# ══════════════════════════════════════════════════════════════════════════════

def _base_runs(hits: float, doubles: float, triples: float, home_runs: float,
               bb: float, hbp: float, ab: float) -> float:
    """
    Simplified Smyth-Patriot Base Runs estimator.
      A  = on-base component (baserunners created)
      B  = advancement factor
      C  = outs component
      D  = directly-scoring events (HR)
      BR = A * B / (B + C) + D
    """
    singles    = max(0.0, hits - doubles - triples - home_runs)
    total_bases = singles + 2 * doubles + 3 * triples + 4 * home_runs
    outs        = max(0.0, ab - hits)            # approximation; includes K

    A = hits + bb + hbp - home_runs
    B = 1.4 * total_bases - 0.6 * hits - 3.0 * home_runs + 0.1 * (bb + hbp)
    C = 3.0 * outs
    D = home_runs

    denom = B + C
    if denom <= 0:
        return D
    return max(0.0, A * B / denom + D)


def build_base_runs_cache(
    game_logs: dict,
    target_pairs: List[Tuple[str, str]],   # [(team, date), ...]
) -> Dict[Tuple[str, str], float]:
    """
    For each (team, date), compute the 14-day trailing mean of
    (actual_runs_scored - base_runs_expected).  Positive = lucky clustering;
    negative = unlucky clustering.
    """
    # Pre-compute delta per (team, date) from game logs
    game_delta: Dict[Tuple[str, str], float] = {}

    for season, teams in game_logs.items():
        for team, logs in teams.items():
            for entry in logs.get('hitting', []):
                date = entry.get('date', '')
                if not date:
                    continue
                try:
                    actual = float(entry.get('runs', 0) or 0)
                    hits   = float(entry.get('hits',   0) or 0)
                    dbl    = float(entry.get('doubles', 0) or 0)
                    tpl    = float(entry.get('triples', 0) or 0)
                    hr     = float(entry.get('homeRuns', 0) or 0)
                    bb     = float(entry.get('baseOnBalls', 0) or 0)
                    hbp    = float(entry.get('hitByPitch', 0) or 0)
                    ab_    = float(entry.get('atBats', 0) or 0)
                except (TypeError, ValueError):
                    continue

                expected = _base_runs(hits, dbl, tpl, hr, bb, hbp, ab_)
                game_delta[(team, date)] = actual - expected

    # Build sorted date-delta pairs per team for rolling window
    team_series: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (team, date), delta in game_delta.items():
        team_series[team].append((date, delta))
    for team in team_series:
        team_series[team].sort(key=lambda x: x[0])

    cache: Dict[Tuple[str, str], float] = {}
    team_target_dates: Dict[str, List[str]] = defaultdict(list)
    for team, date in target_pairs:
        team_target_dates[team].append(date)

    for team, target_dates in team_target_dates.items():
        series = team_series.get(team, [])
        if not series:
            continue
        dates_only   = [r[0] for r in series]
        deltas_only  = [r[1] for r in series]

        for target_date in target_dates:
            window_start = _date_sub(target_date, 14)
            lo = bisect_left(dates_only, window_start)
            hi = bisect_left(dates_only, target_date)

            if lo >= hi:
                continue
            cache[(team, target_date)] = float(np.mean(deltas_only[lo:hi]))

    return cache


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4  —  Dynamic Density Altitude Delta
# ══════════════════════════════════════════════════════════════════════════════

def compute_density_altitude_deltas_static(
    residuals: pd.DataFrame,
) -> List[Optional[float]]:
    """
    Static elevation proxy for density altitude delta.

    home_away_density_altitude_delta = (home_park_elevation - away_home_elevation) / 1000

    Units: thousands of feet.  Positive = thinner air at home than what visitors
    are accustomed to.  Coors Field (5,280 ft) hosting a sea-level team gives +5.27.

    This proxy is computationally free and stable, but cannot capture game-to-game
    weather variation (temperature, humidity).  Dynamic mode (USE_DYNAMIC_WEATHER)
    uses the Open-Meteo API and the thermodynamic formula below when needed.
    """
    deltas: List[Optional[float]] = []
    for _, row in residuals.iterrows():
        ht = _norm(row['home_team'])
        at = _norm(row['away_team'])
        home_elev = STADIUM_ELEVATION_FT.get(ht)
        away_elev = STADIUM_ELEVATION_FT.get(at)
        if home_elev is None or away_elev is None:
            deltas.append(None)
        else:
            deltas.append(round((home_elev - away_elev) / 1000.0, 4))
    return deltas


# ── Dynamic weather functions (USE_DYNAMIC_WEATHER = True) ────────────────────

def _density_altitude_ft(temp_c: float, pressure_hpa: float, rh_pct: float) -> float:
    """
    Thermodynamic density altitude in feet incorporating moist-air correction.

    Uses the ISA equivalence formula:
      DA = 145366.45 * [1 - (P/P0 * T0/Tv)^0.234969]

    where Tv is the virtual temperature (dry-bulb temperature corrected for
    the buoyancy effect of water vapour):
      mixing_ratio r = 0.622 * e_v / (P - e_v)
      Tv = T * (1 + 0.608 * r)

    Saturation vapour pressure via Magnus formula:
      e_s = 6.1078 * exp(17.2694 * T_C / (T_C + 238.3))  [hPa]

    References: FAA AC 00-6B, ICAO Standard Atmosphere (Doc 7488/3).
    """
    T_K  = temp_c + 273.15
    T0   = 288.15   # ISA sea-level temperature (K)
    P0   = 1013.25  # ISA sea-level pressure (hPa)

    e_s  = 6.1078 * np.exp(17.2694 * temp_c / (temp_c + 238.3))
    e_v  = (rh_pct / 100.0) * e_s
    r    = 0.622 * e_v / max(pressure_hpa - e_v, 0.01)
    T_v  = T_K * (1.0 + 0.608 * r)

    return float(145366.45 * (1.0 - (pressure_hpa / P0 * T0 / T_v) ** 0.234969))


def _fetch_weather_for_stadium(
    team: str, geo: dict, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch hourly weather from the Open-Meteo free archive API for one stadium.
    Extracts only the GAME_LOCAL_HOUR (7 PM local) row per date.
    Returns DataFrame: team_fg, date, temp_c, pressure_hpa, rh_pct, da_ft.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={geo['lat']}&longitude={geo['lon']}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m,surface_pressure,relative_humidity_2m"
        f"&timezone={geo['tz']}"
    )
    with urllib.request.urlopen(url, timeout=45) as resp:
        data = json.load(resp)

    h      = data['hourly']
    times  = h['time']
    temps  = h['temperature_2m']
    pres   = h['surface_pressure']
    rhs    = h['relative_humidity_2m']

    suffix = f'T{GAME_LOCAL_HOUR:02d}:00'
    rows   = []
    for i, ts in enumerate(times):
        if not ts.endswith(suffix):
            continue
        t_c = temps[i]
        p   = pres[i]
        rh  = rhs[i]
        if t_c is None or p is None or rh is None:
            continue
        rows.append({
            'team_fg':      team,
            'date':         ts[:10],
            'temp_c':       round(float(t_c), 2),
            'pressure_hpa': round(float(p),   2),
            'rh_pct':       round(float(rh),  1),
            'da_ft':        round(_density_altitude_ft(float(t_c), float(p), float(rh)), 1),
        })
    return pd.DataFrame(rows)


def load_or_fetch_weather(force: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of per-stadium, per-date weather at GAME_LOCAL_HOUR.
    Loads from cache/weather_stadium.csv on subsequent runs; fetches from
    Open-Meteo only on first run or when force=True.

    Columns: team_fg, date, temp_c, pressure_hpa, rh_pct, da_ft
    """
    if WEATHER_CACHE.exists() and not force:
        df = pd.read_csv(WEATHER_CACHE, dtype={'date': str})
        print(f"  Loaded {len(df):,} stadium-days from cache ({WEATHER_CACHE.name})")
        return df

    WEATHER_CACHE.parent.mkdir(exist_ok=True)
    frames = []
    n = len(STADIUM_GEO)
    for i, (team, geo) in enumerate(STADIUM_GEO.items(), 1):
        print(f"  [{i:2d}/{n}] {team} ...", end='', flush=True)
        try:
            df = _fetch_weather_for_stadium(team, geo, WEATHER_START, WEATHER_END)
            frames.append(df)
            print(f" {len(df)} days ok")
        except Exception as exc:
            print(f" FAILED ({exc})")
        time.sleep(0.4)   # stay well inside Open-Meteo's free-tier rate limit

    if not frames:
        print("  WARNING: no weather data fetched; DA feature will be all-NaN")
        return pd.DataFrame(
            columns=['team_fg', 'date', 'temp_c', 'pressure_hpa', 'rh_pct', 'da_ft']
        )

    weather = pd.concat(frames, ignore_index=True)
    weather.to_csv(WEATHER_CACHE, index=False)
    print(f"  Cached {len(weather):,} stadium-days -> {WEATHER_CACHE}")
    return weather


def build_da_delta_cache(
    weather_df: pd.DataFrame,
    target_pairs: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    """
    Compute home_away_density_altitude_delta for each (home_team, game_date).

    delta = game_DA - stadium_month_baseline_DA

    baseline = mean DA at GAME_LOCAL_HOUR for (team, calendar_month) across
    all years in the weather dataset.  A positive delta means today's air is
    thinner than this park's historical average for this time of year — the
    ball will carry further than the market's pricing history anticipates.

    Fallback: when weather data is missing for a game, returns NaN (imputed
    downstream by SimpleImputer).
    """
    if weather_df.empty:
        return {}

    wdf = weather_df.copy()
    wdf['month'] = wdf['date'].str[5:7].astype(int)

    # Stadium-month DA baseline (mean across all available years)
    baseline = (
        wdf.groupby(['team_fg', 'month'])['da_ft']
        .mean()
        .to_dict()
    )

    # (team, date) -> game DA
    da_lookup = wdf.set_index(['team_fg', 'date'])['da_ft'].to_dict()

    cache: Dict[Tuple[str, str], float] = {}
    for team, date in target_pairs:
        game_da = da_lookup.get((team, date))
        if game_da is None:
            continue
        month = int(date[5:7])
        base = baseline.get((team, month))
        if base is None:
            continue
        cache[(team, date)] = round(float(game_da) - float(base), 2)

    return cache


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5  —  Travel Fatigue Score
# ══════════════════════════════════════════════════════════════════════════════

def build_rest_days_cache(
    game_logs: dict,
    target_pairs: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], int]:
    """
    For each (team, date), compute rest_days = days since that team's last game.

    rest_days = 0 → back-to-back (played the previous calendar day)
    rest_days = 1 → one off-day between games
    rest_days >= MAX_REST_DAYS → capped (long breaks, All-Star, etc.)

    Derived purely from the dates in historical_game_logs.json; no game times
    or locations are required.  Returns {} entry for (team, date) if no prior
    game is found in the logs (e.g., season opener).
    """
    # Collect all known game dates per team across all seasons
    team_all_dates: Dict[str, set] = defaultdict(set)
    for season, teams in game_logs.items():
        for team, data in teams.items():
            for entry in data.get('hitting', []):
                d = entry.get('date', '')
                if d:
                    team_all_dates[team].add(d)

    team_sorted: Dict[str, List[str]] = {
        t: sorted(ds) for t, ds in team_all_dates.items()
    }

    cache: Dict[Tuple[str, str], int] = {}
    for team, target_date in target_pairs:
        dates = team_sorted.get(team, [])
        if not dates:
            continue
        idx = bisect_left(dates, target_date) - 1
        if idx < 0:
            continue  # season opener — no prior game
        last_game = dates[idx]
        days_diff = (pd.Timestamp(target_date) - pd.Timestamp(last_game)).days - 1
        cache[(team, target_date)] = min(max(0, days_diff), MAX_REST_DAYS)

    return cache


def compute_travel_fatigue(
    residuals: pd.DataFrame,
    rest_cache: Dict[Tuple[str, str], int],
) -> List[Optional[float]]:
    """
    travel_fatigue = rest_delta + 0.5 * tz_advantage

    rest_delta   = home_rest_days - away_rest_days  (positive = home more rested)
    tz_advantage = |away_home_tz - home_tz|          (static proxy; hours the away
                   team typically crosses traveling from their home city)

    Positive values favour the home team; negative values favour the away team.

    Data limitations:
      - rest_days derived from game-log dates only; game times unavailable so
        DGANG (Day-Game-After-Night-Game) detection is not implemented.
      - tz_advantage uses each team's home city, not their actual previous city,
        because no travel schedule is currently stored.
    """
    scores: List[Optional[float]] = []
    for _, row in residuals.iterrows():
        ht = _norm(row['home_team'])
        at = _norm(row['away_team'])
        gd = row['game_date']

        home_rest = rest_cache.get((ht, gd))
        away_rest = rest_cache.get((at, gd))

        if home_rest is None or away_rest is None:
            scores.append(None)
            continue

        rest_delta   = float(home_rest - away_rest)
        home_tz      = TEAM_TZ_OFFSET.get(ht, -5)
        away_tz      = TEAM_TZ_OFFSET.get(at, -5)
        tz_advantage = float(abs(away_tz - home_tz))

        scores.append(round(rest_delta + 0.5 * tz_advantage, 3))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Load base dataset ──────────────────────────────────────────────────────
    print("Loading market_residuals.csv ...")
    residuals = pd.read_csv(RESIDUALS_CSV, dtype={'home_team': str, 'away_team': str})
    print(f"  {len(residuals):,} games ({residuals['game_date'].min()} to {residuals['game_date'].max()})")

    # Collect (team, date) pairs we need features for.
    # Normalise abbreviations (e.g. 'TB' -> 'TBR') so all three caches
    # use consistent FanGraphs keys throughout.
    all_pairs: List[Tuple[str, str]] = []
    for _, row in residuals.iterrows():
        all_pairs.append((_norm(row['home_team']), row['game_date']))
        all_pairs.append((_norm(row['away_team']), row['game_date']))
    all_pairs = list(set(all_pairs))

    # ── Feature 1: Bullpen Fatigue ─────────────────────────────────────────────
    print("\n[1/3] Building bullpen fatigue cache ...")

    with open(PLAYER_TEAM_MAP, encoding='utf-8') as f:
        player_team_map = json.load(f)

    rp_df = load_rp_data()
    print(f"  Loaded {len(rp_df):,} reliever-game rows from {len(glob.glob(RP_GLOB))} files")

    bp_cache = build_bullpen_fatigue_cache(rp_df, player_team_map, all_pairs)
    print(f"  Cache entries: {len(bp_cache):,}")

    # ── Feature 2: Pitch-Mix Matchup ───────────────────────────────────────────
    print("\n[2/3] Building pitch-mix matchup scores ...")

    with open(GAME_LINEUPS, encoding='utf-8') as f:
        raw_lineups = json.load(f)

    # Build (date, home_fg, away_fg) -> game_data lookup; first game wins for DH
    lineup_map: Dict[Tuple[str, str, str], dict] = {}
    for season, season_games in raw_lineups.items():
        for game_pk, game in season_games.items():
            key = (game['date'], game['home_fg'], game['away_fg'])
            if key not in lineup_map:
                lineup_map[key] = game
    print(f"  {len(lineup_map):,} game lineup entries loaded")

    print("  Loading batter vs-L / vs-R history (all seasons) ...")
    batter_history = load_batter_history()
    print(f"  {len(batter_history):,} (player, hand) pairs indexed")

    home_pitch_mix, away_pitch_mix = compute_pitch_mix_scores(
        residuals, lineup_map, batter_history
    )
    matched  = sum(1 for v in home_pitch_mix if v is not None)
    print(f"  Matched lineup data for {matched:,} / {len(residuals):,} games")

    # ── Feature 3: Base Runs Delta ─────────────────────────────────────────────
    print("\n[3/3] Building Base Runs delta cache ...")

    with open(GAME_LOGS, encoding='utf-8') as f:
        game_logs = json.load(f)

    br_cache = build_base_runs_cache(game_logs, all_pairs)
    print(f"  Cache entries: {len(br_cache):,}")

    # ── Feature 4: Density Altitude Delta ─────────────────────────────────────
    # Toggle: False = fast static elevation proxy (default)
    #         True  = dynamic Open-Meteo thermodynamic DA (requires network)
    USE_DYNAMIC_WEATHER = False

    print(f"\n[4/6] Computing density altitude deltas "
          f"({'dynamic weather' if USE_DYNAMIC_WEATHER else 'static elevation'}) ...")

    if USE_DYNAMIC_WEATHER:
        weather_df = load_or_fetch_weather()
        home_pairs = [(_norm(row['home_team']), row['game_date'])
                      for _, row in residuals.iterrows()]
        da_cache   = build_da_delta_cache(weather_df, list(set(home_pairs)))
        da_deltas: List[Optional[float]] = [
            da_cache.get((_norm(row['home_team']), row['game_date']))
            for _, row in residuals.iterrows()
        ]
    else:
        da_deltas = compute_density_altitude_deltas_static(residuals)

    matched_da = sum(1 for v in da_deltas if v is not None)
    print(f"  Matched: {matched_da:,} / {len(residuals):,} games")

    # ── Feature 5: Travel Fatigue ──────────────────────────────────────────────
    print("\n[5/6] Building rest-days cache for travel fatigue ...")
    rest_cache = build_rest_days_cache(game_logs, all_pairs)
    print(f"  Cache entries: {len(rest_cache):,}")
    fatigue_scores = compute_travel_fatigue(residuals, rest_cache)
    matched_tf = sum(1 for v in fatigue_scores if v is not None)
    print(f"  Matched: {matched_tf:,} / {len(residuals):,} games")

    # ── Feature 6: Market Total Mismatch ──────────────────────────────────────
    print("\n[6/6] Joining fd_total for market_total_mismatch ...")
    hist_odds = pd.read_csv(
        HIST_ODDS_CSV,
        usecols=['date', 'home_team', 'away_team', 'fd_total'],
        dtype={'home_team': str, 'away_team': str},
    ).rename(columns={'date': 'game_date'})
    hist_odds['home_team'] = hist_odds['home_team'].apply(_norm)
    hist_odds['away_team'] = hist_odds['away_team'].apply(_norm)
    hist_odds = hist_odds.dropna(subset=['fd_total'])

    # Join to residuals on (game_date, home_team, away_team)
    residuals_normed = residuals.copy()
    residuals_normed['home_team_n'] = residuals_normed['home_team'].apply(_norm)
    residuals_normed['away_team_n'] = residuals_normed['away_team'].apply(_norm)
    fd_total_map = hist_odds.set_index(
        ['game_date', 'home_team', 'away_team']
    )['fd_total'].to_dict()

    fd_totals: List[Optional[float]] = []
    for _, row in residuals_normed.iterrows():
        key = (row['game_date'], row['home_team_n'], row['away_team_n'])
        fd_totals.append(fd_total_map.get(key))

    matched_tot = sum(1 for v in fd_totals if v is not None)
    print(f"  fd_total matched: {matched_tot:,} / {len(residuals):,} games "
          f"({100*matched_tot/len(residuals):.1f}%)")

    # market_total_mismatch = density_altitude_delta * fd_total
    mismatch: List[Optional[float]] = []
    for da, tot in zip(da_deltas, fd_totals):
        if da is None or tot is None:
            mismatch.append(None)
        else:
            mismatch.append(round(da * tot, 4))

    # ── Merge all features ────────────────────────────────────────────────────
    print("\nMerging features into training manifest ...")

    home_bp_pitches: List[Optional[int]] = []
    away_bp_pitches: List[Optional[int]] = []
    home_br_delta:   List[Optional[float]] = []
    away_br_delta:   List[Optional[float]] = []

    for _, row in residuals.iterrows():
        gd  = row['game_date']
        ht  = _norm(row['home_team'])
        at_ = _norm(row['away_team'])

        home_bp_pitches.append(bp_cache.get((ht,  gd)))
        away_bp_pitches.append(bp_cache.get((at_, gd)))
        home_br_delta.append(br_cache.get((ht,  gd)))
        away_br_delta.append(br_cache.get((at_, gd)))

    out = residuals.copy()
    out['home_bullpen_3d_pitches']           = home_bp_pitches
    out['away_bullpen_3d_pitches']           = away_bp_pitches
    out['home_lineup_vs_away_pitch_mix']     = home_pitch_mix
    out['away_lineup_vs_home_pitch_mix']     = away_pitch_mix
    out['home_team_base_runs_delta']         = home_br_delta
    out['away_team_base_runs_delta']         = away_br_delta
    out['home_away_density_altitude_delta']  = da_deltas
    out['travel_fatigue']                    = fatigue_scores
    out['market_total_mismatch']             = mismatch

    # Coerce numeric types
    for col in ['home_bullpen_3d_pitches', 'away_bullpen_3d_pitches']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    for col in [
        'home_lineup_vs_away_pitch_mix', 'away_lineup_vs_home_pitch_mix',
        'home_team_base_runs_delta', 'away_team_base_runs_delta',
        'home_away_density_altitude_delta', 'travel_fatigue', 'market_total_mismatch',
    ]:
        out[col] = pd.to_numeric(out[col], errors='coerce').round(6)

    out.to_csv(MANIFEST_CSV, index=False)
    print(f"\nWrote {len(out):,} rows -> {MANIFEST_CSV}")

    # ── Coverage report ────────────────────────────────────────────────────────
    n = len(out)
    print("\nFeature coverage (non-null rows):")
    feature_cols = [
        'home_bullpen_3d_pitches', 'away_bullpen_3d_pitches',
        'home_lineup_vs_away_pitch_mix', 'away_lineup_vs_home_pitch_mix',
        'home_team_base_runs_delta', 'away_team_base_runs_delta',
        'home_away_density_altitude_delta', 'travel_fatigue',
        'market_total_mismatch',
    ]
    for col in feature_cols:
        nn = out[col].notna().sum()
        print(f"  {col:44s}: {nn:,} / {n:,} ({100*nn/n:.1f}%)")


if __name__ == '__main__':
    main()
