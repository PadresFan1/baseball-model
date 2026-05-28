"""
Reads SBR Odds file, computes per-game median across the 5 tracked books
(BetMGM, FanDuel, Caesars, bet365, DraftKings), then writes consensus home/away
ML into fd_home_ml / fd_away_ml in historical_data/odds_2026_complete.csv.
"""

import pandas as pd
import numpy as np
import os

SBR_PATH = "SBR Odds/MLB_Odds_through_May26.csv"
HIST_PATH = "historical_data/odds_2026_complete.csv"

# SBR abbreviation → historical abbreviation
TEAM_MAP = {
    "SF":  "SFG",
    "SD":  "SDP",
    "WAS": "WSN",
    "TB":  "TBR",
    "KC":  "KCR",
    "AZ":  "ARI",
}

AWAY_COLS = ["BetMGM Away", "FanDuel Away", "Caesars Away", "bet365 Away", "DraftKings Away"]
HOME_COLS = ["BetMGM Home", "FanDuel Home", "Caesars Home", "bet365 Home", "DraftKings Home"]


def to_int(val):
    """Convert American odds string like '+130' or '-155' to int. Returns NaN on blank."""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        return int(str(val).strip().replace("+", ""))
    except ValueError:
        return np.nan


def median_odds(row, cols):
    """Return rounded median across book columns, formatted as American odds string."""
    vals = [to_int(row[c]) for c in cols if c in row.index]
    clean = [v for v in vals if not np.isnan(v)]
    if not clean:
        return None
    med = int(round(np.median(clean)))
    return f"+{med}" if med >= 0 else str(med)


def normalize_team(abbr):
    a = str(abbr).strip()
    return TEAM_MAP.get(a, a)


def main():
    sbr = pd.read_csv(SBR_PATH, dtype=str)
    hist = pd.read_csv(HIST_PATH, dtype=str)

    sbr["Date"] = sbr["Date"].str.strip()
    sbr["away_norm"] = sbr["Away Team"].str.strip().apply(normalize_team)
    sbr["home_norm"] = sbr["Home Team"].str.strip().apply(normalize_team)

    # Compute consensus lines
    sbr["cons_away_ml"] = sbr.apply(lambda r: median_odds(r, AWAY_COLS), axis=1)
    sbr["cons_home_ml"] = sbr.apply(lambda r: median_odds(r, HOME_COLS), axis=1)

    # Build lookup: (date, home_norm, away_norm) -> list of (cons_away, cons_home)
    # List because doubleheaders can share the same key
    from collections import defaultdict
    lookup = defaultdict(list)
    for _, row in sbr.iterrows():
        key = (row["Date"], row["home_norm"], row["away_norm"])
        lookup[key].append((row["cons_away_ml"], row["cons_home_ml"]))

    # Track usage index per key for doubleheader round-robin
    usage = defaultdict(int)

    matched = 0
    unmatched = 0
    already_filled = 0

    for idx, row in hist.iterrows():
        date = str(row["date"]).strip()
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()

        # Skip rows that already have odds
        if str(row.get("fd_home_ml", "")).strip() not in ("", "nan"):
            already_filled += 1
            continue

        key = (date, home, away)
        entries = lookup.get(key, [])
        ui = usage[key]
        if ui < len(entries):
            away_ml, home_ml = entries[ui]
            usage[key] += 1
            if away_ml is not None:
                hist.at[idx, "fd_away_ml"] = away_ml
            if home_ml is not None:
                hist.at[idx, "fd_home_ml"] = home_ml
            matched += 1
        else:
            unmatched += 1

    hist.to_csv(HIST_PATH, index=False)

    print(f"Done.")
    print(f"  Matched and filled : {matched}")
    print(f"  Already had odds   : {already_filled}")
    print(f"  No SBR match found : {unmatched}")

    # Quick sanity check
    filled = hist[(hist["fd_home_ml"].notna()) & (hist["fd_home_ml"].astype(str).str.strip() != "")]
    print(f"  Rows with fd_home_ml now populated: {len(filled)} / {len(hist)}")


if __name__ == "__main__":
    main()
