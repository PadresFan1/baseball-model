import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Users\super\baseball-model\simulator\data")
PROJECTIONS_DIR = Path(r"C:\Users\super\baseball-model\simulator\projections_raw")

PA_YEARS = [2021, 2022, 2023, 2024, 2025]

FILENAME_RE = re.compile(r"^(zips|steamer)_(bat|pit)_(\d{4})\.csv$", re.IGNORECASE)

BATTER_RAW_COLUMNS = ["AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "wOBA", "SF"]
BATTER_REQUIRED = ["PA", "H", "2B", "3B", "HR", "BB", "SO", "HBP"]

PITCHER_RAW_COLUMNS = ["H", "HR", "SO", "BB", "HBP", "GS", "ER"]
PITCHER_REQUIRED = ["IP", "H", "HR", "SO", "BB", "HBP"]

BATTER_OUTPUT_COLUMNS = (
    ["season", "system", "key_retro", "key_fangraphs", "key_mlbam", "name", "PA"]
    + BATTER_RAW_COLUMNS
    + ["p_k", "p_bb", "p_hbp", "p_hr", "p_1b", "p_2b", "p_3b"]
)
PITCHER_OUTPUT_COLUMNS = (
    ["season", "system", "key_retro", "key_fangraphs", "key_mlbam", "name", "IP", "est_tbf"]
    + PITCHER_RAW_COLUMNS
    + ["p_k", "p_bb", "p_hbp", "p_hr"]
)


def load_crosswalk():
    cw = pd.read_csv(DATA_DIR / "id_crosswalk.csv", dtype=str, keep_default_na=False)
    fg_to_retro = dict(zip(cw["key_fangraphs"], cw["key_retro"]))
    fg_to_mlbam = dict(zip(cw["key_fangraphs"], cw["key_mlbam"]))
    mlbam_to_retro = {m: r for m, r in zip(cw["key_mlbam"], cw["key_retro"]) if m}
    return fg_to_retro, fg_to_mlbam, mlbam_to_retro


def resolve_ids(df, fg_to_retro, fg_to_mlbam, mlbam_to_retro):
    fg_ids = df["PlayerId"].astype(str).str.strip()
    mlbam_ids = df["MLBAMID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    retro_via_fg = fg_ids.map(fg_to_retro)
    mlbam_via_fg = fg_ids.map(fg_to_mlbam)
    retro_via_mlbam = mlbam_ids.map(mlbam_to_retro)

    key_retro = retro_via_fg.where(retro_via_fg.notna(), retro_via_mlbam)
    key_mlbam = mlbam_via_fg.where(mlbam_via_fg.notna() & (mlbam_via_fg != ""), mlbam_ids)

    return fg_ids, key_mlbam, key_retro


def numeric_column(df, col, required, label):
    if col not in df.columns:
        if required:
            print(f"  WARNING: required column '{col}' missing from {label}")
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def safe_div(numerator, denominator):
    return np.where(denominator > 0, numerator / denominator, np.nan)


def process_batter_file(df, season, system, fg_to_retro, fg_to_mlbam, mlbam_to_retro):
    label = f"{system}_bat_{season}"
    fg_ids, key_mlbam, key_retro = resolve_ids(df, fg_to_retro, fg_to_mlbam, mlbam_to_retro)

    out = pd.DataFrame({
        "season": season,
        "system": system,
        "key_retro": key_retro,
        "key_fangraphs": fg_ids,
        "key_mlbam": key_mlbam,
        "name": df["Name"],
        "PA": numeric_column(df, "PA", True, label),
    })
    for col in BATTER_RAW_COLUMNS:
        out[col] = numeric_column(df, col, col in BATTER_REQUIRED, label)

    pa = out["PA"]
    out["p_k"] = safe_div(out["SO"], pa)
    out["p_bb"] = safe_div(out["BB"], pa)
    out["p_hbp"] = safe_div(out["HBP"], pa)
    out["p_hr"] = safe_div(out["HR"], pa)
    out["p_1b"] = safe_div(out["H"] - out["2B"] - out["3B"] - out["HR"], pa)
    out["p_2b"] = safe_div(out["2B"], pa)
    out["p_3b"] = safe_div(out["3B"], pa)

    out = out[out["key_retro"].notna() & (out["key_retro"] != "")]
    return out[BATTER_OUTPUT_COLUMNS]


def process_pitcher_file(df, season, system, fg_to_retro, fg_to_mlbam, mlbam_to_retro):
    label = f"{system}_pit_{season}"
    fg_ids, key_mlbam, key_retro = resolve_ids(df, fg_to_retro, fg_to_mlbam, mlbam_to_retro)

    out = pd.DataFrame({
        "season": season,
        "system": system,
        "key_retro": key_retro,
        "key_fangraphs": fg_ids,
        "key_mlbam": key_mlbam,
        "name": df["Name"],
        "IP": numeric_column(df, "IP", True, label),
    })
    for col in PITCHER_RAW_COLUMNS:
        out[col] = numeric_column(df, col, col in PITCHER_REQUIRED, label)

    out["est_tbf"] = 3 * out["IP"] + out["H"] + out["BB"] + out["HBP"]
    tbf = out["est_tbf"]
    out["p_k"] = safe_div(out["SO"], tbf)
    out["p_bb"] = safe_div(out["BB"], tbf)
    out["p_hbp"] = safe_div(out["HBP"], tbf)
    out["p_hr"] = safe_div(out["HR"], tbf)

    out = out[out["key_retro"].notna() & (out["key_retro"] != "")]
    return out[PITCHER_OUTPUT_COLUMNS]


def build_projection_tables():
    fg_to_retro, fg_to_mlbam, mlbam_to_retro = load_crosswalk()

    batter_frames = []
    pitcher_frames = []
    for path in sorted(PROJECTIONS_DIR.glob("*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            print(f"Skipping unrecognized file: {path.name}")
            continue

        system, kind, year = match.group(1).lower(), match.group(2).lower(), int(match.group(3))
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

        if kind == "bat":
            batter_frames.append(process_batter_file(df, year, system, fg_to_retro, fg_to_mlbam, mlbam_to_retro))
        else:
            pitcher_frames.append(process_pitcher_file(df, year, system, fg_to_retro, fg_to_mlbam, mlbam_to_retro))

    proj_batters = pd.concat(batter_frames, ignore_index=True)
    proj_pitchers = pd.concat(pitcher_frames, ignore_index=True)

    proj_batters.to_csv(DATA_DIR / "proj_batters.csv", index=False)
    proj_pitchers.to_csv(DATA_DIR / "proj_pitchers.csv", index=False)

    print(f"\nproj_batters.csv: {len(proj_batters)} rows")
    print(proj_batters.groupby(["system", "season"]).size().unstack(fill_value=0))

    print(f"\nproj_pitchers.csv: {len(proj_pitchers)} rows")
    print(proj_pitchers.groupby(["system", "season"]).size().unstack(fill_value=0))

    return proj_batters, proj_pitchers


def coverage_report(proj_batters, proj_pitchers):
    print("\n=== Reverse coverage report ===")
    rows = []
    for year in PA_YEARS:
        pa = pd.read_csv(
            DATA_DIR / f"pa_events_{year}.csv",
            usecols=["batter_id", "pitcher_id"],
            dtype=str,
        )

        batter_pa = pa.groupby("batter_id").size()
        pitcher_bf = pa.groupby("pitcher_id").size()

        covered_batter_ids = set(proj_batters.loc[proj_batters["season"] == year, "key_retro"])
        covered_pitcher_ids = set(proj_pitchers.loc[proj_pitchers["season"] == year, "key_retro"])

        batter_covered_mask = batter_pa.index.isin(covered_batter_ids)
        batter_player_pct = batter_covered_mask.sum() / len(batter_pa) * 100
        batter_pa_pct = batter_pa[batter_covered_mask].sum() / batter_pa.sum() * 100

        pitcher_covered_mask = pitcher_bf.index.isin(covered_pitcher_ids)
        pitcher_player_pct = pitcher_covered_mask.sum() / len(pitcher_bf) * 100
        pitcher_bf_pct = pitcher_bf[pitcher_covered_mask].sum() / pitcher_bf.sum() * 100

        rows.append({
            "season": year,
            "batter_player_cov_%": round(batter_player_pct, 1),
            "batter_PA_weighted_%": round(batter_pa_pct, 1),
            "pitcher_player_cov_%": round(pitcher_player_pct, 1),
            "pitcher_BF_weighted_%": round(pitcher_bf_pct, 1),
        })

    print(pd.DataFrame(rows).to_string(index=False))


def main():
    proj_batters, proj_pitchers = build_projection_tables()
    coverage_report(proj_batters, proj_pitchers)


if __name__ == "__main__":
    main()
