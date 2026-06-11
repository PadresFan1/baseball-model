import io
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(r"C:\Users\super\baseball-model\simulator\data")
PROJECTIONS_DIR = Path(r"C:\Users\super\baseball-model\simulator\projections_raw")

CHADWICK_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{shard}.csv"
SHARDS = "0123456789abcdef"

CROSSWALK_COLUMNS = ["key_retro", "key_fangraphs", "key_mlbam", "name_last", "name_first"]

# Checked in priority order. PlayerId/playerid/playerID/key_fangraphs are FanGraphs IDs
# (joined via key_fangraphs); MLBAMID is an MLB Advanced Media ID (joined via key_mlbam).
FG_ID_CANDIDATES = ["PlayerId", "playerid", "playerID", "key_fangraphs", "MLBAMID"]


def build_crosswalk() -> pd.DataFrame:
    frames = []
    for shard in SHARDS:
        url = CHADWICK_URL.format(shard=shard)
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text), dtype=str, keep_default_na=False, low_memory=False))

    people = pd.concat(frames, ignore_index=True)
    crosswalk = people[(people["key_retro"] != "") & (people["key_fangraphs"] != "")][CROSSWALK_COLUMNS]

    out_path = OUT_DIR / "id_crosswalk.csv"
    crosswalk.to_csv(out_path, index=False)
    print(f"\nBuilt crosswalk with {len(crosswalk)} mappings -> {out_path}")
    return crosswalk


def load_crosswalk_lookups(crosswalk: pd.DataFrame) -> tuple[dict, dict]:
    fg_to_retro = dict(zip(crosswalk["key_fangraphs"], crosswalk["key_retro"]))
    mlbam_to_retro = {
        mlbam: retro for mlbam, retro in zip(crosswalk["key_mlbam"], crosswalk["key_retro"]) if mlbam
    }
    return fg_to_retro, mlbam_to_retro


def process_projections(fg_to_retro: dict, mlbam_to_retro: dict) -> None:
    csv_files = sorted(PROJECTIONS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"\nNo CSV files found in {PROJECTIONS_DIR}")
        return

    for path in csv_files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        print(f"\n=== {path.name} ===")
        print(f"columns ({len(df.columns)}): {list(df.columns)}")

        id_col = next((c for c in FG_ID_CANDIDATES if c in df.columns), None)
        if id_col is None:
            print("  No recognizable FanGraphs/MLBAM ID column found - skipping join.")
            continue

        id_map = mlbam_to_retro if id_col == "MLBAMID" else fg_to_retro
        crosswalk_key = "key_mlbam" if id_col == "MLBAMID" else "key_fangraphs"

        ids = df[id_col].astype(str).str.strip()
        matched = ids.map(id_map)
        n_total = len(df)
        n_matched = int(matched.notna().sum())
        n_unmatched = n_total - n_matched
        print(f"  ID column used: {id_col} (joined via {crosswalk_key})")
        print(f"  Joined: {n_matched}/{n_total} ({n_matched / n_total * 100:.1f}%), {n_unmatched} unmatched")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crosswalk = build_crosswalk()
    fg_to_retro, mlbam_to_retro = load_crosswalk_lookups(crosswalk)
    process_projections(fg_to_retro, mlbam_to_retro)


if __name__ == "__main__":
    main()
