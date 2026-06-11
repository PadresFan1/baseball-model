import json, os
import pandas as pd

SNAP = "odds_snapshots_2026.csv"
CAPTURED = os.path.join("logs", "captured_closers.json")

df = pd.read_csv(SNAP)

required = {"run_window_mst", "snapshot_ts_utc", "commence_time_utc", "event_id", "home_team", "away_team"}
missing = required - set(df.columns)
if missing:
    print("Missing expected columns:", missing)
    print("Columns present:", list(df.columns))
    raise SystemExit

wave = df[df["run_window_mst"] == "WAVE_LOGGER"].copy()
if wave.empty:
    print("No WAVE_LOGGER rows found in", SNAP)
    raise SystemExit

wave["snapshot_ts_utc"] = pd.to_datetime(wave["snapshot_ts_utc"], utc=True)
wave["commence_time_utc"] = pd.to_datetime(wave["commence_time_utc"], utc=True)

# closest snapshot to first pitch, one row per game
g = (wave.groupby(["event_id", "home_team", "away_team", "commence_time_utc"], as_index=False)
          ["snapshot_ts_utc"].max())
g["lead_min"] = (g["commence_time_utc"] - g["snapshot_ts_utc"]).dt.total_seconds() / 60.0

def status(m):
    if m < 0:   return "BAD (in-game)"
    if m <= 15: return "OK"
    return "EARLY (>15m)"

g["status"] = g["lead_min"].apply(status)
g = g.sort_values("commence_time_utc")

print(f"Games with a WAVE_LOGGER snapshot: {len(g)}\n")
for _, r in g.iterrows():
    print(f"{r['commence_time_utc']:%Y-%m-%d %H:%M}Z  {r['away_team']} @ {r['home_team']:<22}  lead {r['lead_min']:6.1f} min  {r['status']}")

print("\nSummary:", g["status"].value_counts().to_dict())

if os.path.exists(CAPTURED):
    with open(CAPTURED) as f:
        cc = json.load(f)
    n = len(cc) if isinstance(cc, (list, dict)) else 0
    print(f"\ncaptured_closers.json: {n} gamePks recorded")
else:
    print(f"\ncaptured_closers.json NOT FOUND at {CAPTURED}")
