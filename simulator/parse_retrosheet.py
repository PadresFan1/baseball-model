import csv
import re
from collections import Counter
from pathlib import Path

from pyretrosheet.load import load_games
from pyretrosheet.models.play import advance as _advance

# Patch a pyretrosheet 0.0.10 bug: advances can carry a stolen-base annotation
# in parens, e.g. "BK.2-3(SB3);1-2". The original _iter_fielding_additional_info
# doesn't recognize "SB3" and crashes on int("S"), which aborts the *entire*
# season load (one bad line in 2025KCA.EVA). Add "SB[23H]" to the ignore list.
_IGNORE_PARTS_RE = r"(WP|TH(\d)?|PB|THH|BR|OBS|(B|R)INT|INT|AP)"


def _patched_iter_fielding_additional_info(additional_info):
    for info in additional_info:
        for part in info.split("/"):
            corrected_part = part.replace("!", "") if "!" in part else part

            if re.fullmatch(_IGNORE_PARTS_RE, corrected_part):
                continue
            if re.fullmatch(r"\d-\d", corrected_part):
                continue
            if re.fullmatch(r"\dX", corrected_part):
                continue
            if re.fullmatch(r"\d+H", corrected_part):
                continue
            if re.fullmatch(r"SB[23H]", corrected_part):
                continue

            try:
                _advance.RunAccreditation(corrected_part)
                continue
            except ValueError:
                pass

            yield corrected_part


_advance._iter_fielding_additional_info = _patched_iter_fielding_additional_info

RAW_DIR = r"C:\Users\super\baseball-model\simulator\retrosheet_raw"
OUT_DIR = r"C:\Users\super\baseball-model\simulator\data"
YEARS = [2021, 2022, 2023, 2024, 2025]

PA_COLUMNS = [
    "season", "game_date", "game_id", "park", "home_team", "away_team",
    "inning", "batter_side", "batter_id", "pitcher_id", "outcome",
]

PLAYER_COLUMNS = ["season", "player_id", "name", "bats", "throws"]

OUTCOME_ORDER = ["1B", "2B", "3B", "HR", "BB", "HBP", "K", "OUT"]


def classify_outcome(p):
    if p.is_home_run():
        return "HR"
    if p.is_triple():
        return "3B"
    if p.is_double():
        return "2B"
    if p.is_single():
        return "1B"
    if p.is_walk():
        return "BB"
    if p.is_hit_by_pitch():
        return "HBP"

    batter_event = p.event.description.batter_event
    if batter_event is not None and "STRIKEOUT" in batter_event.name:
        return "K"

    if p.is_an_at_bat():
        return "OUT"

    return None


def parse_season(year, raw_dir, out_dir):
    games = load_games(year, data_dir=Path(raw_dir))

    rows = []
    outcome_counts = Counter()
    pitcher_filled = 0

    for g in games:
        pitcher = {0: None, 1: None}

        for e in g.chronological_events:
            kind = type(e).__name__

            if kind == "Player":
                if e.fielding_position == 1:
                    pitcher[e.team_location.value] = e.id

            elif kind == "Play":
                outcome = classify_outcome(e)
                if outcome is None:
                    continue

                batter_side = e.team_location.value
                pitcher_id = pitcher[1 - batter_side]
                if pitcher_id:
                    pitcher_filled += 1

                outcome_counts[outcome] += 1
                rows.append([
                    year,
                    g.id.date.isoformat(),
                    f"{g.id.home_team_id}{g.id.date:%Y%m%d}{g.id.game_number}",
                    g.info.get("site"),
                    g.home_team_id,
                    g.visiting_team_id,
                    e.inning,
                    batter_side,
                    e.batter_id,
                    pitcher_id or "",
                    outcome,
                ])

    out_path = Path(out_dir) / f"pa_events_{year}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PA_COLUMNS)
        writer.writerows(rows)

    total = len(rows)
    print(f"\n=== {year}: pa_events_{year}.csv ({total} plate appearances) ===")
    for outcome in OUTCOME_ORDER:
        count = outcome_counts.get(outcome, 0)
        pct = (count / total * 100) if total else 0.0
        print(f"  {outcome:>3}: {count:7d} ({pct:5.2f}%)")

    k_pct = outcome_counts.get("K", 0) / total * 100 if total else 0.0
    bb_pct = outcome_counts.get("BB", 0) / total * 100 if total else 0.0
    hr_pct = outcome_counts.get("HR", 0) / total * 100 if total else 0.0
    pitcher_pct = pitcher_filled / total * 100 if total else 0.0
    print(f"  K%:  {k_pct:.2f}%")
    print(f"  BB%: {bb_pct:.2f}%")
    print(f"  HR%: {hr_pct:.2f}%")
    print(f"  Rows with non-empty pitcher_id: {pitcher_pct:.2f}%")


def parse_players(year, raw_dir, out_dir):
    players = {}
    for ros_file in sorted(Path(raw_dir).glob(f"*{year}.ROS")):
        with open(ros_file, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                player_id, last, first, bats, throws, _team, _pos = row[:7]
                players[player_id] = (player_id, f"{first} {last}", bats, throws)

    out_path = Path(out_dir) / f"players_{year}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLAYER_COLUMNS)
        for player_id, name, bats, throws in sorted(players.values()):
            writer.writerow([year, player_id, name, bats, throws])

    print(f"=== {year}: players_{year}.csv ({len(players)} players) ===")


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        parse_season(year, RAW_DIR, OUT_DIR)
        parse_players(year, RAW_DIR, OUT_DIR)


if __name__ == "__main__":
    main()
