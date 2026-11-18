"""Download play-by-play, clean it, rate teams with Elo, save game states.

    python scripts/build_dataset.py --seasons 2019 2020 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbawp import data, elo  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--shots", type=int, nargs="*", default=[2024, 2025])
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # v3 play-by-play only has short names; shot detail has full ones.
    names: dict = {}
    for season in args.shots:
        shots = pd.read_csv(data.download(f"shotdetail_{season}", RAW), usecols=["PLAYER_ID", "PLAYER_NAME"])
        names.update(shots.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"].to_dict())

    events = []
    for season in args.seasons:
        source = "nbastatsv3" if season >= data.V3_FIRST_SEASON else "nbastats"
        path = data.download(f"{source}_{season}", RAW)
        ev = data.load_pbp(path, season, names)
        print(f"{season}-{(season + 1) % 100:02d}: {ev['game_id'].nunique()} games, {len(ev):,} events")
        events.append(ev)
    events = pd.concat(events, ignore_index=True)

    games = elo.run_elo(data.game_results(events))
    games.to_csv(PROCESSED / "games.csv", index=False)

    states = events.merge(games[["game_id", "home_team", "away_team", "elo_diff", "home_win"]],
                          on="game_id", how="inner")
    states.to_parquet(PROCESSED / "states.parquet", index=False)
    print(f"saved {len(states):,} game states from {len(games):,} games")


if __name__ == "__main__":
    main()
