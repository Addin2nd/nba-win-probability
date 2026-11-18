"""Shot-location analysis from stats.nba.com shot detail."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ZONE_COLS = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA"]


def load_shots(path: Path) -> pd.DataFrame:
    cols = ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_NAME", "PERIOD",
            "ACTION_TYPE", "SHOT_TYPE", "SHOT_DISTANCE", "LOC_X", "LOC_Y",
            "SHOT_MADE_FLAG", *ZONE_COLS]
    df = pd.read_csv(path, usecols=cols)
    df["points"] = df["SHOT_TYPE"].str.startswith("3").map({True: 3, False: 2})
    df["zone"] = df["SHOT_ZONE_BASIC"] + " | " + df["SHOT_ZONE_AREA"]
    return df


def zone_efficiency(shots: pd.DataFrame, player: str, min_attempts: int = 15) -> pd.DataFrame:
    """Player FG% and points per shot by zone, against the league in that zone."""
    league = shots.groupby("zone").agg(lg_fg=("SHOT_MADE_FLAG", "mean"))
    mine = shots[shots["PLAYER_NAME"] == player]
    z = mine.groupby("zone").agg(
        attempts=("SHOT_MADE_FLAG", "size"),
        fg=("SHOT_MADE_FLAG", "mean"),
        x=("LOC_X", "median"), y=("LOC_Y", "median"),
        points=("points", "first"),
    ).join(league)
    z["pps"] = z["fg"] * z["points"]
    z["fg_vs_league"] = z["fg"] - z["lg_fg"]
    z = z[(z["attempts"] >= min_attempts) & (z.index.str.split(" \\| ").str[0] != "Backcourt")]
    return z.sort_values("attempts", ascending=False).reset_index()


def shot_diet(shots: pd.DataFrame, team: str) -> pd.DataFrame:
    """Share of attempts by distance band for one team vs the league."""
    bands = pd.cut(shots["SHOT_DISTANCE"], [-1, 4, 14, 22, 30, 100],
                   labels=["0-4 ft", "5-14 ft", "15-22 ft", "23-30 ft", "30+ ft"])
    df = shots.assign(band=bands)
    league = df.groupby("band", observed=False).size() / len(df)
    t = df[df["TEAM_NAME"] == team]
    mine = t.groupby("band", observed=False).size() / len(t)
    return pd.DataFrame({"team": mine, "league": league}).reset_index()
