"""Loading and cleaning NBA play-by-play (stats.nba.com format).

Raw data comes from the public archive at github.com/shufinskiy/nba_data,
which mirrors stats.nba.com play-by-play and shot detail for every season
since 1996-97. `download()` fetches the seasons you ask for.

Conventions used throughout this package
-----------------------------------------
* ``season`` is the year the season *starts*: 2024 means 2024-25.
* Everything is expressed from the **home team's** perspective:
  ``margin = home_score - away_score`` and ``home_win`` is the label.
* ``side`` is +1 for the home team, -1 for the away team, 0 for neutral.
"""
from __future__ import annotations

import io
import lzma
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/{name}.tar.xz"

# stats.nba.com EVENTMSGTYPE codes
MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER = 1, 2, 3, 4, 5
FOUL, VIOLATION, SUB, TIMEOUT, JUMP_BALL = 6, 7, 8, 9, 10
START_PERIOD, END_PERIOD = 12, 13

# Free-throw action types that end a trip to the line (1 of 1, 2 of 2, 3 of 3).
LAST_FREE_THROW = {10, 12, 15}

# PERSONxTYPE: 2/4 = home team/player, 3/5 = visitor team/player.
_SIDE = {2: 1, 4: 1, 3: -1, 5: -1}

PBP_COLUMNS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD",
    "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
    "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ID",
    "PLAYER1_TEAM_ABBREVIATION", "PERSON3TYPE",
]


def download(name: str, raw_dir: Path) -> Path:
    """Download and extract one archive (e.g. ``nbastats_2024``) into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{name}.csv"
    if target.exists():
        return target
    with urllib.request.urlopen(ARCHIVE_URL.format(name=name), timeout=120) as resp:
        payload = resp.read()
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(payload))) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        with tar.extractfile(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def clock_to_seconds(clock: pd.Series) -> pd.Series:
    """'11:42' -> 702 seconds left in the period."""
    parts = clock.astype(str).str.split(":", n=1, expand=True)
    return parts[0].astype(int) * 60 + parts[1].astype(float)


def seconds_remaining(period: pd.Series, clock_seconds: pd.Series) -> pd.Series:
    """Seconds left in the game.

    Regulation: whole periods still to play plus the clock. Overtime: just the
    clock of the current 5-minute period (the game ends when it hits zero
    unless tied, which the model learns via ``margin == 0`` near zero).
    """
    regulation = (4 - period).clip(lower=0) * 720 + clock_seconds
    return np.where(period <= 4, regulation, clock_seconds)


def load_pbp(path: Path, season: int) -> pd.DataFrame:
    """Read one season of raw play-by-play and return a tidy event table."""
    df = pd.read_csv(path, usecols=PBP_COLUMNS, low_memory=False)
    df["row"] = np.arange(len(df))
    df = df[df["PCTIMESTRING"].notna()].copy()

    df["clock"] = clock_to_seconds(df["PCTIMESTRING"])
    # Rows are mostly in order already; enforce period/clock order and keep
    # the file order as a tie-breaker for events at the same second.
    df = df.sort_values(["GAME_ID", "PERIOD", "clock", "row"],
                        ascending=[True, True, False, True], kind="stable")

    # SCORE is "AWAY - HOME" and only present on scoring events.
    score = df["SCORE"].str.split(" - ", n=1, expand=True)
    df["away_score"] = pd.to_numeric(score[0], errors="coerce")
    df["home_score"] = pd.to_numeric(score[1], errors="coerce")
    df[["away_score", "home_score"]] = (
        df.groupby("GAME_ID")[["away_score", "home_score"]].ffill().fillna(0)
    )
    df["scored"] = df["SCORE"].notna()

    df["side"] = df["PERSON1TYPE"].map(_SIDE).fillna(0).astype(int)
    df["season"] = season
    df["seconds_left"] = seconds_remaining(df["PERIOD"], df["clock"])
    df["margin"] = df["home_score"] - df["away_score"]
    df["possession"] = infer_possession(df)

    out = df.rename(columns={
        "GAME_ID": "game_id", "EVENTMSGTYPE": "event_type",
        "EVENTMSGACTIONTYPE": "action_type", "PERIOD": "period",
        "PLAYER1_ID": "player_id", "PLAYER1_NAME": "player",
        "PLAYER1_TEAM_ID": "player_team_id",
        "PLAYER1_TEAM_ABBREVIATION": "team",
    })
    home_first = out["HOMEDESCRIPTION"].fillna(out["VISITORDESCRIPTION"])
    away_first = out["VISITORDESCRIPTION"].fillna(out["HOMEDESCRIPTION"])
    out["description"] = np.where(out["side"] < 0, away_first, home_first)
    keep = ["season", "game_id", "period", "clock", "seconds_left", "event_type",
            "action_type", "side", "player_id", "player", "team", "scored",
            "home_score", "away_score", "margin", "possession", "description"]
    return out[keep].reset_index(drop=True)


def infer_possession(df: pd.DataFrame) -> pd.Series:
    """Which side has the ball *after* each event (+1 home, -1 away, 0 unknown).

    Rules (a simplified possession model, good enough for win probability):
    made field goal or final made free throw -> the other team; rebound ->
    rebounding team; turnover -> the other team; jump ball -> team that
    controlled the tip. Everything else carries the previous value forward,
    and possession resets to unknown at the start of each period.
    """
    etype = df["EVENTMSGTYPE"].to_numpy()
    action = df["EVENTMSGACTIONTYPE"].to_numpy()
    side = df["PERSON1TYPE"].map(_SIDE).fillna(0).to_numpy()
    tip_side = df["PERSON3TYPE"].map(_SIDE).fillna(0).to_numpy()
    scored = df["SCORE"].notna().to_numpy()

    change = np.full(len(df), np.nan)
    change[etype == MADE_SHOT] = -side[etype == MADE_SHOT]
    last_ft = (etype == FREE_THROW) & np.isin(action, list(LAST_FREE_THROW)) & scored
    change[last_ft] = -side[last_ft]
    change[etype == REBOUND] = side[etype == REBOUND]
    change[etype == TURNOVER] = -side[etype == TURNOVER]
    change[etype == JUMP_BALL] = tip_side[etype == JUMP_BALL]
    # An event whose team we could not identify should not wipe possession.
    change[(change == 0)] = np.nan
    change[etype == START_PERIOD] = 0

    s = pd.Series(change, index=df.index)
    return s.groupby(df["GAME_ID"]).ffill().fillna(0).astype(int)


def game_results(events: pd.DataFrame) -> pd.DataFrame:
    """One row per game: teams, final score and who won."""
    last = events.groupby("game_id").tail(1).set_index("game_id")
    teams = team_sides(events)
    res = pd.DataFrame({
        "season": last["season"],
        "home_score": last["home_score"].astype(int),
        "away_score": last["away_score"].astype(int),
    }).join(teams, how="inner")
    res["home_win"] = (res["home_score"] > res["away_score"]).astype(int)
    return res.reset_index()


def team_sides(events: pd.DataFrame) -> pd.DataFrame:
    """Home/away team abbreviation for each game, from player events."""
    player_events = events[events["team"].notna() & (events["side"] != 0)]
    counts = (player_events.groupby(["game_id", "side"])["team"]
              .agg(lambda s: s.mode().iat[0]).unstack())
    return counts.rename(columns={1: "home_team", -1: "away_team"})[["home_team", "away_team"]]
