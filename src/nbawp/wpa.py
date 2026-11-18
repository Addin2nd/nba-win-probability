"""Win Probability Added (WPA): turning a WP curve into player and game stories."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FREE_THROW, MADE_SHOT, MISSED_SHOT, REBOUND, TURNOVER


def add_deltas(pred: pd.DataFrame) -> pd.DataFrame:
    """Change in home win probability caused by each event.

    The first event of a game is measured against the pre-game probability,
    approximated by the model's value at tip-off.
    """
    pred = pred.copy()
    prev = pred.groupby("game_id")["wp_home"].shift(1)
    pred["delta_home"] = (pred["wp_home"] - prev).fillna(0.0)
    return pred


def credit_plays(pred: pd.DataFrame) -> pd.DataFrame:
    """Assign each WP swing to the player responsible, from that player's side.

    * made shot / free throw -> shooter
    * turnover -> player who lost the ball
    * missed shot -> the swing only happens on the defensive rebound that ends
      the possession, so that swing is charged to the *shooter*, not credited
      to the rebounder (most defensive rebounds are uncontested).
    """
    p = add_deltas(pred)
    prev_type = p.groupby("game_id")["event_type"].shift(1)
    prev_player = p.groupby("game_id")["player"].shift(1)
    prev_pid = p.groupby("game_id")["player_id"].shift(1)
    prev_team = p.groupby("game_id")["team"].shift(1)
    prev_side = p.groupby("game_id")["side"].shift(1)

    direct = p["event_type"].isin([MADE_SHOT, FREE_THROW, TURNOVER]) & p["player"].notna()
    after_miss = ((p["event_type"] == REBOUND)
                  & prev_type.isin([MISSED_SHOT, FREE_THROW])
                  & (p["side"] != prev_side)
                  & prev_player.notna())

    credited = pd.concat([
        pd.DataFrame({"game_id": p.loc[direct, "game_id"], "player": p.loc[direct, "player"],
                      "player_id": p.loc[direct, "player_id"], "team": p.loc[direct, "team"],
                      "wpa": p.loc[direct, "delta_home"] * p.loc[direct, "side"],
                      "event_type": p.loc[direct, "event_type"]}),
        pd.DataFrame({"game_id": p.loc[after_miss, "game_id"], "player": prev_player[after_miss],
                      "player_id": prev_pid[after_miss], "team": prev_team[after_miss],
                      "wpa": p.loc[after_miss, "delta_home"] * prev_side[after_miss],
                      "event_type": MISSED_SHOT}),
    ])
    return credited


def player_leaderboard(credited: pd.DataFrame, team: str | None = None, min_games: int = 20) -> pd.DataFrame:
    df = credited if team is None else credited[credited["team"] == team]
    agg = df.groupby(["player_id", "player"]).agg(
        team=("team", lambda s: s.mode().iat[0]),
        games=("game_id", "nunique"),
        wpa=("wpa", "sum"),
        plays=("wpa", "size"),
    ).reset_index()
    agg = agg[agg["games"] >= min_games]
    agg["wpa_per_game"] = agg["wpa"] / agg["games"]
    return agg.sort_values("wpa", ascending=False).reset_index(drop=True)


def game_summaries(pred: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per game for ``team``: result, lowest/highest WP, excitement."""
    p = add_deltas(pred)
    p = p[(p["home_team"] == team) | (p["away_team"] == team)].copy()
    is_home = p["home_team"] == team
    p["wp_team"] = np.where(is_home, p["wp_home"], 1 - p["wp_home"])
    rows = []
    for gid, g in p.groupby("game_id", sort=True):
        home = g["home_team"].iat[0] == team
        last = g.iloc[-1]
        team_pts = last["home_score"] if home else last["away_score"]
        opp_pts = last["away_score"] if home else last["home_score"]
        rows.append({
            "game_id": int(gid),
            "opponent": g["away_team"].iat[0] if home else g["home_team"].iat[0],
            "home": bool(home),
            "team_pts": int(team_pts), "opp_pts": int(opp_pts),
            "won": bool(team_pts > opp_pts),
            "min_wp": float(g["wp_team"].min()),
            "max_wp": float(g["wp_team"].max()),
            "excitement": float(g["delta_home"].abs().sum()),
        })
    return pd.DataFrame(rows)


def biggest_plays(pred: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    p = add_deltas(pred)
    p = p[(p["home_team"] == team) | (p["away_team"] == team)].copy()
    sign = np.where(p["home_team"] == team, 1, -1)
    p["wpa_team"] = p["delta_home"] * sign
    p = p[p["team"] == team]
    cols = ["game_id", "period", "clock", "player", "description", "wpa_team",
            "home_team", "away_team", "home_score", "away_score"]
    return p.nlargest(n, "wpa_team")[cols].reset_index(drop=True)
