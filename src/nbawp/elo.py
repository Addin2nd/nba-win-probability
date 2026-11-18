"""Pre-game team strength with a simple Elo rating.

The in-game model needs to know how good each team is *before* tip-off, and
that number must only use games played earlier (no leakage). Elo does that in
a few lines: ratings are updated game by game in schedule order, with a
margin-of-victory multiplier and partial regression to the mean between
seasons (the FiveThirtyEight NBA recipe).
"""
from __future__ import annotations

import math

import pandas as pd

BASE = 1500.0
K = 20.0
HOME_ADVANTAGE = 70.0  # Elo points; home edge shrank after 2020
CARRY_OVER = 0.75      # share of last season's deviation kept


def expected(diff: float) -> float:
    """Win probability for a rating difference (already including home edge)."""
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    return ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)


def run_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Add pre-game ``home_elo``, ``away_elo`` and ``elo_diff`` to ``games``.

    ``games`` needs season, game_id, home_team, away_team, home_score,
    away_score. Game ids are assigned in schedule order by the NBA, so sorting
    by (season, game_id) replays the season chronologically.
    """
    games = games.sort_values(["season", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    season = None
    home_elo, away_elo = [], []

    for g in games.itertuples(index=False):
        if g.season != season:
            ratings = {t: BASE + CARRY_OVER * (r - BASE) for t, r in ratings.items()}
            season = g.season
        h = ratings.get(g.home_team, BASE)
        a = ratings.get(g.away_team, BASE)
        home_elo.append(h)
        away_elo.append(a)

        diff = h + HOME_ADVANTAGE - a
        p_home = expected(diff)
        margin = g.home_score - g.away_score
        result = 1.0 if margin > 0 else 0.0
        winner_diff = diff if margin > 0 else -diff
        shift = K * mov_multiplier(margin, winner_diff) * (result - p_home)
        ratings[g.home_team] = h + shift
        ratings[g.away_team] = a - shift

    games["home_elo"] = home_elo
    games["away_elo"] = away_elo
    games["elo_diff"] = games["home_elo"] + HOME_ADVANTAGE - games["away_elo"]
    games["elo_prob"] = [expected(d) for d in games["elo_diff"]]
    return games


def pregame_logit(elo_diff: pd.Series) -> pd.Series:
    """Elo difference expressed as log-odds, the natural scale for a logistic model."""
    return elo_diff * math.log(10) / 400.0
