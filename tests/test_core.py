import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbawp import data, elo, wpa  # noqa: E402
from nbawp.model import LogisticWP, _finalise, add_features, expected_calibration_error  # noqa: E402


def raw_events(rows):
    """Build a raw stats.nba.com-shaped frame from compact tuples."""
    cols = ["EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERSON1TYPE", "PERSON3TYPE", "SCORE"]
    df = pd.DataFrame(rows, columns=cols)
    df["GAME_ID"] = 1
    return df


def test_clock_and_time_remaining():
    clock = data.clock_to_seconds(pd.Series(["12:00", "0:05", "4:30"]))
    assert clock.tolist() == [720, 5, 270]
    left = data.seconds_remaining(pd.Series([1, 4, 5]), clock)
    assert list(left) == [2880, 5, 270]


def test_possession_rules():
    ev = raw_events([
        (12, 0, 0, 0, None),     # start of period -> unknown
        (10, 0, 4, 5, None),     # jump ball, visitor controls tip
        (1, 1, 5, 0, "2 - 0"),   # visitor scores -> home ball
        (2, 1, 4, 0, None),      # home misses -> still home until rebound
        (4, 0, 5, 0, None),      # visitor defensive rebound -> visitor
        (5, 1, 5, 0, None),      # visitor turnover -> home
        (3, 11, 5, 0, "3 - 0"),  # FT 1 of 2 made -> no change
        (3, 12, 5, 0, "4 - 0"),  # FT 2 of 2 made -> home
        (4, 0, 3, 0, None),      # visitor *team* rebound -> visitor
    ])
    assert data.infer_possession(ev).tolist() == [0, -1, 1, 1, -1, 1, 1, 1, -1]


def test_elo_uses_only_past_games():
    games = pd.DataFrame({
        "season": [2024] * 3, "game_id": [1, 2, 3],
        "home_team": ["A", "A", "B"], "away_team": ["B", "B", "A"],
        "home_score": [120, 110, 90], "away_score": [100, 100, 100],
    })
    out = elo.run_elo(games)
    assert out.loc[0, "home_elo"] == out.loc[0, "away_elo"] == elo.BASE
    assert out.loc[1, "home_elo"] > out.loc[1, "away_elo"]          # A won game 1
    assert out.loc[2, "away_elo"] > out.loc[1, "home_elo"]          # A won game 2 too


def test_finished_games_are_certain():
    X = pd.DataFrame({"seconds_left": [0, 0, 30], "margin": [3, -2, 3]})
    p = _finalise(X, np.array([0.6, 0.6, 0.6]))
    assert p[0] > 0.999 and p[1] < 0.001 and p[2] == 0.6


def _synthetic_states(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    s = pd.DataFrame({
        "seconds_left": rng.uniform(0, 2880, n),
        "margin": rng.integers(-25, 26, n),
        "possession": rng.choice([-1, 1], n),
        "period": 4,
        "elo_diff": rng.normal(70, 150, n),
    })
    logit = s["margin"] / np.sqrt(s["seconds_left"] / 60 + 1) * 0.9 + s["elo_diff"] / 400 * s["seconds_left"] / 2880
    s["home_win"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return add_features(s)


def test_logistic_model_behaves_like_basketball():
    s = _synthetic_states()
    m = LogisticWP().fit(s, s["home_win"])
    probe = add_features(pd.DataFrame({
        "seconds_left": [60, 60, 2000], "margin": [5, -5, 5],
        "possession": [1, 1, 1], "period": 4, "elo_diff": [0, 0, 0]}))
    p = m.predict(probe)
    assert p[0] > 0.85 and p[1] < 0.15          # late leads are decisive
    assert p[0] > p[2] > 0.5                    # same lead matters more late


def test_calibration_error_zero_for_perfect_forecast():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) < 1e-9


def test_wpa_charges_missed_shot_to_shooter():
    pred = pd.DataFrame({
        "game_id": 1, "event_type": [12, 2, 4, 1],
        "player": [None, "Shooter", "Rebounder", "Scorer"],
        "player_id": [None, 1, 2, 3], "team": [None, "HOM", "AWY", "AWY"],
        "side": [0, 1, -1, -1], "wp_home": [0.50, 0.50, 0.46, 0.40],
    })
    c = wpa.credit_plays(pred).set_index("player")["wpa"]
    assert np.isclose(c["Shooter"], -0.04)      # the miss cost the home team 4 points
    assert np.isclose(c["Scorer"], 0.06)        # away basket, credited from away side
    assert "Rebounder" not in c.index
