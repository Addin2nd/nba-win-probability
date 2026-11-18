"""In-game win probability models.

Two candidates are trained on the same game states and compared out of sample:

* ``LogisticWP`` - logistic regression on a handful of engineered features.
  Fully interpretable: every coefficient has a basketball meaning.
* ``BoostedWP`` - gradient-boosted trees on raw state with monotonic
  constraints (more points / better team / having the ball can never lower
  the home team's win probability).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .elo import pregame_logit

GAME_SECONDS = 48 * 60


def add_features(states: pd.DataFrame) -> pd.DataFrame:
    """Engineered features. ``states`` needs margin, seconds_left, possession, period, elo_diff."""
    s = states.copy()
    minutes = s["seconds_left"] / 60.0
    s["prior"] = pregame_logit(s["elo_diff"])
    s["time_frac"] = (s["seconds_left"].clip(upper=GAME_SECONDS) / GAME_SECONDS)
    s["prior_x_time"] = s["prior"] * s["time_frac"]
    # A 5-point lead is worth far more with 1 minute left than with 40.
    s["margin_per_time"] = s["margin"] / np.sqrt(minutes + 1.0)
    s["poss_per_time"] = s["possession"] / np.sqrt(minutes + 1.0)
    s["is_ot"] = (s["period"] > 4).astype(int)
    return s


LOGIT_FEATURES = ["margin", "margin_per_time", "prior", "prior_x_time",
                  "possession", "poss_per_time", "time_frac", "is_ot"]
TREE_FEATURES = ["margin", "seconds_left", "prior", "possession", "period"]
TREE_MONOTONIC = [1, 0, 1, 1, 0]


@dataclass
class LogisticWP:
    name: str = "Logistic regression"
    features: list[str] = field(default_factory=lambda: list(LOGIT_FEATURES))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticWP:
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        self.pipe.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _finalise(X, self.pipe.predict_proba(X[self.features])[:, 1])

    def coefficients(self) -> pd.Series:
        lr = self.pipe[-1]
        return pd.Series(lr.coef_[0], index=self.features).sort_values(key=abs, ascending=False)


@dataclass
class BoostedWP:
    name: str = "Gradient boosting"
    features: list[str] = field(default_factory=lambda: list(TREE_FEATURES))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BoostedWP:
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0,
            monotonic_cst=TREE_MONOTONIC, random_state=7,
        )
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _finalise(X, self.model.predict_proba(X[self.features])[:, 1])


def _finalise(X: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """Pin decided games to 0/1 when the final buzzer has sounded with a lead."""
    p = p.copy()
    over = (X["seconds_left"].to_numpy() < 1) & (X["margin"].to_numpy() != 0)
    p[over] = (X["margin"].to_numpy()[over] > 0).astype(float)
    return np.clip(p, 1e-4, 1 - 1e-4)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "ece": expected_calibration_error(y, p),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    out = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)
