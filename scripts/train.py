"""Train and compare win-probability models with a strict time split.

    train: 2019-20 .. 2023-24   validate: 2024-25   test: 2025-26

Model choice uses the validation season only. The chosen model is then refit
on train+validation and scored once on the untouched 2025-26 test season.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbawp.model import BoostedWP, LogisticWP, add_features, calibration_table, evaluate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
TRAIN, VALID, TEST = [2019, 2020, 2021, 2022, 2023], 2024, 2025


def time_bucket(seconds_left: pd.Series) -> pd.Series:
    bins = [-1, 60, 300, 720, 1440, 2160, 10_000]
    labels = ["last 1 min", "1-5 min", "5-12 min (Q4)", "Q3", "Q2", "Q1"]
    return pd.cut(seconds_left, bins=bins, labels=labels)


def main() -> None:
    states = add_features(pd.read_parquet(PROCESSED / "states.parquet"))
    train = states[states["season"].isin(TRAIN)]
    valid = states[states["season"] == VALID]
    test = states[states["season"] == TEST]
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,} states")

    report: dict = {"split": {"train": "2019-20 to 2023-24", "valid": "2024-25", "test": "2025-26"},
                    "n_fit_states": int(len(train) + len(valid)), "n_fit_seasons": len(TRAIN) + 1}

    # 1) baselines + candidates on validation
    candidates = [LogisticWP(), BoostedWP()]
    valid_scores = {}
    y_valid = valid["home_win"].to_numpy()
    for m in candidates:
        m.fit(train, train["home_win"])
        valid_scores[m.name] = evaluate(y_valid, m.predict(valid))
    valid_scores["Pre-game Elo only"] = evaluate(
        y_valid, np.clip(1 / (1 + 10 ** (-valid["elo_diff"] / 400)), 1e-4, 1 - 1e-4))
    report["validation"] = valid_scores
    print(pd.DataFrame(valid_scores).T.round(4))

    best = min(candidates, key=lambda m: valid_scores[m.name]["log_loss"])
    print(f"selected: {best.name}")

    # 2) refit everything on train+valid, score once on test
    full = pd.concat([train, valid])
    y_test = test["home_win"].to_numpy()
    test_scores, preds, fitted = {}, {}, {}
    for m in [LogisticWP(), BoostedWP()]:
        fitted[m.name] = m.fit(full, full["home_win"])
        preds[m.name] = m.predict(test)
        test_scores[m.name] = evaluate(y_test, preds[m.name])
    test_scores["Pre-game Elo only"] = evaluate(
        y_test, np.clip(1 / (1 + 10 ** (-test["elo_diff"] / 400)), 1e-4, 1 - 1e-4))
    report["test"] = test_scores
    report["selected"] = best.name
    print(pd.DataFrame(test_scores).T.round(4))

    p = preds[best.name]
    report["calibration"] = calibration_table(y_test, p).round(4).to_dict(orient="list")
    by_time = (pd.DataFrame({"bucket": time_bucket(test["seconds_left"]), "y": y_test, "p": p})
               .groupby("bucket", observed=True)[["y", "p"]]
               .apply(lambda d: pd.Series(evaluate(d["y"].to_numpy(), d["p"].to_numpy())))
               .round(4))
    report["by_time"] = by_time.reset_index().to_dict(orient="list")
    print(by_time)

    report["logit_coefficients"] = fitted["Logistic regression"].coefficients().round(4).to_dict()
    final = fitted[best.name]

    (PROCESSED / "metrics.json").write_text(json.dumps(report, indent=2))
    with open(PROCESSED / "model.pkl", "wb") as f:
        pickle.dump(final, f)

    test_out = test[["game_id", "period", "clock", "seconds_left", "event_type", "side",
                     "player_id", "player", "team", "home_team", "away_team", "home_score",
                     "away_score", "margin", "possession", "description", "home_win"]].copy()
    test_out["wp_home"] = p
    test_out.to_parquet(PROCESSED / "test_predictions.parquet", index=False)
    print("saved model, metrics and 2025-26 predictions")


if __name__ == "__main__":
    main()
