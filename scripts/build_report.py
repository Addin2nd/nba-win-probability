"""Turn model output into the interactive dashboard at docs/index.html.

Everything the page needs is embedded as JSON, so the file works from GitHub
Pages and when opened straight from disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbawp import shots, wpa  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
TEAM, TEAM_NAME, STAR = "GSW", "Golden State Warriors", "Stephen Curry"


def elapsed_seconds(period: pd.Series, clock: pd.Series) -> pd.Series:
    reg = (period.clip(upper=4) - 1) * 720 + (720 - clock)
    ot = 2880 + (period - 5) * 300 + (300 - clock)
    return np.where(period <= 4, reg, ot)


def game_curves(pred: pd.DataFrame, summaries: pd.DataFrame) -> list[dict]:
    p = wpa.add_deltas(pred)
    p = p[(p["home_team"] == TEAM) | (p["away_team"] == TEAM)].copy()
    home = p["home_team"] == TEAM
    p["wp"] = np.where(home, p["wp_home"], 1 - p["wp_home"]).round(3)
    p["t"] = elapsed_seconds(p["period"], p["clock"]).astype(int)
    p["us"] = np.where(home, p["home_score"], p["away_score"]).astype(int)
    p["them"] = np.where(home, p["away_score"], p["home_score"]).astype(int)
    p["swing"] = (p["delta_home"] * np.where(home, 1, -1)).round(3)
    keep = (p["swing"].abs() >= 0.002) | p["event_type"].isin([12, 13])
    p = p[keep]
    meta = summaries.set_index("game_id")
    games = []
    for i, (gid, g) in enumerate(p.groupby("game_id", sort=True), start=1):
        m = meta.loc[gid]
        top = g.reindex(g["swing"].abs().sort_values(ascending=False).index).head(3)
        games.append({
            "id": int(gid), "n": i, "opp": m["opponent"], "home": bool(m["home"]),
            "pts": int(m["team_pts"]), "opp_pts": int(m["opp_pts"]), "won": bool(m["won"]),
            "min_wp": round(float(m["min_wp"]), 3), "max_wp": round(float(m["max_wp"]), 3),
            "excitement": round(float(m["excitement"]), 2),
            "t": g["t"].tolist(), "wp": g["wp"].tolist(),
            "score": (g["us"].astype(str) + "-" + g["them"].astype(str)).tolist(),
            "desc": g["description"].fillna("").str.slice(0, 70).tolist(),
            "top": [{"t": int(r.t), "wp": float(r.wp), "swing": float(r.swing),
                     "desc": str(r.description)[:70]} for r in top.itertuples()],
        })
    return games


def main() -> None:
    metrics = json.loads((PROCESSED / "metrics.json").read_text())
    pred = pd.read_parquet(PROCESSED / "test_predictions.parquet")
    credited = wpa.credit_plays(pred)
    league = wpa.player_leaderboard(credited, min_games=40)
    team = wpa.player_leaderboard(credited, TEAM, min_games=20)
    summaries = wpa.game_summaries(pred, TEAM)
    plays = wpa.biggest_plays(pred, TEAM, n=8)

    shot_df = shots.load_shots(ROOT / "data" / "raw" / "shotdetail_2024.csv")
    zones = shots.zone_efficiency(shot_df, STAR)
    diet = shots.shot_diet(shot_df, TEAM_NAME)
    star_shots = shot_df[shot_df["PLAYER_NAME"] == STAR]

    star_rank = int(league.index[league["player"] == STAR][0]) + 1
    payload = {
        "team": TEAM, "team_name": TEAM_NAME, "star": STAR, "season": "2024-25",
        "record": [int(summaries["won"].sum()), int((~summaries["won"]).sum())],
        "metrics": metrics,
        "games": game_curves(pred, summaries),
        "team_wpa": team[["player", "games", "wpa", "wpa_per_game"]].round(3).to_dict(orient="records"),
        "league_wpa": league.head(15)[["player", "team", "games", "wpa"]].round(3).to_dict(orient="records"),
        "star_rank": star_rank,
        "n_players": int(len(league)),
        "plays": plays.assign(wpa_team=plays["wpa_team"].round(3)).to_dict(orient="records"),
        "zones": zones.round(3).to_dict(orient="records"),
        "star_fg": {"fga": int(len(star_shots)), "fg": round(float(star_shots["SHOT_MADE_FLAG"].mean()), 3),
                    "fg3": round(float(star_shots.loc[star_shots["points"] == 3, "SHOT_MADE_FLAG"].mean()), 3),
                    "fg3a": int((star_shots["points"] == 3).sum())},
        "diet": diet.round(3).to_dict(orient="records"),
    }
    template = (ROOT / "report" / "template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), default=str))
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
