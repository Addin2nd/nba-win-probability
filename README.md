# When is a lead actually safe? NBA win probability, play by play

[![CI](https://github.com/Addin2nd/nba-win-probability/actions/workflows/ci.yml/badge.svg)](https://github.com/Addin2nd/nba-win-probability/actions/workflows/ci.yml)

An in-game win probability model for the NBA, trained on **3.3 million game states** from six seasons and tested on a season it never saw. It's then used to tell the story of the **Golden State Warriors' 2025-26 season**, the most recent complete season heading into 2026-27: which comebacks were really unlikely, which plays swung games, and who moved the needle most.

**▶ Live dashboard: https://addin2nd.github.io/nba-win-probability/**

![Dashboard preview](docs/img/dashboard.png)

## Results (test season 2025-26, never used for training or selection)

| Model | Log loss | Brier | AUC | Calibration error (ECE) |
|---|---:|---:|---:|---:|
| **Logistic regression (selected)** | **0.436** | **0.145** | **0.872** | **0.012** |
| Gradient boosting (monotonic) | 0.437 | 0.145 | 0.871 | 0.014 |
| Pre-game Elo only (baseline) | 0.599 | 0.206 | 0.738 | 0.042 |

* The in-game model cuts the Brier score by **30%** compared with knowing only pre-game team strength.
* The interpretable logistic model **matched or beat gradient-boosted trees** on both the validation and test seasons once the features encoded the right basketball structure: a lead measured relative to the time left (`margin / √(minutes + 1)`), and a pre-game prior that fades as the clock runs.
* It's well calibrated: when it says 70%, the home team wins about 70% of the time (ECE 0.012).
* Error drops from a Brier of 0.20 in the first quarter to 0.04 in the last minute.

### What it says about the 2025-26 Warriors (37-45)

* **Stephen Curry ranked #1 in the NBA in WPA per game** (+0.31) among 340 players with 40+ games. He played only 43 games, so on season totals he's 25th.
* **Biggest comeback:** won at Phoenix 101-97 after trailing 90-76 early in the fourth quarter, when their win chance was **3.6%**.
* **Worst collapse:** lost at home to Chicago 124-130 in overtime after leading by 8 with 91 seconds left, when their win chance was **98.3%**.
* **Biggest single play:** Al Horford's three with 45 seconds left at Houston, **+31.4** win-probability points. The Warriors went on to win in overtime.

## How it works

```
play-by-play (stats.nba.com) ──► clean + infer possession ──► game states
                                        │
results ──► Elo ratings (past games only) ──► pre-game strength
                                        │
          logistic regression  vs  gradient boosting
          train 2019-24 · select on 2024-25 · test once on 2025-26
                                        │
             WP after every play ──► WPA by player, game swings, dashboard
```

1. **Data.** Play-by-play for 2019-20 to 2025-26 (8,286 games) and shot locations from the public [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) archive, a mirror of stats.nba.com. Downloaded by `scripts/build_dataset.py`; nothing large is committed. From 2025-26 the NBA only publishes its newer **v3** play-by-play feed, so `v3_to_classic()` converts it to the classic layout (event codes, clock, score, home/away, full player names joined from shot data), and every season goes through the same pipeline.
2. **Game state** for every event: score margin, seconds left, overtime flag, and **who has the ball**, inferred from made shots, final free throws, rebounds, turnovers and jump balls (`src/nbawp/data.py`).
3. **Pre-game strength** from an Elo rating replayed in schedule order with margin-of-victory scaling and regression to the mean between seasons, so each game only sees earlier results and nothing leaks from the future (`src/nbawp/elo.py`).
4. **Models** (`src/nbawp/model.py`):
   * Logistic regression on engineered features. Each coefficient has a basketball meaning.
   * Histogram gradient boosting with monotonic constraints, so a bigger lead, a stronger team or having the ball can never lower the win probability.
   * Seasons split by time: train 2019-24, pick the model on 2024-25, refit, score once on 2025-26.
5. **Win Probability Added** (`src/nbawp/wpa.py`). Each play's swing goes to the player responsible: made shots and free throws to the shooter, turnovers to the ball-handler, and a missed shot to the shooter once the defense secures the rebound.
6. **Shot map** (`src/nbawp/shots.py`). Curry's field-goal percentage by zone, compared with the league average in that same zone.

## Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make all     # download data (~70 MB compressed), train, rebuild docs/index.html  (~3 min)
make test    # unit tests (possession rules, v3 conversion, Elo leakage, model sanity, WPA credit)
```

## Limitations and next steps

* No lineups or injuries, so a game the star sits out is priced like any other. Adding pre-game availability is the obvious next feature.
* WPA here covers offense at the level of single plays. Defensive stops aren't credited to individual defenders, and assists aren't split from the scorer.
* Possession is inferred with rules, not read from tracking data. About 1-2% of states are "unknown", mostly right after period starts (the v3 feed doesn't say which team won a jump ball).
* Game IDs stand in for chronological order when Elo is replayed. That's accurate for NBA schedules but not exact to the day.

## Project structure

```
src/nbawp/      data.py · elo.py · model.py · wpa.py · shots.py
scripts/        build_dataset.py · train.py · build_report.py
report/         template.html  (dashboard template, data injected at build time)
docs/           index.html     (generated dashboard, served by GitHub Pages)
tests/          unit tests, run in CI on every push
```

---
Built by **Muhamad Addin Al Haadi** · Not affiliated with the NBA or the Golden State Warriors · MIT License
