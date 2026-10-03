# NBA Shot Quality Model

A machine learning project that scores every shot attempt by its quality —
the probability it goes in, given where it was taken from and the defensive
pressure on it — rather than just looking at whether it actually went in.
The result is rendered as a shot chart that shows where good looks come
from, independent of a player's shooting luck on any given night.

**Tech stack:** `nba_api` (shot location data) → `pandas` (feature
pipeline) → `XGBoost` (gradient boosting classifier) → `scikit-learn`
(evaluation) → `matplotlib` (shot chart visualization).

## How it works

1. **Data** — `src/fetch_data.py` pulls shot-by-shot attempt data (court
   location, make/miss, period, shot distance) for a player or team and
   season via `nba_api`'s shot chart endpoint.
2. **Features** — `src/features.py` converts each shot into 5 signals: shot
   distance, shot angle off the basket, distance to the nearest defender,
   shot clock at release, and seconds remaining in the period.
3. **Model** — `src/train.py` trains a gradient boosting classifier
   (XGBoost) that maps those 5 signals to a predicted make probability —
   the shot's "quality" score — and reports validation accuracy, AUC, and
   per-feature importance.
4. **Inference** — `src/predict.py` loads the trained model and scores new
   shots on demand.
5. **Visualization** — `src/visualize.py` draws a half-court diagram and
   plots every shot at its real court location, colored on a red-to-green
   scale by predicted quality rather than by make/miss.

## Project structure

```
nba-shot-quality-model/
├── data/
│   ├── raw/            # shots.csv — shot-by-shot attempt data
│   └── processed/      # features.csv — engineered feature rows
├── models/              # shot_quality_model.json, metrics.json
├── src/
│   ├── fetch_data.py              # nba_api shot chart data pull
│   ├── generate_synthetic_data.py # offline stand-in, same schema
│   ├── features.py                # shot -> feature-row engineering
│   ├── train.py                   # XGBoost training loop + eval
│   ├── predict.py                 # load model, score new shots
│   └── visualize.py               # half-court shot quality chart
├── output/               # shot_chart.png
└── requirements.txt
```
