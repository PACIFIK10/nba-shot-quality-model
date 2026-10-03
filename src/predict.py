"""
Loads the trained model and scores new shots with a predicted make
probability -- the "shot quality" number the rest of the project is built
around.
"""

import pandas as pd
import xgboost as xgb

from features import FEATURE_COLUMNS

MODEL_PATH = "../models/shot_quality_model.json"

_model = None


def get_model() -> xgb.XGBClassifier:
    global _model
    if _model is None:
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
    return _model


def predict_quality(features_df: pd.DataFrame) -> pd.Series:
    """features_df must contain the FEATURE_COLUMNS; returns make probability per row."""
    model = get_model()
    probs = model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1]
    return pd.Series(probs, index=features_df.index, name="predicted_quality")


def predict_one(shot_distance, shot_angle, defender_distance, shot_clock, seconds_remaining) -> float:
    row = pd.DataFrame([{
        "shot_distance": shot_distance,
        "shot_angle": shot_angle,
        "defender_distance": defender_distance,
        "shot_clock": shot_clock,
        "seconds_remaining": seconds_remaining,
    }])
    return float(predict_quality(row).iloc[0])


if __name__ == "__main__":
    # Example: a wide-open 3 with plenty of shot clock vs. a heavily
    # contested, late-clock mid-range heave.
    open_three = predict_one(26, 10, 7.5, 18, 400)
    contested_midrange = predict_one(17, 25, 1.5, 2, 15)
    print(f"Open 3, early shot clock:         {open_three:.1%}")
    print(f"Contested mid-range, clock winding down: {contested_midrange:.1%}")
