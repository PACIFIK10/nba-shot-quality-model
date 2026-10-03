"""
Turns a raw shot-attempt row into the feature vector the model trains on.

Five signals per shot:
  shot_distance      -- feet from the basket
  shot_angle         -- degrees off the basket's center line (0 = straight on)
  defender_distance  -- feet to the nearest defender at release
  shot_clock         -- seconds left on the shot clock at release
  seconds_remaining  -- seconds left in the period (game-situation pressure)

defender_distance and shot_clock come from the NBA's player-tracking data,
which isn't exposed by the public stats API -- real pulls that lack them
fall back to a neutral default (see IMPUTED_* below) so the pipeline still
runs; the synthetic dataset carries real per-shot values for both.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "shot_distance", "shot_angle", "defender_distance",
    "shot_clock", "seconds_remaining",
]

IMPUTED_DEFENDER_DISTANCE = 4.0  # league-average-ish open/contested midpoint
IMPUTED_SHOT_CLOCK = 14.0        # roughly the middle of a possession


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["shot_distance"] = df["SHOT_DISTANCE"].astype(float)

    # Angle off the basket's center line: 0 = straight-on, 90 = along the baseline.
    loc_x = df["LOC_X"].astype(float)
    loc_y = df["LOC_Y"].astype(float).clip(lower=0.1)
    out["shot_angle"] = np.degrees(np.arctan2(loc_x.abs(), loc_y))

    if "DEFENDER_DISTANCE" in df.columns:
        out["defender_distance"] = df["DEFENDER_DISTANCE"].astype(float)
    else:
        out["defender_distance"] = IMPUTED_DEFENDER_DISTANCE

    if "SHOT_CLOCK" in df.columns:
        out["shot_clock"] = df["SHOT_CLOCK"].astype(float)
    else:
        out["shot_clock"] = IMPUTED_SHOT_CLOCK

    out["seconds_remaining"] = (
        df["MINUTES_REMAINING"].astype(float) * 60 + df["SECONDS_REMAINING"].astype(float)
    )

    out["made"] = df["SHOT_MADE_FLAG"].astype(int)

    # Carry a few descriptive columns through for the visualization step.
    for passthrough in ("LOC_X", "LOC_Y", "SHOT_ZONE_BASIC", "SHOT_TYPE", "PLAYER_NAME"):
        if passthrough in df.columns:
            out[passthrough] = df[passthrough]

    return out


def build_dataset(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    return build_features(df)


if __name__ == "__main__":
    dataset = build_dataset("../data/raw/shots.csv")
    dataset.to_csv("../data/processed/features.csv", index=False)
    print(f"Wrote {len(dataset)} feature rows -> ../data/processed/features.csv")
