"""
Generates a synthetic shot-attempt dataset with the same schema fetch_data.py
produces, plus the two tracking-data columns (defender_distance, shot_clock)
that the public nba_api endpoints don't expose. This lets the rest of the
pipeline -- feature engineering, training, visualization -- be built and
tested end-to-end without depending on live access to stats.nba.com.

Shot outcomes aren't random: each shot's make probability is generated from
a ground-truth function of its distance, how tightly it's defended, and the
shot clock at release, so the dataset carries the same kind of signal a real
shot-quality model is trained to recover.

Usage:
    python generate_synthetic_data.py --num-shots 8000
"""

import argparse

import numpy as np
import pandas as pd

TEAMS = [
    "Los Angeles Lakers", "Boston Celtics", "Golden State Warriors",
    "Milwaukee Bucks", "Phoenix Suns", "Denver Nuggets",
    "Miami Heat", "New York Knicks",
]

PLAYERS = [
    "J. Carter", "M. Okafor", "D. Reyes", "T. Nwosu", "K. Albrecht",
    "L. Fitzgerald", "R. Delgado", "S. Whitfield", "A. Kowalski", "B. Osei",
]


def shot_zone(distance_ft: float, loc_x: float) -> str:
    if distance_ft < 4:
        return "Restricted Area"
    if distance_ft < 14:
        return "In The Paint (Non-RA)"
    if distance_ft < 22:
        return "Mid-Range"
    if abs(loc_x) > 220 and distance_ft < 24:
        return "Left Corner 3" if loc_x < 0 else "Right Corner 3"
    return "Above the Break 3"


def true_make_probability(distance_ft, defender_distance, shot_clock):
    """Ground-truth function the synthetic labels are sampled from.

    Calibrated so an average-defended shot roughly tracks real NBA
    percentages by distance (~63% at the rim, ~40% mid-range, ~36% on
    threes), with openness, shot-clock pressure, and late-clock panic
    shifting the number up or down from there.
    """
    base = 0.6535 - 0.01174 * distance_ft
    openness_bonus = np.clip(0.03 * (defender_distance - 4.2), -0.15, 0.15)
    rushed_penalty = np.where(shot_clock < 4, -0.12, 0.0)
    late_clock_penalty = np.where(shot_clock < 2, -0.08, 0.0)
    prob = base + openness_bonus + rushed_penalty + late_clock_penalty
    return np.clip(prob, 0.03, 0.92)


def sample_shot_distances(num_shots: int, rng: np.random.Generator) -> np.ndarray:
    """Mimics a real shot-distance distribution: lots of shots at the rim,
    a smaller mid-range share, and a big cluster right around the 3PT line
    -- rather than spreading shots uniformly across the whole half court."""
    zone = rng.choice(["rim", "paint", "mid", "three"], num_shots, p=[0.38, 0.14, 0.18, 0.30])
    distance = np.empty(num_shots)
    distance[zone == "rim"] = np.clip(rng.normal(2.0, 1.2, (zone == "rim").sum()), 0.2, 4)
    distance[zone == "paint"] = rng.uniform(4, 14, (zone == "paint").sum())
    distance[zone == "mid"] = rng.uniform(14, 22, (zone == "mid").sum())
    distance[zone == "three"] = np.clip(rng.normal(25.5, 1.8, (zone == "three").sum()), 22.5, 29)
    return distance


def generate(num_shots: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance_ft = sample_shot_distances(num_shots, rng)
    angle_deg = rng.uniform(-68, 68, num_shots)
    angle_rad = np.radians(angle_deg)
    loc_x = distance_ft * 10 * np.sin(angle_rad)
    loc_y = distance_ft * 10 * np.cos(angle_rad)

    defender_distance = np.clip(rng.normal(4.2, 2.3, num_shots), 0.3, 15.0)
    shot_clock = np.clip(rng.uniform(0, 24, num_shots), 0, 24)

    period = rng.integers(1, 5, num_shots)
    minutes_remaining = rng.integers(0, 12, num_shots)
    seconds_remaining = rng.integers(0, 60, num_shots)

    make_prob = true_make_probability(distance_ft, defender_distance, shot_clock)
    made = rng.binomial(1, make_prob)

    shot_type = np.where(distance_ft >= 22.5, "3PT Field Goal", "2PT Field Goal")
    zone = [shot_zone(d, x) for d, x in zip(distance_ft, loc_x)]

    df = pd.DataFrame({
        "GAME_ID": rng.integers(22300000, 22300999, num_shots),
        "PLAYER_NAME": rng.choice(PLAYERS, num_shots),
        "TEAM_NAME": rng.choice(TEAMS, num_shots),
        "PERIOD": period,
        "MINUTES_REMAINING": minutes_remaining,
        "SECONDS_REMAINING": seconds_remaining,
        "SHOT_DISTANCE": distance_ft.round(1),
        "SHOT_TYPE": shot_type,
        "SHOT_ZONE_BASIC": zone,
        "LOC_X": loc_x.round(1),
        "LOC_Y": loc_y.round(1),
        "SHOT_MADE_FLAG": made,
        "DEFENDER_DISTANCE": defender_distance.round(2),
        "SHOT_CLOCK": shot_clock.round(1),
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic shot data")
    parser.add_argument("--num-shots", type=int, default=8000)
    parser.add_argument("--out", default="../data/raw/shots.csv")
    args = parser.parse_args()

    df = generate(args.num_shots)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} synthetic shots -> {args.out}")
    print(f"Overall make rate: {df['SHOT_MADE_FLAG'].mean():.1%}")


if __name__ == "__main__":
    main()
