"""
Draws an NBA half-court diagram and scatters shot attempts on it, colored by
the model's predicted shot quality (make probability) rather than by
make/miss -- so the chart shows *where good looks come from*, not just
where shots happened to go in.

Coordinates follow nba_api's convention: the hoop sits at (0, 0), LOC_X/
LOC_Y are in tenths of a foot, and the court extends toward positive Y.

Usage:
    python visualize.py --features ../data/processed/features.csv --player "J. Carter"
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle

from predict import predict_quality


def draw_court(ax):
    """Draws the hoop, backboard, paint, free-throw circle, restricted area,
    and three-point line using the NBA's official court dimensions
    (in tenths of a foot, matching LOC_X / LOC_Y)."""
    ax.add_patch(Circle((0, 0), radius=7.5, linewidth=1.5, color="black", fill=False))
    ax.add_patch(Rectangle((-30, -7.5), 60, -1, linewidth=1.5, color="black"))
    ax.add_patch(Rectangle((-80, -47.5), 160, 190, linewidth=1.5, color="black", fill=False))
    ax.add_patch(Rectangle((-60, -47.5), 120, 190, linewidth=1.5, color="black", fill=False))
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=1.5, color="black"))
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=1.5, color="black"))

    # Corner three straight segments + the arc between them.
    ax.plot([-220, -220], [-47.5, 92.5], linewidth=1.5, color="black")
    ax.plot([220, 220], [-47.5, 92.5], linewidth=1.5, color="black")
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=1.5, color="black"))

    ax.add_patch(Rectangle((-250, -47.5), 500, 470, linewidth=1.5, color="black", fill=False))

    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 430)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_shot_quality(df: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 8.5))
    draw_court(ax)

    scatter = ax.scatter(
        df["LOC_X"], df["LOC_Y"],
        c=df["predicted_quality"], cmap="RdYlGn",
        vmin=0.25, vmax=0.70,
        s=28, alpha=0.75, edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted make probability (shot quality)")

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved shot chart -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a shot-quality chart")
    parser.add_argument("--features", default="../data/processed/features.csv")
    parser.add_argument("--player", default=None, help="Filter to one player (optional)")
    parser.add_argument("--out", default="../output/shot_chart.png")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if args.player:
        df = df[df["PLAYER_NAME"] == args.player]
        title = f"Shot Quality — {args.player}"
    else:
        title = "Shot Quality — All Shots"

    df = df.copy()
    df["predicted_quality"] = predict_quality(df)

    plot_shot_quality(df, title, args.out)


if __name__ == "__main__":
    main()
