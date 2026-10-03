"""
Pulls real shot-attempt data from the NBA's own stats API via nba_api.

For each requested player (or team), fetches every shot attempt in a season
using the ShotChartDetail endpoint: court location, make/miss, shot distance,
shot type, period, and minutes remaining. This is the same public data source
official shot-chart tools and shot-quality models are built on top of.

Two columns the public API does not expose -- distance to the nearest
defender and shot-clock value at release -- are not available outside the
NBA's own tracking data product, so they are added separately as a synthetic
overlay (see generate_synthetic_data.py) layered on top of these real shots
when real defender-tracking access isn't available.

Usage:
    python fetch_data.py --season 2023-24 --player "Stephen Curry"
    python fetch_data.py --season 2023-24 --team "Golden State Warriors"
"""

import argparse
import sys
import time

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players, teams

OUTPUT_PATH = "../data/raw/shots.csv"


def resolve_player_id(name: str) -> int:
    matches = players.find_players_by_full_name(name)
    if not matches:
        raise ValueError(f"No player found matching '{name}'")
    return matches[0]["id"]


def resolve_team_id(name: str) -> int:
    matches = teams.find_teams_by_full_name(name)
    if not matches:
        raise ValueError(f"No team found matching '{name}'")
    return matches[0]["id"]


def fetch_shots(season: str, player_id: int = 0, team_id: int = 0) -> pd.DataFrame:
    response = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    df = response.get_data_frames()[0]
    return df[[
        "GAME_ID", "PLAYER_NAME", "TEAM_NAME", "PERIOD",
        "MINUTES_REMAINING", "SECONDS_REMAINING",
        "SHOT_DISTANCE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
        "LOC_X", "LOC_Y", "SHOT_MADE_FLAG",
    ]]


def main():
    parser = argparse.ArgumentParser(description="Fetch NBA shot-chart data")
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--player", help="Full player name, e.g. 'Stephen Curry'")
    parser.add_argument("--team", help="Full team name, e.g. 'Golden State Warriors'")
    args = parser.parse_args()

    if not args.player and not args.team:
        print("Provide --player or --team (or both: a player's shots filtered to a team).")
        sys.exit(1)

    player_id = resolve_player_id(args.player) if args.player else 0
    team_id = resolve_team_id(args.team) if args.team else 0

    print(f"Fetching {args.season} shot chart data...")
    df = fetch_shots(args.season, player_id=player_id, team_id=team_id)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} shot attempts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
