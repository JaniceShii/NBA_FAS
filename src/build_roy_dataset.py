"""
build_roy_dataset.py

Collects historical rookie data from NBA.com using the open-source `nba_api` package.

What this script does:
- Loops over a range of NBA seasons (e.g., 1996-97 to 2023-24)
- For each season:
    - Downloads league-wide player stats (regular season)
    - Filters to rookies (using PLAYER_EXPERIENCE or EXP)
    - Downloads team stats for context (wins, win%, team points per game)
    - Adds a few engineered features (per-36 stats, scoring share, etc.)
    - Optionally enriches with player draft/position info via `CommonPlayerInfo`
- Concatenates all seasons into a single DataFrame
- Saves the result to CSV for downstream modeling (e.g., ROY prediction)

Requirements:
    pip install nba_api pandas numpy

NOTE:
- NBA.com can rate-limit or block if you send too many requests too fast.
  This script sleeps between requests to be polite.
"""

import time
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from requests.exceptions import ReadTimeout

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    commonplayerinfo,
)


# ---------------------- CONFIGURATION ---------------------- #

# rebounds, assists, points, steals, blocks, turnovers, gp, min, plus_minus

# First season's starting year, e.g. 1996 -> "1996-97"
START_YEAR = 1986
# if 86 has all data stats


# Last season's starting year (inclusive).
# Example: 2023 -> covers "2023-24"
END_YEAR = 2024

# Where to write the final dataset
OUTPUT_CSV = "rookie_dataset.csv"

# Delay between HTTP requests (in seconds)
API_SLEEP_SECONDS = 1.0

# Whether to call CommonPlayerInfo for every player (slow / may timeout)
# For the challenge, it's safer to keep this False.
ENRICH_PLAYER_INFO = False


# ---------------------- HELPER FUNCTIONS ---------------------- #

def make_season_string(start_year: int) -> str:
    """
    Convert a starting year like 1996 into an NBA season string "1996-97".
    """
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_team_stats_for_season(season: str) -> pd.DataFrame:
    """
    Fetch team-level stats (per-game) for a given season.
    Includes wins, win%, and team points per game.
    Handles missing columns gracefully.
    """
    print(f"  -> Fetching team stats for {season} ...")
    resp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    )
    df = resp.get_data_frames()[0]

    # We only keep columns we care about for context, IF they exist
    desired_cols = [
        "TEAM_ID",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "GP",
        "W",
        "L",
        "W_PCT",
        "PTS",  # team points per game
    ]

    existing_cols = [c for c in desired_cols if c in df.columns]
    if len(existing_cols) < len(desired_cols):
        missing = set(desired_cols) - set(existing_cols)
        print(f"    WARNING: Missing team columns for {season}: {missing}")

    df = df[existing_cols].copy()

    # Rename PTS -> TEAM_PTS if present
    if "PTS" in df.columns:
        df.rename(columns={"PTS": "TEAM_PTS"}, inplace=True)

    return df


def fetch_rookie_player_stats_for_season(season: str) -> pd.DataFrame:
    """
    Fetch league-wide player stats for a season and filter to rookies.
    Uses PLAYER_EXPERIENCE == 'R' or EXP == 0 if available.
    """
    print(f"  -> Fetching player stats for {season} ...")
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    )
    df = resp.get_data_frames()[0]

    # Filter to rookies based on available experience column
    if "PLAYER_EXPERIENCE" in df.columns:
        rookies = df[df["PLAYER_EXPERIENCE"] == "R"].copy()
    elif "EXP" in df.columns:
        rookies = df[df["EXP"] == 0].copy()
    else:
        # Fallback: keep everyone and let user handle later (unlikely case)
        print("    WARNING: No PLAYER_EXPERIENCE or EXP column found. Not filtering.")
        rookies = df.copy()

    # Add season column
    rookies["SEASON"] = season

    # Drop players with extremely low minutes/games (tiny sample size)
    if "GP" in rookies.columns and "MIN" in rookies.columns:
        rookies = rookies[(rookies["GP"] >= 10) & (rookies["MIN"] > 0)]

    return rookies


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add some simple engineered features for modeling.
    - Per-36 scoring / rebounding / assists / steals / blocks
    - Scoring share vs team points
    - Simple "impact score"
    """
    df = df.copy()

    # Avoid division by zero
    min_per_game = df["MIN"].replace(0, np.nan)

    # Per-36 stats
    df["PTS_per36"] = df["PTS"] * 36.0 / min_per_game
    df["REB_per36"] = df.get("REB", 0) * 36.0 / min_per_game
    df["AST_per36"] = df.get("AST", 0) * 36.0 / min_per_game
    df["STL_per36"] = df.get("STL", 0) * 36.0 / min_per_game
    df["BLK_per36"] = df.get("BLK", 0) * 36.0 / min_per_game

    # Scoring share relative to team PPG
    if "TEAM_PTS" in df.columns:
        df["SCORING_SHARE"] = df["PTS"] / df["TEAM_PTS"]
    else:
        df["SCORING_SHARE"] = np.nan

    # Simple "impact" metric (toy feature): points + rebounds + 2*assists per game
    df["IMPACT_SCORE"] = df["PTS"] + df.get("REB", 0) + 2 * df.get("AST", 0)

    return df


def fetch_player_info(
    player_id: int,
    cache: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fetch player biographical/draft info via CommonPlayerInfo.
    Uses an in-memory cache so each player is fetched at most once.
    Includes basic timeout handling to avoid crashing.
    """
    if player_id in cache:
        return cache[player_id]

    print(f"    -> Fetching CommonPlayerInfo for PLAYER_ID={player_id}")
    try:
        resp = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        info_df = resp.get_data_frames()[0]
        row = info_df.iloc[0]

        data = {
            "PLAYER_ID": int(row["PERSON_ID"]),
            "POSITION": row.get("POSITION", None),
            "HEIGHT": row.get("HEIGHT", None),
            "WEIGHT": row.get("WEIGHT", None),
            "DRAFT_YEAR": row.get("DRAFT_YEAR", None),
            "DRAFT_ROUND": row.get("DRAFT_ROUND", None),
            "DRAFT_NUMBER": row.get("DRAFT_NUMBER", None),
            "SCHOOL": row.get("SCHOOL", None),
            "COUNTRY": row.get("COUNTRY", None),
        }
    except ReadTimeout:
        print(f"    WARNING: timeout on PLAYER_ID={player_id}, filling Nones")
        data = {
            "PLAYER_ID": player_id,
            "POSITION": None,
            "HEIGHT": None,
            "WEIGHT": None,
            "DRAFT_YEAR": None,
            "DRAFT_ROUND": None,
            "DRAFT_NUMBER": None,
            "SCHOOL": None,
            "COUNTRY": None,
        }

    cache[player_id] = data
    return data


def enrich_with_player_info(
    df: pd.DataFrame,
    cache: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Merge player biographical/draft info onto the season-level rookie stats.
    """
    df = df.copy()
    if "PLAYER_ID" not in df.columns:
        print("WARNING: PLAYER_ID column missing; cannot enrich with player info.")
        return df

    unique_ids = df["PLAYER_ID"].unique()
    info_rows: List[Dict[str, Any]] = []

    for pid in unique_ids:
        # Be nice to the API; sleep between calls
        time.sleep(API_SLEEP_SECONDS)
        info_rows.append(fetch_player_info(int(pid), cache))

    info_df = pd.DataFrame(info_rows)

    # Merge on PLAYER_ID
    merged = df.merge(info_df, on="PLAYER_ID", how="left")
    return merged


# ---------------------- MAIN PIPELINE ---------------------- #

def main():
    seasons = [make_season_string(y) for y in range(START_YEAR, END_YEAR + 1)]
    print("Seasons to fetch:", seasons)

    all_season_frames = []
    player_info_cache: Dict[int, Dict[str, Any]] = {}

    for season in seasons:
        print(f"\n=== Season {season} ===")

        # 1) Team stats for context
        team_df = fetch_team_stats_for_season(season)
        time.sleep(API_SLEEP_SECONDS)

        # 2) Rookie player stats
        rookie_stats = fetch_rookie_player_stats_for_season(season)
        time.sleep(API_SLEEP_SECONDS)

        if rookie_stats.empty:
            print(f"  -> No rookies found for {season} (after filters). Skipping.")
            continue

        # 3) Merge team context (wins, win%, team points per game)
        merged = rookie_stats.merge(
            team_df,
            on="TEAM_ID",
            how="left",
            suffixes=("", "_TEAM"),
        )

        # 4) Add engineered features
        merged = add_engineered_features(merged)

        # 5) Enrich with player info (draft, position, etc.) – optional
        if ENRICH_PLAYER_INFO:
            merged = enrich_with_player_info(merged, player_info_cache)

        all_season_frames.append(merged)

    if not all_season_frames:
        print("No data collected. Check configuration or API responses.")
        return

    final_df = pd.concat(all_season_frames, ignore_index=True)

    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Saving to {OUTPUT_CSV} ...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
