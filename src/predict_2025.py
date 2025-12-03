"""
predict_2025.py

Use the trained ROY model (data/roy_model.pkl) to generate predicted
probabilities of winning ROY for a given season (e.g. 2025-26).

Expected files:
- data/rookie_dataset.csv : created by build_roy_dataset.py
- data/roy_model.pkl      : trained model from train_roy.py

Output:
- predictions.csv         : two columns -> player_name, probability
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ---------------------- CONFIG ---------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

ROOKIE_CSV = DATA_DIR / "rookie_dataset.csv"
MODEL_PATH = DATA_DIR / "roy_model.pkl"
OUTPUT_CSV = PROJECT_ROOT / "predictions.csv"

TARGET_SEASON = "2025-26"

FEATURES = [
    "PTS",         # points per game
    "PTS_per36",   # points per 36

    "REB",         # rebounds per game
    "REB_per36",   # rebounds per 36

    "AST",         # assists per game
    "AST_per36",   # assists per 36

    "MIN",         # minutes per game

    "STL",         # steals per game
    "STL_per36",   # steals per 36

    "BLK",         # blocks per game
    "BLK_per36",   # blocks per 36

    "TOV",         # turnovers per game

    "GP",          # games played

    "EFG_PCT",     # effective FG%
    "PLUS_MINUS",  # on/off plus-minus

    "TS_PCT",      # true shooting percentage (computed)
]



# ---------------------- MAIN ---------------------- #

def main():
    # 1. Load rookie dataset
    print(f"Loading rookie data from {ROOKIE_CSV} ...")
    df = pd.read_csv(ROOKIE_CSV)
    print("Full dataset shape:", df.shape)

    # 2. Compute TS_PCT exactly like in train_roy.py
    if {"PTS", "FGA", "FTA"}.issubset(df.columns):
        denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
        denom = denom.replace(0, np.nan)  # avoid divide-by-zero
        df["TS_PCT"] = df["PTS"] / denom
        df["TS_PCT"] = df["TS_PCT"].fillna(0.0)
        print("Computed TS_PCT.")
    else:
        print("WARNING: Missing PTS/FGA/FTA columns; setting TS_PCT = 0.")
        df["TS_PCT"] = 0.0

    # 3. Compute SEASON_START and IS_ROOKIE
    if "SEASON" not in df.columns:
        raise ValueError("rookie_dataset.csv must contain a 'SEASON' column.")

    df["SEASON_START"] = df["SEASON"].str.slice(0, 4).astype(int)
    df = df.sort_values(["PLAYER_ID", "SEASON_START"])
    first_season = df.groupby("PLAYER_ID")["SEASON_START"].transform("min")
    df["IS_ROOKIE"] = (df["SEASON_START"] == first_season).astype(int)

    # 4. Filter to target season AND rookies only
    season_mask = df["SEASON"] == TARGET_SEASON
    df_season = df[season_mask & (df["IS_ROOKIE"] == 1)].copy()

    if df_season.empty:
        raise ValueError(
            f"No rookies found for SEASON == '{TARGET_SEASON}'. "
            "Check that build_roy_dataset.py included this season and that "
            "players have stats for it."
        )

    print(f"Rows for season {TARGET_SEASON} rookies: {df_season.shape[0]}")

    # 5. Select features (and handle any missing ones)
    available_features = [f for f in FEATURES if f in df_season.columns]
    missing_features = [f for f in FEATURES if f not in df_season.columns]

    if missing_features:
        print("WARNING: These features are missing and will be ignored:", missing_features)

    if not available_features:
        raise ValueError("None of the specified FEATURES are present for this season.")

    X_pred = df_season[available_features].fillna(0)

    # 6. Load trained model
    print(f"Loading trained model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    # 7. Predict probabilities
    print("Predicting ROY probabilities for rookies...")
    probs = model.predict_proba(X_pred)[:, 1]
    df_season["probability"] = probs

    # 8. Build predictions.csv (player_name, probability)
    output = (
        df_season[["PLAYER_NAME", "probability"]]
        .rename(columns={"PLAYER_NAME": "player_name"})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 rookie candidates:")
    print(output.head(10))

    print(f"\nSaving predictions to {OUTPUT_CSV} ...")
    output.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
