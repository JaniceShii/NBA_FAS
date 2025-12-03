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

# Change this to whichever season you want to predict for.
# Right now your dataset goes up to END_YEAR = 2024 in build_roy_dataset.py,
# which corresponds to the season string "2024-25".
# For 2025-26 predictions, you'll need to rebuild the dataset with END_YEAR = 2025.
TARGET_SEASON = "2025-26"

# Must match the features used during training in train_roy.py
FEATURES = [
    "PTS",         # ppg
    "REB",         # rpg
    "AST",         # apg
    "MIN",         # mpg
    "STL",         # spg
    "BLK",         # bpg
    "TOV",         # turnovers
    "GS",          # games started
    "EFG_PCT",     # eFG%
    "PLUS_MINUS",  # plus/minus
    "TS_PCT",      # true shooting percentage (computed below)
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

    # 3. Filter to the target season
    if "SEASON" not in df.columns:
        raise ValueError("rookie_dataset.csv must contain a 'SEASON' column.")

    season_mask = df["SEASON"] == TARGET_SEASON
    df_season = df[season_mask].copy()

    if df_season.empty:
        raise ValueError(
            f"No rows found for SEASON == '{TARGET_SEASON}'. "
            "Make sure build_roy_dataset.py included this season "
            "or update TARGET_SEASON to one that exists (e.g. '2024-25')."
        )

    print(f"Rows for season {TARGET_SEASON}: {df_season.shape[0]}")

    # 4. Select features (and handle any missing ones)
    available_features = [f for f in FEATURES if f in df_season.columns]
    missing_features = [f for f in FEATURES if f not in df_season.columns]

    if missing_features:
        print("WARNING: These features are missing and will be ignored:", missing_features)

    if not available_features:
        raise ValueError("None of the specified FEATURES are present for this season.")

    X_pred = df_season[available_features].fillna(0)

    # 5. Load trained model
    print(f"Loading trained model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    # 6. Predict probabilities
    print("Predicting ROY probabilities...")
    probs = model.predict_proba(X_pred)[:, 1]
    df_season["probability"] = probs

    # 7. Build predictions.csv (player_name, probability)
    output = (
        df_season[["PLAYER_NAME", "probability"]]
        .rename(columns={"PLAYER_NAME": "player_name"})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 candidates:")
    print(output.head(10))

    print(f"\nSaving predictions to {OUTPUT_CSV} ...")
    output.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
