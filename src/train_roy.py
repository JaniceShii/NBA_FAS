"""
train_roy.py

Train a logistic regression model to predict Rookie of the Year (ROY)
using historical rookie data in data/rookie_dataset.csv.

Expected files:
- data/rookie_dataset.csv : created by build_roy_dataset.py
- data/roy_winners.csv     : manual winners with columns:
      SEASON, PLAYER_NAME, WON_ROY
  where WON_ROY = 1 for the actual winner, 0 or missing otherwise.

Output:
- data/roy_model.pkl      : trained sklearn Pipeline (StandardScaler + LogisticRegression)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ---------------------- CONFIG ---------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

ROOKIE_CSV = DATA_DIR / "rookie_dataset.csv"
WINNERS_CSV = DATA_DIR / "roy_winners.csv"
MODEL_PATH = DATA_DIR / "roy_model.pkl"

# Features you specified
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
    # --- 1. Load rookie dataset ---
    print(f"Loading rookie data from {ROOKIE_CSV} ...")
    df = pd.read_csv(ROOKIE_CSV)

    print("Initial shape:", df.shape)

    # --- 2. Compute TS_PCT (True Shooting Percentage) ---
    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    if {"PTS", "FGA", "FTA"}.issubset(df.columns):
        denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
        # avoid divide-by-zero
        denom = denom.replace(0, np.nan)
        df["TS_PCT"] = df["PTS"] / denom
        df["TS_PCT"] = df["TS_PCT"].fillna(0.0)
        print("Computed TS_PCT.")
    else:
        # If for some reason cols are missing, just fill TS_PCT with NaN/0
        print("WARNING: Missing PTS/FGA/FTA columns; TS_PCT will be set to 0.")
        df["TS_PCT"] = 0.0

    # --- 3. Load ROY winners ---
    print(f"Loading ROY Winners from {WINNERS_CSV} ...")
    winners = pd.read_csv(WINNERS_CSV)

    # Make sure WON_ROY is integer (1 for winner, 0 otherwise)
    if "WON_ROY" not in winners.columns:
        raise ValueError("roy_winners.csv must have a 'WON_ROY' column.")

    # Merge winners onto the rookie dataframe
    df = df.merge(
        winners[["SEASON", "PLAYER_NAME", "WON_ROY"]],
        on=["SEASON", "PLAYER_NAME"],
        how="left",
    )

    # Winners have 1, non-winners NaN -> set to 0
    df["WON_ROY"] = df["WON_ROY"].fillna(0).astype(int)

    # --- 4. Keep only rows with at least one non-null feature ---
    available_features = [f for f in FEATURES if f in df.columns]
    if not available_features:
        raise ValueError("None of the specified FEATURES are present in the dataset.")

    print("Using features:", available_features)

    # Drop rows where all features are NaN
    df_features = df[available_features]
    mask_non_all_na = ~df_features.isna().all(axis=1)
    df = df[mask_non_all_na].copy()
    df_features = df_features[mask_non_all_na].copy()

    # Fill remaining NaNs with 0 for modeling
    df_features = df_features.fillna(0)

    # --- 5. Define X, y ---
    X = df_features
    y = df["WON_ROY"]

    print("Final labeled dataset shape:", X.shape)

    # Sanity check: ROY class balance
    print("Class balance (WON_ROY counts):")
    print(y.value_counts())

    # --- 6. Split into train / validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # keep class balance similar in train/val
    )

    # --- 7. Build pipeline: StandardScaler + LogisticRegression ---
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # because only 1 ROY per season
        )),
    ])

    # --- 8. Train model ---
    print("\nTraining logistic regression model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- 9. Evaluate on validation set ---
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_val_pred)
    try:
        auc = roc_auc_score(y_val, y_val_proba)
    except ValueError:
        auc = float("nan")  # if only one class present in y_val

    print(f"\nValidation Accuracy: {acc:.3f}")
    print(f"Validation ROC-AUC: {auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_val_pred, digits=3))

    # --- 10. Inspect feature coefficients (importance) ---
    logreg = pipeline.named_steps["logreg"]
    coefs = logreg.coef_[0]

    coef_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": coefs,
    }).sort_values("coefficient", ascending=False)

    print("\nFeature coefficients (higher → more associated with winning ROY):")
    print(coef_df.to_string(index=False))

    # --- 11. Save model to disk ---
    import joblib
    print(f"\nSaving trained model to {MODEL_PATH} ...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
