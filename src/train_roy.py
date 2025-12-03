import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. Load data
rookies = pd.read_csv("rookie_dataset.csv")
labels = pd.read_csv("roy_labels.csv")

# Ensure season string formats match (e.g., "1996-97")
print(rookies["SEASON"].unique()[:10])
print(labels["SEASON"].unique()[:10])

# 2. Merge labels onto rookies (winner rows get 1, others NaN for now)
df = rookies.merge(
    labels[["SEASON", "PLAYER_NAME", "WON_ROY"]],
    on=["SEASON", "PLAYER_NAME"],
    how="left"
)

# 3. Add a numeric season for convenience
df["SEASON_START"] = df["SEASON"].str.slice(0, 4).astype(int)

# 4. For seasons where ROY is known (<= 2023-24), set NaN -> 0 (non-winners)
known_seasons_mask = df["SEASON_START"] <= 2023
df.loc[known_seasons_mask & df["WON_ROY"].isna(), "WON_ROY"] = 0

# 5. Keep only rows with labels for training
train_df = df[known_seasons_mask].copy()
train_df["WON_ROY"] = train_df["WON_ROY"].astype(int)
print(train_df.shape)

