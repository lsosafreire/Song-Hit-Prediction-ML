import numpy as np
import pandas as pd

# ==========================================
# STEP 1: LOAD DATASETS
# ==========================================
# Load the core Spotify tracks dataset containing raw audio features and metadata
tracks_df = pd.read_csv("dataset.csv")

# Load Billboard Hot 100 chart dataset to establish rigorous track-level ground-truth "hits"
hot100_df = pd.read_csv("hot100.csv", low_memory=False)

print(f"Original Tracks Dataset Shape: {tracks_df.shape}")

# ==========================================
# STEP 2: CREATE CLASSIFICATION TARGET (is_hit)
# ==========================================
# Extract song titles from Billboard Hot 100 into a lowercase set of verified hit singles
hit_songs = set(hot100_df["Song"].str.lower().dropna())

# Create the binary target variable (is_hit):
# Label tracks as 1 if their name appears in the Hot 100 hit set, and 0 otherwise
tracks_df["is_hit"] = (
    tracks_df["track_name"].str.lower().isin(hit_songs).astype(int)
)
print(
    f"Target Variable Created. Total Hot 100 Hits Identified: {tracks_df['is_hit'].sum()}"
)

# ==========================================
# STEP 3: CATEGORICAL ENCODING (DUMMY VARIABLES)
# ==========================================
# Convert text/categorical attributes (musical key, time signature, and mode)
# into separate binary indicator columns (1 or 0) using dtype=int.
df_expanded = pd.get_dummies(
    tracks_df,
    columns=["key", "time_signature", "mode"],
    drop_first=False,
    dtype=int,
)

# ==========================================
# STEP 4: LOGARITHMIC TRANSFORMATIONS
# ==========================================
# Apply log transformations to normalize skewed outliers across duration and continuous audio metrics
df_expanded["log_duration"] = np.log1p(df_expanded["duration_ms"])
df_expanded["log_speechiness"] = np.log1p(df_expanded["speechiness"])
df_expanded["log_acousticness"] = np.log1p(df_expanded["acousticness"])
df_expanded["log_instrumentalness"] = np.log1p(df_expanded["instrumentalness"])
df_expanded["log_liveness"] = np.log1p(df_expanded["liveness"])

# ==========================================
# STEP 5: INTERACTION FEATURES
# ==========================================
# Multiply complementary audio features together to capture synergistic musical qualities
df_expanded["energy_danceability"] = (
    df_expanded["energy"] * df_expanded["danceability"]
)
df_expanded["valence_energy"] = df_expanded["valence"] * df_expanded["energy"]
df_expanded["acousticness_instrumentalness"] = (
    df_expanded["acousticness"] * df_expanded["instrumentalness"]
)
df_expanded["loudness_tempo"] = df_expanded["loudness"] * df_expanded["tempo"]

# Additional interactions to robustly expand our feature space
df_expanded["speechiness_energy"] = (
    df_expanded["speechiness"] * df_expanded["energy"]
)
df_expanded["liveness_danceability"] = (
    df_expanded["liveness"] * df_expanded["danceability"]
)
df_expanded["acousticness_energy"] = (
    df_expanded["acousticness"] * df_expanded["energy"]
)
df_expanded["valence_danceability"] = (
    df_expanded["valence"] * df_expanded["danceability"]
)

# ==========================================
# STEP 6: POLYNOMIAL FEATURES (Non-Linearity)
# ==========================================
# Define continuous audio columns to expand
continuous_cols = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
]

# Loop through each continuous feature and create squared (_sq) and cubed (_cubed) terms.
for col in continuous_cols:
  df_expanded[f"{col}_sq"] = df_expanded[col] ** 2
  df_expanded[f"{col}_cubed"] = df_expanded[col] ** 3

# ==========================================
# STEP 7: VERIFY, PRINT & EXPORT DATASET
# ==========================================
print(f"\nFinal Expanded Dataset Shape: {df_expanded.shape}")
print(f"Total Features Generated: {len(df_expanded.columns)}")

print("\n--- COMPLETE LIST OF GENERATED FEATURES ---")
for i, col in enumerate(df_expanded.columns, 1):
  print(f"{i}. {col}")

# Export the master engineered dataset
df_expanded.to_csv("Master_Spotify_Hit_Prediction_Dataset.csv", index=False)
print("Dataset successfully exported to Master_Spotify_Hit_Prediction_Dataset.csv!")

# Export sample to Excel
df_expanded.head(1000000).to_excel(
    "Master_Spotify_Hit_Prediction_Excel.xlsx", index=False, sheet_name="Engineered Features"
)