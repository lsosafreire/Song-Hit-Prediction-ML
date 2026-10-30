import numpy as np
import pandas as pd

# ==========================================
# STEP 1: LOAD DATASETS
# ==========================================
# Load core Spotify tracks dataset containing audio features and metadata
tracks_df = pd.read_csv("spotify_tracks.csv")

# Load Billboard chart datasets to establish ground-truth "hits"
hot100_df = pd.read_csv("hot100.csv")
billboard200_df = pd.read_csv("billboard200.csv")

print(f"Original Tracks Dataset Shape: {tracks_df.shape}")

# ==========================================
# STEP 2: CREATE CLASSIFICATION TARGET (is_hit)
# ==========================================
# Combine song titles from Billboard Hot 100 and Billboard 200 into a master set of hits
hit_songs = (
    set(hot100_df["Song"].str.lower().dropna())
    | set(billboard200_df["Song"].str.lower().dropna())
)

# Label tracks as 1 if they appear on the charts, 0 otherwise
tracks_df["is_hit"] = (
    tracks_df["track_name"].str.lower().isin(hit_songs).astype(int)
)
print(
    f"Target Variable Created. Total Hits Identified: {tracks_df['is_hit'].sum()}"
)

# ==========================================
# STEP 3:
# ==========================================

df_expanded = pd.get_dummies(
    tracks_df,
    columns=["key", "time_signature", "mode", "language"],
    drop_first=False,
)

# ==========================================
# STEP 4: LOGARITHMIC TRANSFORMATIONS
# ==========================================
# Apply log transformation to track duration to normalize skewed outliers
df_expanded["log_duration"] = np.log1p(df_expanded["duration_ms"])

# ==========================================
# STEP 5: INTERACTION FEATURES
# ==========================================
# Multiply complementary audio features to capture synergistic effects
# (e.g., songs that succeed require both high energy AND high danceability)
df_expanded["energy_danceability"] = (
    df_expanded["energy"] * df_expanded["danceability"]
)
df_expanded["valence_energy"] = df_expanded["valence"] * df_expanded["energy"]
df_expanded["acousticness_instrumentalness"] = (
    df_expanded["acousticness"] * df_expanded["instrumentalness"]
)
df_expanded["loudness_tempo"] = df_expanded["loudness"] * df_expanded["tempo"]

# ==========================================
# STEP 6: POLYNOMIAL FEATURES (Non-Linearity)
# ==========================================
# Create squared and cubed terms for continuous audio metrics
# This allows models (like Logistic Regression) to map curved relationships rather than straight lines.
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

for col in continuous_cols:
  df_expanded[f"{col}_sq"] = df_expanded[col] ** 2
  df_expanded[f"{col}_cubed"] = df_expanded[col] ** 3

# ==========================================
# STEP 7: VERIFY & PRINT FEATURES
# ==========================================
print(f"\nFinal Expanded Dataset Shape: {df_expanded.shape}")
print(f"Total Features Generated: {len(df_expanded.columns)}")

print("\n--- COMPLETE LIST OF GENERATED FEATURES ---")
for i, col in enumerate(df_expanded.columns, 1):
  print(f"{i}. {col}")

# Optional: Save the master dataset for your ML training pipeline
# df_expanded.to_csv("Master_Spotify_Hit_Prediction_Dataset.csv", index=False)


# Assuming your expanded dataframe is named df_expanded
# index=False ensures pandas doesn't save an extra column of row numbers (0, 1, 2...)
df_expanded.to_csv("Master_Spotify_Hit_Prediction_Dataset.csv", index=False)

print("Dataset successfully exported to Master_Spotify_Hit_Prediction_Dataset.csv!")


# Save a sample of your engineered dataset to Excel for reporting/presentations
df_expanded.head(1000000).to_excel(
    "Master_Spotify_Hit_Prediction_Excel.xlsx", index=False, sheet_name="Engineered Features"
)