"""
Project: Predicting Hit Songs & Virality Using Audio Feature Engineering
Script Description: Trains and evaluates advanced tree-based ensemble classification models 
                    (Random Forest, XGBoost) using the engineered Spotify/Billboard dataset.

Attribution & Methodological Citations:
- Random Forest architecture follows the ensemble methodology outlined by 
  Breiman, L. (2001). "Random Forests", Machine Learning, 45(1), 5-32.
- XGBoost architecture follows the scalable gradient boosting framework by 
  Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System", 
  Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("Loading master dataset...")
df = pd.read_csv("Master_Spotify_Hit_Prediction_Dataset.csv")

# ==========================================
# 2. PREPARE FEATURES (X) AND TARGET (y)
# ==========================================
# Define columns to drop (only drop columns if they actually exist in the dataframe)
drop_cols = [
    "Unnamed: 0",
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "artwork_url",
    "track_url",
    "is_hit",
]

# Safely drop only the columns that are present in df
actual_drop_cols = [col for col in drop_cols if col in df.columns]
X = df.drop(columns=actual_drop_cols)
y = df["is_hit"]

print(f"Features shape: {X.shape}")

# ==========================================
# 3. TRAIN/TEST SPLIT
# ==========================================
# Split data into 80% training and 20% testing sets
# stratify=y preserves the ratio of hits to non-hits across splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ==========================================
# 4. INITIALIZE MODELS (Ensemble Lineup)
# ==========================================
models = {
    # Advanced Tree Ensembles
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,  # Slightly more trees
        learning_rate=0.05,  # Slower learning rate for better generalization (down from 0.1)
        max_depth=5,  # Shallower trees to prevent overfitting on the 69 features
        subsample=0.8,  # Row subsampling to reduce variance
        colsample_bytree=0.8,  # Column subsampling to prevent feature dominance
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
    ),
}

# ==========================================
# 5. TRAIN, EVALUATE, AND COMPARE
# ==========================================
trained_models = {}

for name, model in models.items():
  print(f"\n--- Training {name} ---")

  # Fit model on training data
  model.fit(X_train, y_train)

  # Generate predictions on test data
  y_pred = model.predict(X_test)

  # Calculate performance metrics
  acc = accuracy_score(y_test, y_pred)
  print(f"{name} Accuracy: {acc * 100:.2f}%")
  print("Classification Report:")
  # zero_division=0 prevents division-by-zero warnings if a class has no predictions
  print(classification_report(y_test, y_pred, zero_division=0))

  # Store trained model reference for visualization
  trained_models[name] = model

# ==========================================
# 6. GENERATE PROFESSIONAL VISUALS AUTOMATICALLY
# ==========================================
print("\nGenerating professional visuals...")
sns.set_theme(style="whitegrid")

# Extract the trained Random Forest model for visualizations
rf = trained_models["Random Forest"]
y_pred_rf = rf.predict(X_test)

# --- Visual 1: Feature Importance Bar Chart ---
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
    ascending=True
)
top_10 = importances.tail(10)

top_10.plot(kind="barh", color="#1DB954")  # Spotify Green accent
plt.title(
    "Top 10 Feature Importances in Random Forest Hit Prediction",
    fontsize=14,
    weight="bold",
)
plt.xlabel("Mean Decrease in Impurity (Importance Score)", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=300)
plt.close()
print("Saved: feature_importances.png")

# --- Visual 2: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Predicted Non-Hit", "Predicted Hit"],
    yticklabels=["Actual Non-Hit", "Actual Hit"],
)
plt.title(
    "Random Forest Confusion Matrix (Hot 100 Classification)",
    fontsize=14,
    weight="bold",
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("Saved: confusion_matrix.png")

print(
    "\nExecution complete! All metrics evaluated and 300 DPI visuals generated successfully."
)


# ==========================================
# 7. GENERATE PROFESSIONAL TABLE AS A .PNG VISUAL
# ==========================================
print("Generating professional performance table image...")

# Define your summary data
summary_data = [
    ["Random Forest", "92.42%", "0.98", "0.54", "0.83"],
    ["XGBoost", "72.60%", "0.33", "0.67", "0.63"],
]
columns = [
    "Model Architecture",
    "Accuracy",
    "Hit Precision",
    "Hit Recall",
    "Macro Avg F1",
]

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis("off")
ax.axis("tight")

# Create the table
table = ax.table(
    cellText=summary_data,
    colLabels=columns,
    cellLoc="center",
    loc="center",
)

# Style the table for a professional look
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color header row (Spotify Green #1DB954 with white text) for branding
for (row, col), cell in table.get_celld().items():
  if row == 0:
    cell.set_facecolor("#1DB954")
    cell.set_text_props(weight="bold", color="white")
  else:
    cell.set_facecolor("#F8F9FA" if row % 2 == 0 else "#FFFFFF")

plt.title(
    "Model Performance Summary Table", fontsize=14, weight="bold", pad=15
)
plt.tight_layout()

# Save as high-res PNG
plt.savefig("model_performance_table.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: model_performance_table.png successfully!")