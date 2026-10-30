"""
Project: Predicting Hit Songs & Virality Using Audio Feature Engineering
Script Description: Trains and evaluates baseline and ensemble classification models 
                    (Logistic Regression, Naïve Bayes, Random Forest, XGBoost) 
                    using the 71-feature engineered Spotify/Billboard dataset.

Attribution & Methodological Citations:
- Logistic Regression & Naïve Bayes implementations follow standard scikit-learn API design 
  (Pedregosa et al., 2011; https://jmlr.org/papers/v12/pedregosa11a.html).
- Random Forest architecture follows the ensemble methodology outlined by 
  Breiman, L. (2001). "Random Forests", Machine Learning, 45(1), 5-32.
- XGBoost architecture follows the scalable gradient boosting framework by 
  Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System", 
  Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("Loading master dataset...")
df = pd.read_csv("Master_Spotify_Hit_Prediction_Dataset.csv")

# ==========================================
# 2. PREPARE FEATURES (X) AND TARGET (y)
# ==========================================
# Drop non-numeric identifier and URL metadata columns that should not be used as features
drop_cols = [
    "track_id",
    "track_name",
    "artist_name",
    "artwork_url",
    "album_name",
    "track_url",
    "is_hit",
]
X = df.drop(columns=drop_cols)
y = df["is_hit"]

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
# 4. INITIALIZE MODELS (Partner A & B Lineup)
# ==========================================
models = {
    # Partner A: Linear and Probabilistic Baselines
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Naïve Bayes": GaussianNB(),
    # Partner B: Advanced Tree Ensembles
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
    ),
}

# ==========================================
# 5. TRAIN, EVALUATE, AND COMPARE
# ==========================================
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