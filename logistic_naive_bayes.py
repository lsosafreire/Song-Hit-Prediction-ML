
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ======================================
# 1. Load the dataset
# ======================================

df = pd.read_csv("Master_Spotify_Hit_Prediction_Dataset-2.csv")

print("Dataset loaded!")
print("Total songs:", len(df))

# ======================================
# 2. Prepare the data
# ======================================

# Remove columns that are not used for training
drop_cols = [
    "Unnamed: 0",
    "track_id",
    "track_name",
    "artist_name",
    "artists",
    "album_name",
    "track_genre",
    "artwork_url",
    "track_url",
    "is_hit"
]

# Only remove columns that exist
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols)
y = df["is_hit"]

# Keep numeric features only
X = X.select_dtypes(include="number")

print("Number of features:", X.shape[1])

# ======================================
# 3. Split training and testing data
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ======================================
# 4. Logistic Regression
# ======================================

print("\n--- Logistic Regression ---")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test, logistic_predictions
)

print("Accuracy:", round(logistic_accuracy * 100, 2), "%")
print(classification_report(
    y_test, logistic_predictions, zero_division=0
))

# ======================================
# 5. Naive Bayes
# ======================================

print("\n--- Naive Bayes ---")

bayes_model = GaussianNB()

bayes_model.fit(X_train, y_train)

bayes_predictions = bayes_model.predict(X_test)

bayes_accuracy = accuracy_score(
    y_test, bayes_predictions
)

print("Accuracy:", round(bayes_accuracy * 100, 2), "%")
print(classification_report(
    y_test, bayes_predictions, zero_division=0
))

# ======================================
# 6. Compare the two models
# ======================================

print("\n--- Model Comparison ---")

print("Logistic Regression:", round(logistic_accuracy * 100, 2), "%")
print("Naive Bayes:", round(bayes_accuracy * 100, 2), "%")

if logistic_accuracy > bayes_accuracy:
    print("Logistic Regression has higher accuracy.")
elif bayes_accuracy > logistic_accuracy:
    print("Naive Bayes has higher accuracy.")
else:
    print("Both models have the same accuracy.")

# ======================================
# 7. Collect metrics for visualization
# ======================================

logistic_precision = precision_score(y_test, logistic_predictions, zero_division=0)
logistic_recall = recall_score(y_test, logistic_predictions, zero_division=0)
logistic_f1 = f1_score(y_test, logistic_predictions, zero_division=0)

bayes_precision = precision_score(y_test, bayes_predictions, zero_division=0)
bayes_recall = recall_score(y_test, bayes_predictions, zero_division=0)
bayes_f1 = f1_score(y_test, bayes_predictions, zero_division=0)

model_names = ["Logistic Regression", "Naive Bayes"]
accuracies = [logistic_accuracy, bayes_accuracy]
precisions = [logistic_precision, bayes_precision]
recalls = [logistic_recall, bayes_recall]
f1_scores = [logistic_f1, bayes_f1]

# ======================================
# 8. Bar chart of model metrics
# ======================================

x = range(len(model_names))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar([i - 1.5 * width for i in x], accuracies, width=width, label="Accuracy")
plt.bar([i - 0.5 * width for i in x], precisions, width=width, label="Precision")
plt.bar([i + 0.5 * width for i in x], recalls, width=width, label="Recall")
plt.bar([i + 1.5 * width for i in x], f1_scores, width=width, label="F1-score")

plt.xticks(list(x), model_names)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Logistic Regression vs Naive Bayes")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================
# 9. Confusion Matrix - Logistic Regression
# ======================================

logistic_cm = confusion_matrix(y_test, logistic_predictions)
disp1 = ConfusionMatrixDisplay(confusion_matrix=logistic_cm, display_labels=["Non-Hit", "Hit"])
disp1.plot(values_format="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()

# ======================================
# 10. Confusion Matrix - Naive Bayes
# ======================================

bayes_cm = confusion_matrix(y_test, bayes_predictions)
disp2 = ConfusionMatrixDisplay(confusion_matrix=bayes_cm, display_labels=["Non-Hit", "Hit"])
disp2.plot()
plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.show()