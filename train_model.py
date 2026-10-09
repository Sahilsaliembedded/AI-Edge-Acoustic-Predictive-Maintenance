import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

X = np.load("X.npy")
y = np.load("y.npy")

print("========================================")
print("AI MOTOR SOUND MODEL TRAINING")
print("========================================")

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

normal_count = np.sum(y == 0)
abnormal_count = np.sum(y == 1)

print("Normal samples:", normal_count)
print("Abnormal samples:", abnormal_count)
print("Total samples:", len(y))


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 3. CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 4. TRAIN
# ============================================================

print()
print("Training Random Forest...")

model.fit(X_train, y_train)


# ============================================================
# 5. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("========================================")
print("MODEL EVALUATION")
print("========================================")

print("Test Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print()

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("                 Predicted")
print("              Normal  Abnormal")
print("Actual Normal    ", cm[0][0], "     ", cm[0][1])
print("Actual Abnormal  ", cm[1][0], "     ", cm[1][1])


# ============================================================
# 6. TRAIN FINAL MODEL USING ALL 40 SAMPLES
# ============================================================

print()
print("========================================")
print("FINAL MODEL TRAINING")
print("========================================")

model.fit(X, y)

print("Final model trained using all", len(X), "samples.")


# ============================================================
# 7. SAVE MODEL
# ============================================================

joblib.dump(model, "motor_model.pkl")

print()
print("Model saved as: motor_model.pkl")

print()
print("========================================")
print("TRAINING COMPLETE")
print("========================================")