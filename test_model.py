import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Load features
X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Classes:", np.unique(y, return_counts=True))

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nCross-validation results:")

for i, score in enumerate(scores):
    print(
        f"Fold {i+1}: "
        f"{score * 100:.2f}%"
    )

print(
    f"\nMean accuracy: "
    f"{scores.mean() * 100:.2f}%"
)

# Train final model
model.fit(X, y)

# Training evaluation
pred = model.predict(X)

print("\nClassification report:")

print(
    classification_report(
        y,
        pred,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )
)

print("\nConfusion matrix:")

print(
    confusion_matrix(y, pred)
)

# Save model
joblib.dump(
    model,
    "motor_model.pkl"
)

print("\nModel saved as motor_model.pkl")