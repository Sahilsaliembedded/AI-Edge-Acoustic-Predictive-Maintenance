import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# ============================================================
# LOAD DATA
# ============================================================

X = np.load("X.npy")
y = np.load("y.npy")

print("========================================")
print("FEATURE SELECTION TEST")
print("========================================")

print("Dataset:", X.shape)
print("Normal:", np.sum(y == 0))
print("Abnormal:", np.sum(y == 1))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# TEST DIFFERENT NUMBER OF FEATURES
# ============================================================

feature_counts = [10, 15, 20, 30, 40, 60, 80, 108]

results = []

for k in feature_counts:

    print()
    print("----------------------------------------")
    print("Testing", k, "features")
    print("----------------------------------------")

    model = Pipeline([
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=k
            )
        ),

        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    average = scores.mean() * 100
    std = scores.std() * 100

    print("Fold scores:")

    for i, score in enumerate(scores, 1):
        print(
            f"Fold {i}: {score * 100:.2f}%"
        )

    print(
        "Average:",
        round(average, 2),
        "%"
    )

    print(
        "Std:",
        round(std, 2),
        "%"
    )

    results.append(
        (k, average, std)
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("FEATURE COUNT COMPARISON")
print("========================================")

print(
    f"{'Features':<12}"
    f"{'Average':<15}"
    f"{'Std Dev':<15}"
)

for k, avg, std in results:

    print(
        f"{k:<12}"
        f"{avg:.2f}%{'':<9}"
        f"{std:.2f}%"
    )


# ============================================================
# BEST FEATURE COUNT
# ============================================================

best = max(
    results,
    key=lambda x: x[1]
)

print()
print("========================================")
print("BEST RESULT")
print("========================================")

print("Features:", best[0])
print("Average Accuracy:", round(best[1], 2), "%")
print("Standard Deviation:", round(best[2], 2), "%")