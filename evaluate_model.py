import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load trained model
model = joblib.load("motor_model.pkl")

# Load features and labels
X = np.load("X.npy")
y = np.load("y.npy")

# Predict all 10 samples
y_pred = model.predict(X)

# Print results
print("\n==========================")
print("ALL SAMPLE PREDICTIONS")
print("==========================")

correct = 0

for i in range(len(y)):
    actual = "Normal" if y[i] == 0 else "Abnormal"
    predicted = "Normal" if y_pred[i] == 0 else "Abnormal"

    if y[i] == y_pred[i]:
        result = "CORRECT"
        correct += 1
    else:
        result = "WRONG"

    print(
        f"Sample {i+1}: "
        f"Actual={actual}, "
        f"Predicted={predicted} -> {result}"
    )

accuracy = correct / len(y) * 100

print("\n==========================")
print("RESULT")
print("==========================")
print("Correct:", correct)
print("Total:", len(y))
print("Accuracy on these samples:", round(accuracy, 2), "%")

# Confusion matrix
cm = confusion_matrix(y, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"]
)

display.plot()
plt.title("Motor Sound Classification")
plt.show()
