import os
import numpy as np
import joblib
from scipy.io import wavfile

from extract_features import extract_features

MODEL_FILE = "motor_model.pkl"

model = joblib.load(MODEL_FILE)

folders = [
    ("Normal", 0),
    ("Abnormal", 1)
]

print()
print("================================================")
print("MOTOR AUDIO PREDICTION CHECK")
print("================================================")

correct = 0
total = 0

for folder, actual_label in folders:

    folder_path = os.path.join("dataset", folder)

    print()
    print("---------------", folder, "---------------")

    for filename in sorted(os.listdir(folder_path)):

        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(folder_path, filename)

        sample_rate, audio = wavfile.read(filepath)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float64)

        features = extract_features(audio, sample_rate)

        X = features.reshape(1, -1)

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        # Find probability corresponding to class 0 and 1
        normal_index = list(model.classes_).index(0)
        abnormal_index = list(model.classes_).index(1)

        normal_probability = probabilities[normal_index]
        abnormal_probability = probabilities[abnormal_index]

        if prediction == 0:
            predicted_name = "Normal"
            confidence = normal_probability
        else:
            predicted_name = "Abnormal"
            confidence = abnormal_probability

        actual_name = "Normal" if actual_label == 0 else "Abnormal"

        if prediction == actual_label:
            status = "OK"
            correct += 1
        else:
            status = "WRONG"

        total += 1

        print(
            f"{filename:15} "
            f"Actual={actual_name:8} "
            f"Prediction={predicted_name:8} "
            f"Confidence={confidence*100:5.1f}% "
            f"{status}"
        )

print()
print("================================================")
print("SUMMARY")
print("================================================")

print("Correct:", correct)
print("Total:", total)

if total > 0:
    print("Accuracy:", round(correct / total * 100, 2), "%")