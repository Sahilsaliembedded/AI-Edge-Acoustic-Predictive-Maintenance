import numpy as np
import librosa
import joblib
import sys

# Load trained model
model = joblib.load("motor_model.pkl")


def extract_features(file_path):

    # Load audio
    y, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # FFT
    fft = np.fft.rfft(y)
    magnitude = np.abs(fft)

    # Remove DC component
    magnitude = magnitude[1:]

    # First 100 FFT features
    if len(magnitude) >= 100:
        features = magnitude[:100]
    else:
        features = np.pad(
            magnitude,
            (0, 100 - len(magnitude))
        )

    return features


# Audio file given by user
audio_file = sys.argv[1]

# Extract features
features = extract_features(audio_file)

# Reshape for model
features = features.reshape(1, -1)

# Prediction
prediction = model.predict(features)[0]

# Probability
probability = model.predict_proba(features)[0]

print("\n==========================")
print("  AI MOTOR SOUND RESULT")
print("==========================")

if prediction == 0:
    print("Prediction : NORMAL")
else:
    print("Prediction : ABNORMAL")

print("Normal probability   :", round(probability[0] * 100, 2), "%")
print("Abnormal probability :", round(probability[1] * 100, 2), "%")