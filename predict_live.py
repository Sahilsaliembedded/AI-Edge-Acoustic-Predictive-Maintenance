import numpy as np
import wave
import joblib
from extract_features import extract_features

WAV_FILE = "live_recording.wav"
MODEL_FILE = "motor_model.pkl"

print("Loading audio...")

with wave.open(WAV_FILE, "rb") as wav:
    sample_rate = wav.getframerate()
    frames = wav.getnframes()
    audio = wav.readframes(frames)

samples = np.frombuffer(audio, dtype=np.int16)

print("Sample rate:", sample_rate)
print("Samples:", len(samples))

print("Extracting features...")

features = extract_features(
    samples,
    sample_rate
)

print("Features:", features.shape)

print("Loading AI model...")

model = joblib.load(MODEL_FILE)

X = features.reshape(1, -1)

prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print()
print("==============================")
print("       MOTOR CONDITION")
print("==============================")

if prediction == 0:
    print("Prediction: NORMAL")
else:
    print("Prediction: ABNORMAL")

print()
print("Probabilities:")

for cls, prob in zip(model.classes_, probabilities):

    if cls == 0:
        name = "NORMAL"
    else:
        name = "ABNORMAL"

    print(f"{name}: {prob * 100:.2f}%")

print("==============================")
print()
print("Live WAV AI analysis complete.")