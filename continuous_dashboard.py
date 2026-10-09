import os
import time
import wave
import joblib
import numpy as np
import sounddevice as sd

# ============================================================
# MOTOR AI - CONTINUOUS STABLE PREDICTION
# ============================================================

MODEL_FILE = "motor_model.pkl"

SAMPLE_RATE = 16000
RECORD_SECONDS = 2

# Number of recent predictions used for majority voting
VOTE_WINDOW = 7

# Minimum confidence before accepting a prediction
CONFIDENCE_THRESHOLD = 0.65

# Number of votes required
MIN_VOTES = 5

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading AI model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Classes:", model.classes_)

# ------------------------------------------------------------
# Prediction history
# ------------------------------------------------------------

prediction_history = []


# ============================================================
# AUDIO RECORDING
# ============================================================

def record_audio():

    print("\nRecording motor sound...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    audio = audio.flatten()

    print("Recording complete.")

    return audio


# ============================================================
# SAVE TEMP WAV
# ============================================================

def save_wav(audio, filename="live_recording.wav"):

    with wave.open(filename, "wb") as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        wf.writeframes(audio.tobytes())

    return filename


# ============================================================
# FEATURE EXTRACTION
# ============================================================
#
# IMPORTANT:
# This function MUST produce the same 108 features
# used during training.
#
# If your existing extract_features.py already contains
# this function, import/use that function instead.
# ============================================================

def extract_features_from_wav(filename):

    # Import your existing feature extraction function
    from extract_features import extract_features

    features = extract_features(filename)

    features = np.asarray(features)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return features


# ============================================================
# SINGLE AI PREDICTION
# ============================================================

def predict_single(filename):

    features = extract_features_from_wav(filename)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    class_index = list(model.classes_).index(prediction)

    confidence = float(probabilities[class_index])

    # Your training labels:
    # 0 = Normal
    # 1 = Abnormal

    if int(prediction) == 0:
        label = "NORMAL"
    else:
        label = "ABNORMAL"

    return label, confidence


# ============================================================
# STABLE DECISION
# ============================================================

def stable_decision(label, confidence):

    global prediction_history

    # --------------------------------------------------------
    # Ignore weak predictions
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        print(
            f"Low confidence: {confidence * 100:.1f}% "
            f"-> UNCERTAIN"
        )

        return "UNCERTAIN", confidence

    # --------------------------------------------------------
    # Add accepted prediction
    # --------------------------------------------------------

    prediction_history.append(label)

    # Keep only latest N predictions
    prediction_history = prediction_history[-VOTE_WINDOW:]

    # --------------------------------------------------------
    # Count votes
    # --------------------------------------------------------

    normal_votes = prediction_history.count("NORMAL")
    abnormal_votes = prediction_history.count("ABNORMAL")

    total_votes = len(prediction_history)

    # --------------------------------------------------------
    # Require minimum number of votes
    # --------------------------------------------------------

    if total_votes < MIN_VOTES:

        return "WARMING UP", confidence

    # --------------------------------------------------------
    # Majority decision
    # --------------------------------------------------------

    if normal_votes > abnormal_votes:

        stable_label = "NORMAL"

        vote_confidence = (
            normal_votes / total_votes
        )

    elif abnormal_votes > normal_votes:

        stable_label = "ABNORMAL"

        vote_confidence = (
            abnormal_votes / total_votes
        )

    else:

        stable_label = "UNCERTAIN"
        vote_confidence = 0.50

    return stable_label, vote_confidence


# ============================================================
# MAIN LOOP
# ============================================================

print("\n==========================================")
print(" AI-EDGE MOTOR PREDICTIVE MAINTENANCE")
print(" Stable Live Prediction")
print("==========================================")

print("\nControls:")
print("CTRL+C = Stop")
print("\nStarting monitoring...\n")


try:

    while True:

        # ----------------------------------------------------
        # Record motor
        # ----------------------------------------------------

        audio = record_audio()

        # ----------------------------------------------------
        # Save recording
        # ----------------------------------------------------

        filename = save_wav(audio)

        print(
            f"Saved: {os.path.abspath(filename)}"
        )

        # ----------------------------------------------------
        # AI prediction
        # ----------------------------------------------------

        label, confidence = predict_single(filename)

        print(
            f"AI prediction: {label} "
            f"({confidence * 100:.1f}%)"
        )

        # ----------------------------------------------------
        # Stable prediction
        # ----------------------------------------------------

        stable_label, stable_confidence = stable_decision(
            label,
            confidence
        )

        print("------------------------------------------")

        print(
            f"STABLE RESULT: {stable_label}"
        )

        print(
            f"STABLE CONFIDENCE: "
            f"{stable_confidence * 100:.1f}%"
        )

        print(
            f"Recent predictions: "
            f"{prediction_history}"
        )

        print("------------------------------------------")

        # ----------------------------------------------------
        # Demo message
        # ----------------------------------------------------

        if stable_label == "NORMAL":

            print("✅ MOTOR CONDITION: NORMAL")

        elif stable_label == "ABNORMAL":

            print("⚠️ MOTOR CONDITION: ABNORMAL")

        elif stable_label == "UNCERTAIN":

            print("🟡 MOTOR CONDITION: UNCERTAIN")

        else:

            print("🔵 AI SYSTEM: WARMING UP")

        time.sleep(0.5)


except KeyboardInterrupt:

    print("\nMonitoring stopped.")