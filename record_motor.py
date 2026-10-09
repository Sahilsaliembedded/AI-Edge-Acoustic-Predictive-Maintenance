import sounddevice as sd
from scipy.io.wavfile import write
import os

SAMPLE_RATE = 16000
DURATION = 10

folder = input("Enter folder (Normal/Abnormal): ").strip()

if folder not in ["Normal", "Abnormal"]:
    print("Use only Normal or Abnormal")
    exit()

os.makedirs(f"dataset/{folder}", exist_ok=True)

existing = [
    f for f in os.listdir(f"dataset/{folder}")
    if f.endswith(".wav")
]

number = len(existing) + 1

filename = f"dataset/{folder}/{folder.lower()}{number}.wav"

input("Start the motor, then press ENTER to record...")

print("Recording...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

write(filename, SAMPLE_RATE, audio)

print()
print("Recording complete.")
print("Saved:", filename)