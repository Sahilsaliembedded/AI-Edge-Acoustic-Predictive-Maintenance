import serial
import time
import wave
import numpy as np
import os

PORT = "COM8"
BAUD = 921600

SAMPLE_RATE = 16000
RECORD_SECONDS = 10

TOTAL_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
TOTAL_BYTES = TOTAL_SAMPLES * 2

OUTPUT_FILE = "live_recording.wav"

print("====================================")
print(" ESP32 MOTOR AUDIO RECORDER")
print("====================================")

print("Connecting to ESP32...")

try:
    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=1
    )
except Exception as e:
    print("ERROR:", e)
    exit()

print("Connected:", PORT)

time.sleep(2)

# Remove old serial data
ser.reset_input_buffer()

print("Starting 10-second recording...")
print("Keep the motor running.")

# Tell ESP32 to start
ser.write(b"R")
ser.flush()

raw_audio = bytearray()

start_time = time.time()

while len(raw_audio) < TOTAL_BYTES:

    remaining = TOTAL_BYTES - len(raw_audio)

    data = ser.read(min(4096, remaining))

    if data:
        raw_audio.extend(data)

    elapsed = time.time() - start_time

    progress = (
        len(raw_audio) / TOTAL_BYTES
    ) * 100

    print(
        f"\rReceived: {len(raw_audio):6d} / "
        f"{TOTAL_BYTES} bytes "
        f"({progress:6.2f}%)",
        end=""
    )

    # Longer timeout
    if elapsed > RECORD_SECONDS + 30:
        print("\n\nRecording timed out.")
        print("Received:", len(raw_audio), "bytes")
        ser.close()
        exit()

print("\n\nAudio received successfully.")

ser.close()

# Convert bytes to int16 audio
audio = np.frombuffer(
    raw_audio,
    dtype=np.int16
)

audio = audio[:TOTAL_SAMPLES]

# Save WAV
with wave.open(OUTPUT_FILE, "wb") as wav:

    wav.setnchannels(1)

    wav.setsampwidth(2)

    wav.setframerate(SAMPLE_RATE)

    wav.writeframes(
        audio.tobytes()
    )

minimum = int(audio.min())
maximum = int(audio.max())

rms = float(
    np.sqrt(
        np.mean(
            audio.astype(np.float64) ** 2
        )
    )
)

print("------------------------------------")
print("Saved:", os.path.abspath(OUTPUT_FILE))
print("Samples:", len(audio))
print("Recording time:", len(audio) / SAMPLE_RATE, "seconds")
print("Minimum:", minimum)
print("Maximum:", maximum)
print("RMS:", round(rms, 2))
print("------------------------------------")

if minimum == 0 and maximum == 0:

    print("WARNING: Audio is completely zero.")

elif rms < 10:

    print("WARNING: Audio signal is extremely weak.")

else:

    print("SUCCESS: Non-zero microphone audio received.")

print("ESP32 -> WAV recording successful.")