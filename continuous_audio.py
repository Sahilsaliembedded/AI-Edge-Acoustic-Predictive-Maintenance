import serial
import numpy as np
import wave
import time

PORT = "COM8"
BAUD = 921600

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

print("Opening ESP32...")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

print("Connected:", PORT)

# Same startup delay as the working receiver
time.sleep(4)

print("Starting continuous capture...")
print("Keep motor running.")
print("Press Ctrl+C to stop.")
print()

window_number = 0

try:

    while True:

        window_number += 1

        samples = np.empty(
            WINDOW_SAMPLES,
            dtype=np.int16
        )

        received = 0

        print(
            f"\nWindow {window_number} starting..."
        )

        while received < WINDOW_SAMPLES:

            data = ser.read(4096)

            if len(data) == 0:
                continue

            # Make sure we only process complete int16 samples
            usable = len(data) - (len(data) % 2)

            if usable == 0:
                continue

            values = np.frombuffer(
                data[:usable],
                dtype="<i2"
            )

            count = min(
                len(values),
                WINDOW_SAMPLES - received
            )

            samples[
                received:received + count
            ] = values[:count]

            received += count

            percent = (
                received /
                WINDOW_SAMPLES
            ) * 100

            print(
                f"\rRecording: {percent:6.2f}%",
                end="",
                flush=True
            )

        print()

        filename = f"live_window_{window_number}.wav"

        with wave.open(filename, "wb") as wav:

            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(
                samples.tobytes()
            )

        minimum = int(samples.min())
        maximum = int(samples.max())

        rms = np.sqrt(
            np.mean(
                samples.astype(np.float64) ** 2
            )
        )

        print(f"Saved: {filename}")
        print(f"Minimum: {minimum}")
        print(f"Maximum: {maximum}")
        print(f"RMS: {rms:.2f}")

except KeyboardInterrupt:

    print("\nStopping monitoring...")

finally:

    ser.close()
    print("ESP32 connection closed.")