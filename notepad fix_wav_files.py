import os
import numpy as np
from scipy.io import wavfile

SRC = "dataset"
DST = "dataset_clean"

TARGET_SR = 16000
TARGET_SAMPLES = 16000 * 10   # exactly 10 seconds

print("========================================")
print("WAV FILE CLEANUP")
print("========================================")

for label in ["Normal", "Abnormal"]:

    src_dir = os.path.join(SRC, label)
    dst_dir = os.path.join(DST, label)

    os.makedirs(dst_dir, exist_ok=True)

    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(".wav")
    ]

    print()
    print(label, "files:", len(files))

    for filename in sorted(files):

        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)

        try:
            sample_rate, audio = wavfile.read(src)

            # Convert stereo -> mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            audio = np.asarray(audio, dtype=np.float64)

            # Check sample rate
            if sample_rate != TARGET_SR:
                print(
                    "SKIPPED:",
                    filename,
                    "| Sample rate:",
                    sample_rate
                )
                continue

            original_samples = len(audio)

            # Make every recording exactly 160000 samples
            if len(audio) > TARGET_SAMPLES:
                audio = audio[:TARGET_SAMPLES]

            elif len(audio) < TARGET_SAMPLES:
                audio = np.pad(
                    audio,
                    (0, TARGET_SAMPLES - len(audio))
                )

            # Convert safely to int16
            audio = np.clip(
                audio,
                -32768,
                32767
            ).astype(np.int16)

            # Write a completely new WAV file
            wavfile.write(
                dst,
                TARGET_SR,
                audio
            )

            print(
                f"Fixed: {label}/{filename} | "
                f"{original_samples} -> {len(audio)} samples"
            )

        except Exception as e:
            print("ERROR:", filename)
            print(e)

print()
print("========================================")
print("CLEANUP COMPLETE")
print("========================================")
print()
print("Clean files saved in:")
print("dataset_clean/Normal")
print("dataset_clean/Abnormal")