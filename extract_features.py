import numpy as np


def extract_features(audio, sample_rate=16000):
    """
    Extract 108 acoustic features from an audio signal.

    Returns:
        numpy array with exactly 108 features.
    """

    audio = np.asarray(audio, dtype=np.float64)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Avoid empty input
    if len(audio) == 0:
        return np.zeros(108, dtype=np.float64)

    features = []

    # --------------------------------------------------
    # 1. Time-domain statistical features
    # --------------------------------------------------

    features.append(np.mean(audio))
    features.append(np.std(audio))
    features.append(np.var(audio))
    features.append(np.min(audio))
    features.append(np.max(audio))
    features.append(np.ptp(audio))
    features.append(np.sqrt(np.mean(audio ** 2)))
    features.append(np.mean(np.abs(audio)))
    features.append(np.median(audio))
    features.append(np.percentile(audio, 25))
    features.append(np.percentile(audio, 75))
    features.append(np.percentile(audio, 90))
    features.append(np.percentile(audio, 95))
    features.append(np.percentile(audio, 99))

    # Zero crossing rate
    zero_crossings = np.sum(
        np.diff(np.signbit(audio)).astype(np.float64)
    )

    zcr = zero_crossings / max(len(audio) - 1, 1)

    features.append(zcr)

    # --------------------------------------------------
    # 2. FFT features
    # --------------------------------------------------

    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)

    frequencies = np.fft.rfftfreq(
        len(audio),
        d=1.0 / sample_rate
    )

    magnitude_sum = np.sum(magnitude) + 1e-12

    # Spectral centroid
    spectral_centroid = np.sum(
        frequencies * magnitude
    ) / magnitude_sum

    features.append(spectral_centroid)

    # Spectral spread
    spectral_spread = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2)
            * magnitude
        ) / magnitude_sum
    )

    features.append(spectral_spread)

    # Spectral bandwidth
    spectral_bandwidth = np.sum(
        np.abs(frequencies - spectral_centroid)
        * magnitude
    ) / magnitude_sum

    features.append(spectral_bandwidth)

    # Spectral rolloff
    cumulative = np.cumsum(magnitude)
    rolloff_point = 0.85 * cumulative[-1]

    rolloff_index = np.searchsorted(
        cumulative,
        rolloff_point
    )

    rolloff_index = min(
        rolloff_index,
        len(frequencies) - 1
    )

    features.append(frequencies[rolloff_index])

    # Spectral flatness
    geometric_mean = np.exp(
        np.mean(np.log(magnitude + 1e-12))
    )

    arithmetic_mean = np.mean(magnitude) + 1e-12

    spectral_flatness = geometric_mean / arithmetic_mean

    features.append(spectral_flatness)

    # --------------------------------------------------
    # 3. FFT energy bands
    # --------------------------------------------------

    bands = [
        (0, 100),
        (100, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 6000),
        (6000, 8000),
        (8000, 10000),
        (10000, 12000),
        (12000, 14000),
        (14000, 16000)
    ]

    for low, high in bands:

        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )

        if np.any(mask):
            band_energy = np.mean(
                magnitude[mask] ** 2
            )
        else:
            band_energy = 0

        features.append(band_energy)

    # --------------------------------------------------
    # 4. Frame-based features
    # --------------------------------------------------

    frame_size = 1024
    hop_size = 512

    frames = []

    for start in range(
        0,
        len(audio) - frame_size + 1,
        hop_size
    ):
        frame = audio[
            start:start + frame_size
        ]

        frames.append(frame)

    if len(frames) == 0:
        frames = [audio]

    frame_features = []

    for frame in frames:

        rms = np.sqrt(
            np.mean(frame ** 2)
        )

        peak = np.max(
            np.abs(frame)
        )

        mean_abs = np.mean(
            np.abs(frame)
        )

        std = np.std(frame)

        frame_features.append(
            [rms, peak, mean_abs, std]
        )

    frame_features = np.asarray(
        frame_features
    )

    # Statistics of frame RMS
    rms_values = frame_features[:, 0]

    features.append(np.mean(rms_values))
    features.append(np.std(rms_values))
    features.append(np.min(rms_values))
    features.append(np.max(rms_values))
    features.append(np.median(rms_values))
    features.append(np.percentile(rms_values, 25))
    features.append(np.percentile(rms_values, 75))
    features.append(np.percentile(rms_values, 90))
    features.append(np.percentile(rms_values, 95))
    features.append(np.percentile(rms_values, 99))

    # Statistics of frame peak
    peak_values = frame_features[:, 1]

    features.append(np.mean(peak_values))
    features.append(np.std(peak_values))
    features.append(np.min(peak_values))
    features.append(np.max(peak_values))
    features.append(np.median(peak_values))
    features.append(np.percentile(peak_values, 25))
    features.append(np.percentile(peak_values, 75))
    features.append(np.percentile(peak_values, 90))
    features.append(np.percentile(peak_values, 95))
    features.append(np.percentile(peak_values, 99))

    # Statistics of frame mean absolute value
    mean_abs_values = frame_features[:, 2]

    features.append(np.mean(mean_abs_values))
    features.append(np.std(mean_abs_values))
    features.append(np.min(mean_abs_values))
    features.append(np.max(mean_abs_values))
    features.append(np.median(mean_abs_values))
    features.append(np.percentile(mean_abs_values, 25))
    features.append(np.percentile(mean_abs_values, 75))
    features.append(np.percentile(mean_abs_values, 90))
    features.append(np.percentile(mean_abs_values, 95))
    features.append(np.percentile(mean_abs_values, 99))

    # Statistics of frame standard deviation
    std_values = frame_features[:, 3]

    features.append(np.mean(std_values))
    features.append(np.std(std_values))
    features.append(np.min(std_values))
    features.append(np.max(std_values))
    features.append(np.median(std_values))
    features.append(np.percentile(std_values, 25))
    features.append(np.percentile(std_values, 75))
    features.append(np.percentile(std_values, 90))
    features.append(np.percentile(std_values, 95))
    features.append(np.percentile(std_values, 99))

    # --------------------------------------------------
    # 5. Spectral band statistics
    # --------------------------------------------------

    # Divide FFT into 20 frequency bands
    num_bands = 20

    band_edges = np.linspace(
        0,
        sample_rate / 2,
        num_bands + 1
    )

    for i in range(num_bands):

        low = band_edges[i]
        high = band_edges[i + 1]

        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )

        if np.any(mask):
            energy = np.mean(
                magnitude[mask] ** 2
            )
        else:
            energy = 0

        features.append(energy)

    # --------------------------------------------------
    # Force exactly 108 features
    # --------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float64
    )

    if len(features) < 108:

        features = np.pad(
            features,
            (0, 108 - len(features))
        )

    elif len(features) > 108:

        features = features[:108]

    return features

# ============================================================
# DATASET PROCESSING
# ============================================================

import os
import numpy as np
from scipy.io import wavfile

NORMAL_DIR = "dataset/Normal"
ABNORMAL_DIR = "dataset/Abnormal"

X = []
y = []

# -------------------------
# Process Normal recordings
# -------------------------

for filename in sorted(os.listdir(NORMAL_DIR)):

    if not filename.lower().endswith(".wav"):
        continue

    filepath = os.path.join(NORMAL_DIR, filename)

    print("Processing Normal:", filename)

    sample_rate, audio = wavfile.read(filepath)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)

    feature = extract_features(
        audio,
        sample_rate
    )

    X.append(feature)
    y.append(0)


# -------------------------
# Process Abnormal recordings
# -------------------------

for filename in sorted(os.listdir(ABNORMAL_DIR)):

    if not filename.lower().endswith(".wav"):
        continue

    filepath = os.path.join(ABNORMAL_DIR, filename)

    print("Processing Abnormal:", filename)

    sample_rate, audio = wavfile.read(filepath)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)

    feature = extract_features(
        audio,
        sample_rate
    )

    X.append(feature)
    y.append(1)


# -------------------------
# Convert to NumPy arrays
# -------------------------

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.int64)

print()
print("================================")
print("FEATURE EXTRACTION COMPLETE")
print("================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Unique labels:",
    np.unique(y, return_counts=True)
)

print(
    "Different feature rows:",
    len(np.unique(X, axis=0))
)

# Save
np.save("X.npy", X)
np.save("y.npy", y)

print()
print("Saved:")
print("X.npy")
print("y.npy")