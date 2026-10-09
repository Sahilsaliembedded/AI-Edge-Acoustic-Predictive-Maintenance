import streamlit as st
import numpy as np
import sounddevice as sd
import joblib
import os
import wave
from datetime import datetime

# ============================================================
# YOUR ORIGINAL FEATURE EXTRACTOR
# ============================================================

from extract_features import extract_features


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Edge Acoustic Predictive Maintenance",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 10
MODEL_FILE = "motor_model.pkl"
MAX_HISTORY = 10


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9aa4b2;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .normal-box {
        background-color: #102a1b;
        border: 2px solid #27ae60;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .abnormal-box {
        background-color: #321414;
        border: 2px solid #e74c3c;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .unknown-box {
        background-color: #1c2430;
        border: 2px solid #3498db;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .prediction {
        font-size: 32px;
        font-weight: bold;
    }

    .analysis-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        margin-top: 25px;
    }

    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .metric-value {
        font-size: 22px;
        font-weight: bold;
    }

    .metric-label {
        color: #8b949e;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "analyses" not in st.session_state:
    st.session_state.analyses = []

if "signal_number" not in st.session_state:
    st.session_state.signal_number = 0


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        return None, f"{MODEL_FILE} was not found."

    try:
        # IMPORTANT:
        # motor_model.pkl was created using joblib.dump()
        model = joblib.load(MODEL_FILE)

        return model, None

    except Exception as error:
        return None, str(error)


model, model_error = load_model()


# ============================================================
# MODEL CHECK
# ============================================================

if model_error:

    st.error(
        f"❌ AI Model Error: {model_error}"
    )

    st.info(
        "Check that motor_model.pkl is inside the motor_ai folder."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="title">
        🎙️ AI-EDGE ACOUSTIC PREDICTIVE MAINTENANCE
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Based Acoustic Machine Condition Monitoring
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

st.success(
    "🟢 AI Model Loaded | 108 Acoustic Features | NumPy FFT Processing"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Settings")

    st.write(f"**Sample Rate:** {SAMPLE_RATE} Hz")

    st.write(f"**Recording:** {RECORD_SECONDS} seconds")

    st.write("**Features:** 108")

    st.write("**Processing:** NumPy FFT")

    st.write("**Model:** Random Forest")

    st.write("**Model File:** motor_model.pkl")

    st.divider()

    st.subheader("🎯 Demo Procedure")

    st.write("1. Start the motor.")

    st.write("2. Keep microphone near motor.")

    st.write("3. Click NEW SOUND ANALYSIS.")

    st.write("4. Wait for recording.")

    st.write("5. Show AI prediction.")

    st.write("6. Change motor condition.")

    st.write("7. Record another signal.")

    st.divider()

    if st.button(
        "🗑️ Clear All Signals",
        use_container_width=True
    ):

        st.session_state.analyses = []
        st.session_state.signal_number = 0

        st.rerun()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_wav(audio, filename):

    audio_int16 = np.int16(
        np.clip(
            audio,
            -1.0,
            1.0
        ) * 32767
    )

    with wave.open(
        filename,
        "wb"
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(
            audio_int16.tobytes()
        )


def calculate_rms(audio):

    return float(
        np.sqrt(
            np.mean(
                audio ** 2
            ) + 1e-12
        )
    )


def calculate_peak(audio):

    return float(
        np.max(
            np.abs(audio)
        )
    )


def calculate_zcr(audio):

    if len(audio) < 2:
        return 0.0

    crossings = np.sum(
        np.diff(
            np.signbit(audio)
        ).astype(np.float64)
    )

    return float(
        crossings /
        max(
            len(audio) - 1,
            1
        )
    )


def calculate_dominant_frequency(audio):

    if len(audio) == 0:
        return 0.0

    fft = np.fft.rfft(audio)

    magnitude = np.abs(fft)

    frequencies = np.fft.rfftfreq(
        len(audio),
        1.0 / SAMPLE_RATE
    )

    index = np.argmax(magnitude)

    return float(
        frequencies[index]
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_audio(audio):

    try:

        # ----------------------------------------------------
        # USE ORIGINAL 108-FEATURE EXTRACTOR
        # ----------------------------------------------------

        features = extract_features(
            audio,
            SAMPLE_RATE
        )

        features = np.asarray(
            features,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if features.shape[0] != 108:

            return (
                "ERROR",
                0.0,
                features,
                "Feature count is not 108."
            )

        X = features.reshape(
            1,
            -1
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(X)[0]

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = 0.0

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(X)[0]

            confidence = float(
                np.max(probabilities) * 100
            )

        # ----------------------------------------------------
        # LABEL CONVERSION
        # ----------------------------------------------------

        prediction_text = str(
            prediction
        ).strip().lower()

        if prediction_text in [
            "0",
            "normal",
            "healthy"
        ]:

            label = "NORMAL"

        elif prediction_text in [
            "1",
            "abnormal",
            "fault",
            "faulty",
            "warning"
        ]:

            label = "ABNORMAL"

        else:

            try:

                numeric_value = int(
                    prediction
                )

                if numeric_value == 0:
                    label = "NORMAL"
                else:
                    label = "ABNORMAL"

            except Exception:

                label = prediction_text.upper()

        return (
            label,
            confidence,
            features,
            None
        )

    except Exception as error:

        return (
            "ERROR",
            0.0,
            np.zeros(108),
            str(error)
        )


# ============================================================
# NEW SOUND ANALYSIS
# ============================================================

st.subheader(
    "🎤 New Sound Signal"
)

st.write(
    "Every recording creates a separate AI analysis card."
)


if st.button(
    "▶ START NEW SOUND ANALYSIS",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # SIGNAL NUMBER
    # --------------------------------------------------------

    st.session_state.signal_number += 1

    number = st.session_state.signal_number

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    status = st.empty()

    status.info(
        f"🎤 Recording Sound Signal #{number}..."
    )

    try:

        audio = sd.rec(
            int(
                RECORD_SECONDS *
                SAMPLE_RATE
            ),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        progress = st.progress(0)

        # Simple visual progress indicator
        for i in range(100):

            progress.progress(i + 1)

        sd.wait()

        progress.empty()

        audio = audio.flatten()

    except Exception as error:

        st.error(
            f"🎤 Microphone Error: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # REMOVE DC OFFSET
    # --------------------------------------------------------

    audio = (
        audio -
        np.mean(audio)
    )

    status.success(
        "✅ Recording complete."
    )

    # --------------------------------------------------------
    # AUDIO METRICS
    # --------------------------------------------------------

    rms = calculate_rms(audio)

    peak = calculate_peak(audio)

    zcr = calculate_zcr(audio)

    dominant_frequency = calculate_dominant_frequency(
        audio
    )

    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Extracting 108 features and running AI..."
    ):

        (
            label,
            confidence,
            features,
            prediction_error
        ) = predict_audio(audio)

    # --------------------------------------------------------
    # SAVE AUDIO
    # --------------------------------------------------------

    timestamp = datetime.now()

    filename = (
        f"sound_signal_"
        f"{number}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
        f".wav"
    )

    try:

        save_wav(
            audio,
            filename
        )

        file_status = (
            f"Saved: {filename}"
        )

    except Exception as error:

        file_status = (
            f"Save error: {error}"
        )

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    result = {

        "number": number,

        "time":
            timestamp.strftime(
                "%H:%M:%S"
            ),

        "audio":
            audio.copy(),

        "filename":
            filename,

        "label":
            label,

        "confidence":
            confidence,

        "rms":
            rms,

        "peak":
            peak,

        "zcr":
            zcr,

        "dominant_frequency":
            dominant_frequency,

        "features":
            features,

        "error":
            prediction_error,

        # IMPORTANT:
        # Save the status for THIS signal.
        "file_status":
            file_status
    }

    st.session_state.analyses.insert(
        0,
        result
    )

    # Keep only latest 10 signals
    st.session_state.analyses = (
        st.session_state.analyses[:MAX_HISTORY]
    )

    st.rerun()


# ============================================================
# ANALYSIS HISTORY
# ============================================================

st.divider()

st.subheader(
    f"📊 Sound Signal History "
    f"({len(st.session_state.analyses)})"
)


if len(
    st.session_state.analyses
) == 0:

    st.info(
        "No sound signal analyzed yet."
    )


# ============================================================
# DISPLAY EACH SIGNAL
# ============================================================

for result in st.session_state.analyses:

    number = result["number"]

    label = result["label"]

    confidence = result["confidence"]

    rms = result["rms"]

    peak = result["peak"]

    zcr = result["zcr"]

    dominant_frequency = (
        result["dominant_frequency"]
    )

    audio = result["audio"]

    filename = result["filename"]

    timestamp = result["time"]

    prediction_error = result["error"]

    file_status = result["file_status"]

    # --------------------------------------------------------
    # CARD
    # --------------------------------------------------------

    st.markdown(
        '<div class="analysis-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        f"## 🔊 Sound Signal #{number}"
    )

    st.caption(
        f"Recorded at {timestamp}"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if label == "NORMAL":

        st.markdown(
            f"""
            <div class="normal-box">

                <div>
                    AI MACHINE CONDITION
                </div>

                <div class="prediction">
                    🟢 NORMAL
                </div>

                <div>
                    AI Confidence:
                    <b>{confidence:.1f}%</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    elif label == "ABNORMAL":

        st.markdown(
            f"""
            <div class="abnormal-box">

                <div>
                    AI MACHINE CONDITION
                </div>

                <div class="prediction">
                    🔴 ABNORMAL
                </div>

                <div>
                    AI Confidence:
                    <b>{confidence:.1f}%</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="unknown-box">

                <div>
                    AI RESULT
                </div>

                <div class="prediction">
                    ⚠️ {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if prediction_error:

        st.error(
            f"Prediction error: {prediction_error}"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {confidence:.1f}%
                </div>

                <div class="metric-label">
                    AI CONFIDENCE
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {rms:.5f}
                </div>

                <div class="metric-label">
                    RMS
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {peak:.5f}
                </div>

                <div class="metric-label">
                    PEAK
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {zcr:.4f}
                </div>

                <div class="metric-label">
                    ZERO CROSSING
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {dominant_frequency:.0f}
                </div>

                <div class="metric-label">
                    DOMINANT Hz
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # WAVEFORM
    # --------------------------------------------------------

    st.write("### 📈 Acoustic Waveform")

    display_audio = audio

    if len(display_audio) > 6000:

        step = max(
            1,
            len(display_audio) // 6000
        )

        display_audio = display_audio[::step]

    st.line_chart(
        display_audio
    )

    # --------------------------------------------------------
    # AUDIO PLAYER
    # --------------------------------------------------------

    st.write("### 🔊 Recorded Sound")

    try:

        with open(
            filename,
            "rb"
        ) as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/wav"
            )

    except Exception:

        st.warning(
            "Recorded audio file is unavailable."
        )

    st.caption(
        f"📁 {file_status}"
    )

    # --------------------------------------------------------
    # FEATURE INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🔬 Show AI Feature Information"
    ):

        st.write(
            "Feature vector size:"
        )

        st.code(
            str(
                len(
                    result["features"]
                )
            )
        )

        st.write(
            "First 20 features:"
        )

        st.write(
            result["features"][:20]
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# SUMMARY
# ============================================================

if len(
    st.session_state.analyses
) > 0:

    st.divider()

    st.subheader(
        "📋 Overall Analysis"
    )

    normal_count = sum(
        1
        for x in st.session_state.analyses
        if x["label"] == "NORMAL"
    )

    abnormal_count = sum(
        1
        for x in st.session_state.analyses
        if x["label"] == "ABNORMAL"
    )

    total = len(
        st.session_state.analyses
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Total Signals",
            total
        )

    with c2:

        st.metric(
            "Normal Signals",
            normal_count
        )

    with c3:

        st.metric(
            "Abnormal Signals",
            abnormal_count
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Edge Acoustic Predictive Maintenance Node | "
    "ESP32 / Acoustic Sensor / 108-Feature AI Analysis"
)