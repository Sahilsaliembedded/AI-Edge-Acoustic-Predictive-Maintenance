import streamlit as st
import pandas as pd
import serial
import serial.tools.list_ports
import time
from datetime import datetime
from collections import deque

# ============================================================
# AI-EDGE ACOUSTIC PREDICTIVE MAINTENANCE
# STABLE LIVE DASHBOARD
# ============================================================

PORT = "COM8"
BAUD_RATE = 115200

MAX_HISTORY = 40

VOTE_WINDOW = 7
MIN_VOTES = 5

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Motor Monitoring",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "running": False,
    "window": 0,
    "history": [],
    "serial": None,
    "connected": False,
    "demo": False,

    "prediction_buffer": [],

    "stable_result": "WARMING UP",

    "stable_confidence": 0.0
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# SERIAL
# ============================================================

def find_esp32():

    ports = serial.tools.list_ports.comports()

    for p in ports:

        if p.device.upper() == PORT.upper():

            return p.device

    return None


def connect_esp32():

    try:

        port = find_esp32()

        if port is None:
            return None

        ser = serial.Serial(
            port,
            BAUD_RATE,
            timeout=0.05
        )

        time.sleep(1)

        ser.reset_input_buffer()

        return ser

    except Exception as e:

        print("Serial error:", e)

        return None


def disconnect_esp32():

    try:

        if st.session_state.serial is not None:

            st.session_state.serial.close()

    except Exception:

        pass

    st.session_state.serial = None
    st.session_state.connected = False


# ============================================================
# STABLE AI VOTING
# ============================================================

def stable_prediction(label, confidence):

    buffer = st.session_state.prediction_buffer

    # Ignore weak predictions
    if confidence < 0.65:

        return "UNCERTAIN", confidence

    buffer.append(label)

    if len(buffer) > VOTE_WINDOW:

        buffer.pop(0)

    # Need enough samples
    if len(buffer) < MIN_VOTES:

        return "WARMING UP", confidence

    normal_votes = buffer.count("NORMAL")
    abnormal_votes = buffer.count("ABNORMAL")

    total = len(buffer)

    if normal_votes > abnormal_votes:

        result = "NORMAL"

        stable_confidence = normal_votes / total

    elif abnormal_votes > normal_votes:

        result = "ABNORMAL"

        stable_confidence = abnormal_votes / total

    else:

        result = "UNCERTAIN"

        stable_confidence = 0.50

    return result, stable_confidence


# ============================================================
# READ ESP32
# ============================================================

def read_esp32():

    ser = st.session_state.serial

    if ser is None:

        return None

    try:

        if ser.in_waiting == 0:

            return None

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:

            return None

        print("ESP32:", line)

        if not line.startswith("DATA,"):

            return None

        parts = line.split(",")

        if len(parts) < 5:

            return None

        rms = float(parts[1])

        normal = float(parts[2])

        abnormal = float(parts[3])

        raw_result = parts[4].strip().upper()

        # ----------------------------------------------------
        # Raw AI result
        # ----------------------------------------------------

        if raw_result not in ["NORMAL", "ABNORMAL"]:

            return None

        confidence = max(
            normal,
            abnormal
        ) / 100.0

        # ----------------------------------------------------
        # Stable AI result
        # ----------------------------------------------------

        stable_result, stable_confidence = stable_prediction(
            raw_result,
            confidence
        )

        st.session_state.stable_result = stable_result
        st.session_state.stable_confidence = stable_confidence

        return {

            "rms": rms,

            "normal": normal,

            "abnormal": abnormal,

            "result": stable_result,

            "raw_result": raw_result,

            "confidence": confidence * 100,

            "stable_confidence":
                stable_confidence * 100
        }

    except Exception as e:

        print("Read error:", e)

        return None


# ============================================================
# DEMO DATA
# ============================================================

def demo_data(window):

    # Controlled demonstration sequence

    if window <= 7:

        return {
            "rms": 2600 + window * 80,
            "normal": 90,
            "abnormal": 10,
            "raw_result": "NORMAL"
        }

    else:

        return {
            "rms": 4300 + window * 70,
            "normal": 15,
            "abnormal": 85,
            "raw_result": "ABNORMAL"
        }


# ============================================================
# ADD DATA
# ============================================================

def add_result(data):

    st.session_state.window += 1

    row = {

        "Time":
            datetime.now().strftime("%H:%M:%S"),

        "Window":
            st.session_state.window,

        "RMS":
            round(data["rms"], 2),

        "Normal %":
            round(data["normal"], 2),

        "Abnormal %":
            round(data["abnormal"], 2),

        "AI":
            data["raw_result"],

        "Stable Result":
            data["result"],

        "Confidence":
            round(data["confidence"], 1),

        "Stable Confidence":
            round(
                data["stable_confidence"],
                1
            )
    }

    st.session_state.history.append(row)

    st.session_state.history = (
        st.session_state.history[-MAX_HISTORY:]
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        padding:10px;
    ">

    <h1>
    ⚙️ AI-Edge Acoustic Predictive Maintenance
    </h1>

    <p style="font-size:18px;">
    Continuous Motor Condition Monitoring
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONTROLS
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    if st.button(
        "▶ START MONITORING",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.running = True

        if not st.session_state.demo:

            if st.session_state.serial is None:

                st.session_state.serial = connect_esp32()

            st.session_state.connected = (
                st.session_state.serial is not None
            )

        st.rerun()


with c2:

    if st.button(
        "■ STOP",
        use_container_width=True
    ):

        st.session_state.running = False

        disconnect_esp32()

        st.rerun()


with c3:

    if st.button(
        "↻ RESET AI",
        use_container_width=True
    ):

        st.session_state.window = 0

        st.session_state.history = []

        st.session_state.prediction_buffer = []

        st.session_state.stable_result = "WARMING UP"

        st.session_state.stable_confidence = 0

        st.rerun()


# ============================================================
# DEMO MODE
# ============================================================

st.checkbox(
    "🧪 Demo/Test Mode",
    key="demo"
)


if st.session_state.demo:

    st.info(
        "🧪 Demo mode active"
    )

elif st.session_state.connected:

    st.success(
        f"🟢 ESP32 connected: {PORT}"
    )

else:

    st.warning(
        f"🟠 ESP32 not connected: {PORT}"
    )


# ============================================================
# MAIN RESULT
# ============================================================

st.markdown(
    "<h2 style='text-align:center;'>"
    "CURRENT MOTOR CONDITION"
    "</h2>",
    unsafe_allow_html=True
)

result = st.session_state.stable_result

confidence = (
    st.session_state.stable_confidence
)


if result == "NORMAL":

    st.success(
        f"## ✅ NORMAL\n\n"
        f"Stable confidence: "
        f"{confidence:.1f}%"
    )

elif result == "ABNORMAL":

    st.error(
        f"## ⚠️ ABNORMAL\n\n"
        f"Stable confidence: "
        f"{confidence:.1f}%"
    )

elif result == "UNCERTAIN":

    st.warning(
        f"## 🟡 UNCERTAIN\n\n"
        f"Confidence: "
        f"{confidence:.1f}%"
    )

else:

    st.info(
        "## 🔵 AI WARMING UP\n\n"
        "Collecting multiple audio windows..."
    )


# ============================================================
# AI VOTING STATUS
# ============================================================

st.subheader("🧠 AI Decision Stability")

buffer = st.session_state.prediction_buffer

normal_votes = buffer.count("NORMAL")
abnormal_votes = buffer.count("ABNORMAL")

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Normal Votes",
        normal_votes
    )

with m2:

    st.metric(
        "Abnormal Votes",
        abnormal_votes
    )

with m3:

    st.metric(
        "Samples Used",
        len(buffer)
    )


st.progress(
    min(
        len(buffer) / VOTE_WINDOW,
        1.0
    )
)


# ============================================================
# RMS
# ============================================================

st.subheader("📈 Live Motor RMS")

if st.session_state.history:

    df = pd.DataFrame(
        st.session_state.history
    )

    st.line_chart(
        df.set_index("Window")[["RMS"]],
        height=300
    )

else:

    st.info(
        "RMS data will appear here."
    )


# ============================================================
# PROBABILITY
# ============================================================

st.subheader("📊 AI Probability")

if st.session_state.history:

    df = pd.DataFrame(
        st.session_state.history
    )

    st.line_chart(
        df.set_index("Window")[
            ["Normal %", "Abnormal %"]
        ],
        height=300
    )

else:

    st.info(
        "AI probabilities will appear here."
    )


# ============================================================
# HISTORY
# ============================================================

st.subheader("📋 Prediction History")

if st.session_state.history:

    df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No predictions yet."
    )


# ============================================================
# CONTINUOUS LOOP
# ============================================================

if st.session_state.running:

    data = None

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    if st.session_state.demo:

        time.sleep(1)

        values = demo_data(
            st.session_state.window + 1
        )

        stable_result, stable_confidence = stable_prediction(
            values["raw_result"],
            max(
                values["normal"],
                values["abnormal"]
            ) / 100
        )

        st.session_state.stable_result = stable_result

        st.session_state.stable_confidence = (
            stable_confidence
        )

        data = {

            "rms":
                values["rms"],

            "normal":
                values["normal"],

            "abnormal":
                values["abnormal"],

            "raw_result":
                values["raw_result"],

            "result":
                stable_result,

            "confidence":
                max(
                    values["normal"],
                    values["abnormal"]
                ),

            "stable_confidence":
                stable_confidence * 100
        }

    # --------------------------------------------------------
    # REAL ESP32
    # --------------------------------------------------------

    else:

        data = read_esp32()

        if data is None:

            time.sleep(0.2)

    # --------------------------------------------------------
    # ADD
    # --------------------------------------------------------

    if data is not None:

        add_result(data)

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Edge Acoustic Predictive Maintenance Node | "
    "ESP32 + Acoustic Sensor + Machine Learning"
)