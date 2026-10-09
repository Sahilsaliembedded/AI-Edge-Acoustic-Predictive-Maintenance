import streamlit as st
import pandas as pd
import serial
import serial.tools.list_ports
import time
from datetime import datetime

# ============================================================
# AI-EDGE ACOUSTIC PREDICTIVE MAINTENANCE
# JUDGE-FRIENDLY LIVE DASHBOARD
# ============================================================

PORT = "COM8"
BAUD_RATE = 115200

MAX_HISTORY = 30

# Stable decision settings
VOTE_WINDOW = 7
MIN_VOTES = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    .status-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .status-value {
        font-size: 30px;
        font-weight: 750;
    }

    .status-detail {
        font-size: 16px;
        margin-top: 8px;
    }

    .current-normal {
        background: #eaf8ef;
        border-color: #8ed1a5;
    }

    .current-abnormal {
        background: #fff0f0;
        border-color: #ee9999;
    }

    .stable-normal {
        background: #eaf8ef;
        border-color: #8ed1a5;
    }

    .stable-abnormal {
        background: #fff0f0;
        border-color: #ee9999;
    }

    .uncertain {
        background: #fff8df;
        border-color: #e5c45a;
    }

    .warming {
        background: #eef5ff;
        border-color: #9ab8e8;
    }

    .pipeline {
        text-align: center;
        font-size: 17px;
        padding: 14px;
        border-radius: 10px;
        background: #f5f7fa;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
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
    "demo": False
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# SERIAL FUNCTIONS
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
            port=port,
            baudrate=BAUD_RATE,
            timeout=0.05
        )

        time.sleep(1)

        ser.reset_input_buffer()

        return ser

    except Exception as e:

        print("Connection error:", e)

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

        # Expected:
        #
        # DATA,RMS,NORMAL%,ABNORMAL%,RESULT
        #
        # Example:
        #
        # DATA,1800,45,55,ABNORMAL

        if not line.startswith("DATA,"):

            return None

        parts = line.split(",")

        if len(parts) < 5:

            return None

        rms = float(parts[1])

        normal = float(parts[2])

        abnormal = float(parts[3])

        result = parts[4].strip().upper()

        if result not in ["NORMAL", "ABNORMAL"]:

            return None

        return {
            "rms": rms,
            "normal": normal,
            "abnormal": abnormal,
            "result": result
        }

    except Exception as e:

        print("Read error:", e)

        return None


# ============================================================
# DEMO DATA
# ============================================================

def demo_data(window):

    sequence = [

        (1000, 92, 8, "NORMAL"),
        (1050, 89, 11, "NORMAL"),
        (1100, 86, 14, "NORMAL"),
        (1200, 82, 18, "NORMAL"),
        (1300, 78, 22, "NORMAL"),

        (1800, 45, 55, "ABNORMAL"),
        (2200, 25, 75, "ABNORMAL"),
        (2400, 15, 85, "ABNORMAL"),
        (2300, 20, 80, "ABNORMAL"),

        (1400, 72, 28, "NORMAL"),
        (1100, 90, 10, "NORMAL")

    ]

    return sequence[
        (window - 1) % len(sequence)
    ]


# ============================================================
# ADD RESULT
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

        "Result":
            data["result"]
    }

    st.session_state.history.append(row)

    st.session_state.history = (
        st.session_state.history[-MAX_HISTORY:]
    )


# ============================================================
# CURRENT PREDICTION
# ============================================================

def get_current_prediction():

    if not st.session_state.history:

        return "WARMING UP", 0.0

    latest = st.session_state.history[-1]

    if latest["Result"] == "NORMAL":

        return "NORMAL", latest["Normal %"]

    return "ABNORMAL", latest["Abnormal %"]


# ============================================================
# STABLE PREDICTION
# ============================================================

def get_stable_prediction():

    history = st.session_state.history

    if len(history) == 0:

        return "WARMING UP", 0, 0, 0

    recent = history[-VOTE_WINDOW:]

    normal_votes = sum(
        1
        for x in recent
        if x["Result"] == "NORMAL"
    )

    abnormal_votes = sum(
        1
        for x in recent
        if x["Result"] == "ABNORMAL"
    )

    count = len(recent)

    if count < 3:

        return (
            "WARMING UP",
            normal_votes,
            abnormal_votes,
            count
        )

    if abnormal_votes >= MIN_VOTES:

        return (
            "ABNORMAL",
            normal_votes,
            abnormal_votes,
            count
        )

    if normal_votes >= MIN_VOTES:

        return (
            "NORMAL",
            normal_votes,
            abnormal_votes,
            count
        )

    return (
        "UNCERTAIN",
        normal_votes,
        abnormal_votes,
        count
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '⚙️ AI-Edge Acoustic Predictive Maintenance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Motor Condition Monitoring using Acoustic Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="pipeline">
        🎙️ Acoustic Signal
        &nbsp; → &nbsp;
        📊 Feature Analysis
        &nbsp; → &nbsp;
        🤖 Random Forest AI
        &nbsp; → &nbsp;
        ⚙️ Motor Condition
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONTROLS
# ============================================================

c1, c2, c3 = st.columns([2, 1, 1])


with c1:

    if st.button(
        "▶ START MONITORING",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.running = True

        if not st.session_state.demo:

            if st.session_state.serial is None:

                st.session_state.serial = (
                    connect_esp32()
                )

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
        "↻ RESET",
        use_container_width=True
    ):

        st.session_state.running = False

        disconnect_esp32()

        st.session_state.window = 0

        st.session_state.history = []

        st.rerun()


# ============================================================
# DEMO MODE
# ============================================================

st.checkbox(
    "🧪 Demo/Test Mode - simulate motor readings",
    key="demo"
)


# ============================================================
# CONNECTION STATUS
# ============================================================

c1, c2 = st.columns(2)


with c1:

    if st.session_state.demo:

        st.info(
            "🧪 Demo mode active"
        )

    elif st.session_state.connected:

        st.success(
            f"🟢 ESP32 CONNECTED — {PORT}"
        )

    else:

        st.warning(
            f"🟠 ESP32 NOT CONNECTED — {PORT}"
        )


with c2:

    if st.session_state.running:

        st.success(
            "🔴 LIVE MONITORING ACTIVE"
        )

    else:

        st.info(
            "Monitoring stopped"
        )


# ============================================================
# CURRENT WINDOW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📡 Live Monitoring'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    latest = st.session_state.history[-1]

    st.metric(
        "Current Analysis Window",
        latest["Window"]
    )

else:

    st.metric(
        "Current Analysis Window",
        0
    )


# ============================================================
# PREDICTIONS
# ============================================================

current_result, current_confidence = (
    get_current_prediction()
)

stable_result, normal_votes, abnormal_votes, vote_count = (
    get_stable_prediction()
)


# ============================================================
# CURRENT AI PREDICTION CARD
# ============================================================

if current_result == "NORMAL":

    current_class = "current-normal"

    current_icon = "🟢"

elif current_result == "ABNORMAL":

    current_class = "current-abnormal"

    current_icon = "🔴"

else:

    current_class = "warming"

    current_icon = "🔵"


st.markdown(
    f"""
    <div class="status-card {current_class}">

        <div class="status-title">
            CURRENT AI PREDICTION
        </div>

        <div class="status-value">
            {current_icon} {current_result}
        </div>

        <div class="status-detail">
            AI Confidence: <b>{current_confidence:.1f}%</b>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STABLE MOTOR CONDITION CARD
# ============================================================

if stable_result == "NORMAL":

    stable_class = "stable-normal"

    stable_icon = "🟢"

elif stable_result == "ABNORMAL":

    stable_class = "stable-abnormal"

    stable_icon = "🔴"

elif stable_result == "UNCERTAIN":

    stable_class = "uncertain"

    stable_icon = "🟡"

else:

    stable_class = "warming"

    stable_icon = "🔵"


st.markdown(
    f"""
    <div class="status-card {stable_class}">

        <div class="status-title">
            STABLE MOTOR CONDITION
        </div>

        <div class="status-value">
            {stable_icon} {stable_result}
        </div>

        <div class="status-detail">

            Recent voting:
            <b>{normal_votes}</b> Normal
            &nbsp; | &nbsp;
            <b>{abnormal_votes}</b> Abnormal

            &nbsp; | &nbsp;

            {vote_count}/{VOTE_WINDOW} windows

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LIVE SIGNAL METRICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Live Signal Metrics'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    latest = st.session_state.history[-1]

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "RMS Level",
            f"{latest['RMS']:.0f}"
        )

    with m2:

        st.metric(
            "Normal Probability",
            f"{latest['Normal %']:.1f}%"
        )

    with m3:

        st.metric(
            "Abnormal Probability",
            f"{latest['Abnormal %']:.1f}%"
        )

    with m4:

        st.metric(
            "Analysis Windows",
            latest["Window"]
        )

else:

    st.info(
        "Start monitoring to receive live acoustic data."
    )


# ============================================================
# RMS GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 Live RMS Signal'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    df = pd.DataFrame(
        st.session_state.history
    )

    rms_chart = df.set_index(
        "Window"
    )[["RMS"]]

    st.line_chart(
        rms_chart,
        height=300
    )

else:

    st.info(
        "RMS history will appear here."
    )


# ============================================================
# AI PROBABILITY GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 AI Probability Trend'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    df = pd.DataFrame(
        st.session_state.history
    )

    probability_chart = df.set_index(
        "Window"
    )[[
        "Normal %",
        "Abnormal %"
    ]]

    st.line_chart(
        probability_chart,
        height=300
    )

else:

    st.info(
        "AI probability history will appear here."
    )


# ============================================================
# VOTING SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🧠 Stability Analysis'
    '</div>',
    unsafe_allow_html=True
)

v1, v2, v3 = st.columns(3)


with v1:

    st.metric(
        "Normal Votes",
        normal_votes
    )


with v2:

    st.metric(
        "Abnormal Votes",
        abnormal_votes
    )


with v3:

    st.metric(
        "Stable Decision",
        stable_result
    )


st.caption(
    f"Stable condition uses the latest "
    f"{VOTE_WINDOW} analysis windows. "
    f"{MIN_VOTES} matching votes are required "
    f"for a stable NORMAL or ABNORMAL decision."
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📋 Recent Prediction History'
    '</div>',
    unsafe_allow_html=True
)

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
        "No prediction history yet."
    )


# ============================================================
# CONTINUOUS MONITORING LOOP
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

        data = {

            "rms":
                values[0],

            "normal":
                values[1],

            "abnormal":
                values[2],

            "result":
                values[3]
        }


    # --------------------------------------------------------
    # REAL ESP32
    # --------------------------------------------------------

    else:

        data = read_esp32()

        if data is None:

            time.sleep(0.2)


    # --------------------------------------------------------
    # STORE DATA
    # --------------------------------------------------------

    if data is not None:

        add_result(data)


    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Edge Acoustic Predictive Maintenance Node | "
    "ESP32 + Microphone + Random Forest | "
    "Real-Time Acoustic Monitoring"
)