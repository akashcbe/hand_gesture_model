import streamlit as st
import numpy as np
import math
import time
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋", layout="wide")

# ── UI Styling ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #0d0d0d;
    color: #e0e0e0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0d0d0d;
}

.gesture-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-left: 4px solid #00f5a0;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.85rem;
}

.status-box {
    background: #111;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    font-size: 1.1rem;
    letter-spacing: 2px;
    color: #00f5a0;
    margin-bottom: 16px;
}

.info-pill {
    display: inline-block;
    background: #1f1f1f;
    border: 1px solid #444;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #aaa;
    margin: 3px;
}

footer { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Load Libraries ────────────────────────────────────────────
@st.cache_resource
def load_libs():
    import os
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    return cv2, mp_hands, mp_drawing


cv2, mp_hands, mp_drawing = load_libs()
libs_ok = True


# ── Gesture Logic ─────────────────────────────────────────────
def fingers_up(lm):
    tips = [8, 12, 16, 20]
    return [1 if lm[tip].y < lm[tip - 2].y else 0 for tip in tips]


def detect_gesture(lm, fingers, w, h):
    dist = math.hypot((lm[4].x - lm[8].x) * w, (lm[4].y - lm[8].y) * h)

    if fingers == [0, 0, 0, 1]:
        return "SCREENSHOT", (0, 220, 180)
    elif fingers == [1, 0, 0, 0]:
        return "MOVE CURSOR", (0, 200, 255)
    elif fingers == [1, 1, 0, 0]:
        return "SCROLL", (255, 200, 0)
    elif fingers == [1, 1, 1, 1]:
        return "RIGHT CLICK", (255, 100, 100)
    elif dist < 30:
        return "CLICK / DOUBLE CLICK", (180, 100, 255)
    else:
        return "HAND DETECTED", (200, 200, 200)


def process_frame(frame, detector):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    gesture = None

    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        lm = hl.landmark

        fingers = fingers_up(lm)
        gesture, color = detect_gesture(lm, fingers, w, h)

        mp_drawing.draw_landmarks(
            frame,
            hl,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(frame, gesture, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame, gesture


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✋ Hand Gesture Control")

    source = st.radio(
        "Input Source",
        ["📷 Webcam (Live)", "🎞️ Upload Video", "🖼️ Upload Image"]
    )

    st.markdown("### Gestures")
    st.markdown("☝️ Move Cursor")
    st.markdown("👌 Click")
    st.markdown("✌️ Scroll")
    st.markdown("🖐️ Right Click")
    st.markdown("🤙 Screenshot")

    confidence = st.slider("Confidence", 0.5, 1.0, 0.7)


# ── Main UI ───────────────────────────────────────────────────
st.title("✋ Hand Gesture Recognition")

status_ph = st.empty()
frame_ph = st.empty()
gesture_out = st.empty()
detection_out = st.empty()


# ── Webcam ────────────────────────────────────────────────────
if source == "📷 Webcam (Live)":
    status_ph.markdown('<div class="status-box">⚡ WEBCAM LIVE</div>', unsafe_allow_html=True)

    img_file = st.camera_input("Capture hand")

    if img_file:
        frame = cv2.cvtColor(np.array(Image.open(img_file)), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        ) as detector:

            processed, gesture = process_frame(frame, detector)

        frame_ph.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        if gesture:
            gesture_out.markdown(f'<div class="status-box">{gesture}</div>', unsafe_allow_html=True)
            detection_out.markdown('<span class="info-pill">🟢 Detected</span>', unsafe_allow_html=True)
        else:
            gesture_out.markdown('<div class="status-box">NO HAND</div>', unsafe_allow_html=True)


# ── Video ─────────────────────────────────────────────────────
elif source == "🎞️ Upload Video":
    video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if video:
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.write(video.read())

        cap = cv2.VideoCapture(tf.name)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence
        ) as detector:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                processed, gesture = process_frame(frame, detector)

                frame_ph.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        cap.release()
        os.remove(tf.name)


# ── Image ─────────────────────────────────────────────────────
elif source == "🖼️ Upload Image":
    img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if img:
        frame = cv2.cvtColor(np.array(Image.open(img).convert("RGB")), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=confidence) as detector:
            processed, gesture = process_frame(frame, detector)

        frame_ph.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        if gesture:
            st.success(f"Detected: {gesture}")
        else:
            st.warning("No hand detected")


# ── Default ───────────────────────────────────────────────────
else:
    status_ph.markdown('<div class="status-box">SELECT INPUT SOURCE</div>', unsafe_allow_html=True)
