import streamlit as st
import numpy as np
import math
import time
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

# ─────────────────────────────────────────────
# UI STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body {
    background-color: #0d0d0d;
    color: #e0e0e0;
    font-family: monospace;
}

.status-box {
    background: #111;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    color: #00f5a0;
    font-size: 18px;
    margin-bottom: 10px;
}

.gesture-box {
    background: #1a1a1a;
    padding: 10px;
    border-left: 4px solid #00f5a0;
    margin-bottom: 8px;
}

.info {
    color: #aaa;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD LIBRARIES SAFELY
# ─────────────────────────────────────────────
@st.cache_resource
def load_libs():
    import os
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

    import cv2
    import mediapipe as mp

    return cv2, mp.solutions.hands, mp.solutions.drawing_utils


cv2, mp_hands, mp_drawing = load_libs()


# ─────────────────────────────────────────────
# GESTURE LOGIC
# ─────────────────────────────────────────────
def fingers_up(lm):
    tips = [8, 12, 16, 20]
    return [1 if lm[i].y < lm[i - 2].y else 0 for i in tips]


def detect_gesture(lm, fingers, w, h):
    dist = math.hypot(
        (lm[4].x - lm[8].x) * w,
        (lm[4].y - lm[8].y) * h
    )

    if fingers == [0, 0, 0, 1]:
        return "SCREENSHOT"
    elif fingers == [1, 0, 0, 0]:
        return "MOVE CURSOR"
    elif fingers == [1, 1, 0, 0]:
        return "SCROLL"
    elif fingers == [1, 1, 1, 1]:
        return "RIGHT CLICK"
    elif dist < 30:
        return "CLICK"
    else:
        return "HAND DETECTED"


def process(frame, detector):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    gesture = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(lm)
        gesture = detect_gesture(lm, fingers, w, h)

        cv2.putText(frame, gesture, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    return frame, gesture


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("✋ Hand Gesture Recognition System")

source = st.sidebar.radio(
    "Choose Input",
    ["📷 Webcam", "🖼️ Image", "🎞️ Video"]
)

confidence = st.sidebar.slider("Confidence", 0.5, 1.0, 0.7)

frame_holder = st.empty()
status_holder = st.empty()


# ─────────────────────────────────────────────
# WEBCAM (SAFE STREAMLIT VERSION)
# ─────────────────────────────────────────────
if source == "📷 Webcam":
    status_holder.markdown('<div class="status-box">WEBCAM MODE</div>', unsafe_allow_html=True)

    img = st.camera_input("Capture hand")

    if img:
        frame = cv2.cvtColor(np.array(Image.open(img)), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence
        ) as detector:

            processed, gesture = process(frame, detector)

        frame_holder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        if gesture:
            st.success(gesture)


# ─────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────
elif source == "🖼️ Image":
    img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img:
        frame = cv2.cvtColor(np.array(Image.open(img).convert("RGB")), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=confidence) as detector:
            processed, gesture = process(frame, detector)

        frame_holder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        if gesture:
            st.success(gesture)
        else:
            st.warning("No hand detected")


# ─────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────
elif source == "🎞️ Video":
    video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(video.read())

        cap = cv2.VideoCapture(temp.name)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence
        ) as detector:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                processed, gesture = process(frame, detector)

                frame_holder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        cap.release()
        os.remove(temp.name)
