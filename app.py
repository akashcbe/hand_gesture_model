import streamlit as st
import numpy as np
import math
import time
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Space Mono', monospace; background-color: #0d0d0d; color: #e0e0e0; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0d0d; }
.gesture-card { background: #1a1a1a; border: 1px solid #2e2e2e; border-left: 4px solid #00f5a0; border-radius: 6px; padding: 14px 18px; margin-bottom: 10px; font-size: 0.85rem; }
.gesture-card span { color: #00f5a0; font-weight: 700; }
.status-box { background: #111; border: 1px solid #333; border-radius: 8px; padding: 16px; text-align: center; font-size: 1.1rem; letter-spacing: 2px; color: #00f5a0; margin-bottom: 16px; }
.info-pill { display: inline-block; background: #1f1f1f; border: 1px solid #444; border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; color: #aaa; margin: 3px; }
footer { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_libs():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    # Download hand landmarker model if not present
    model_path = "/tmp/hand_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    return cv2, mp, mp_python, mp_vision, model_path


def fingers_up(landmarks, w, h):
    lm = landmarks
    tips = [8, 12, 16, 20]
    return [1 if lm[tip].y < lm[tip - 2].y else 0 for tip in tips]


def detect_gesture(landmarks, fingers, w, h):
    lm = landmarks
    dist = math.hypot((lm[4].x - lm[8].x) * w, (lm[4].y - lm[8].y) * h)
    if fingers == [0, 0, 0, 1]:   return "SCREENSHOT",            (0, 220, 180)
    elif fingers == [1, 0, 0, 0]: return "MOVE CURSOR",           (0, 200, 255)
    elif fingers == [1, 1, 0, 0]: return "SCROLL",                (255, 200, 0)
    elif fingers == [1, 1, 1, 1]: return "RIGHT CLICK",           (255, 100, 100)
    elif dist < 30:                return "CLICK / DOUBLE CLICK",  (180, 100, 255)
    else:                          return "HAND DETECTED",          (200, 200, 200)


def process_frame(cv2, mp_python, mp_vision, model_path, frame):
    h, w, _ = frame.shape
    gesture = None

    # Build detector options
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    with mp_vision.HandLandmarker.create_from_options(opts) as detector:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        fingers = fingers_up(lm, w, h)
        gesture, color = detect_gesture(lm, fingers, w, h)

        # Draw landmarks manually
        for connection in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                           (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                           (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]:
            pt1 = (int(lm[connection[0]].x * w), int(lm[connection[0]].y * h))
            pt2 = (int(lm[connection[1]].x * w), int(lm[connection[1]].y * h))
            cv2.line(frame, pt1, pt2, (100, 100, 255), 2)
        for point in lm:
            cx, cy = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 245, 160), -1)

        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(frame, gesture, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        for i, (f, label) in enumerate(zip(fingers, ["IDX", "MID", "RNG", "PNK"])):
            cv2.putText(frame, label, (w - 220 + i * 52, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 245, 160) if f else (80, 80, 80), 2)

    return frame, gesture


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✋ Hand Gesture\n### Control Panel")
    st.markdown("---")
    source = st.radio("Input Source", ["📷 Webcam (Live)", "🎞️ Upload Video", "🖼️ Upload Image"])
    st.markdown("---")
    st.markdown("**Gesture Guide**")
    for icon, action in [
        ("☝️ Index only",    "Move Cursor"),
        ("👌 Thumb + Index",  "Click / Double Click"),
        ("✌️ Index + Middle", "Scroll"),
        ("🖐️ All 4 fingers", "Right Click"),
        ("🤙 Pinky only",     "Screenshot"),
    ]:
        st.markdown(f'<div class="gesture-card"><span>{icon}</span><br>{action}</div>', unsafe_allow_html=True)

st.markdown("# ✋ Hand Gesture Recognition")
st.markdown('<div style="color:#666;font-size:0.85rem;margin-bottom:24px;">Real-time hand gesture detection · MediaPipe Tasks API</div>', unsafe_allow_html=True)

status_ph = st.empty()
col1, col2 = st.columns([3, 1])
with col2:
    st.markdown("### Live Stats")
    gesture_out   = st.empty()
    detection_out = st.empty()
with col1:
    frame_ph = st.empty()

# Load libs
try:
    with st.spinner("Loading model (first run may take ~10s)..."):
        cv2, mp, mp_python, mp_vision, model_path = load_libs()
    libs_ok = True
except Exception as e:
    st.error(f"❌ Failed to load libraries: {e}")
    libs_ok = False

# ── Webcam ────────────────────────────────────────────────────
if libs_ok and source == "📷 Webcam (Live)":
    status_ph.markdown('<div class="status-box">⚡ WEBCAM LIVE</div>', unsafe_allow_html=True)
    img_file = st.camera_input("Point your hand at the camera")
    if img_file:
        frame = cv2.cvtColor(np.array(Image.open(img_file)), cv2.COLOR_RGB2BGR)
        with st.spinner("Detecting..."):
            processed, gesture = process_frame(cv2, mp_python, mp_vision, model_path, frame)
        frame_ph.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if gesture:
            gesture_out.markdown(f'<div class="status-box" style="font-size:0.9rem">{gesture}</div>', unsafe_allow_html=True)
            detection_out.markdown('<span class="info-pill">🟢 Hand Detected</span>', unsafe_allow_html=True)
        else:
            gesture_out.markdown('<div class="status-box" style="color:#555;">NO HAND</div>', unsafe_allow_html=True)
            detection_out.markdown('<span class="info-pill">🔴 No Detection</span>', unsafe_allow_html=True)

# ── Video ─────────────────────────────────────────────────────
elif libs_ok and source == "🎞️ Upload Video":
    uv = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uv:
        status_ph.markdown('<div class="status-box">🎞️ PROCESSING VIDEO</div>', unsafe_allow_html=True)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(uv.read()); tf.flush()
        cap   = cv2.VideoCapture(tf.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prog  = st.progress(0, text="Processing…")
        stop  = st.button("⏹ Stop")
        fc, gcounts = 0, {}
        t0 = time.time()
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            processed, gesture = process_frame(cv2, mp_python, mp_vision, model_path, frame)
            frame_ph.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            if gesture:
                gcounts[gesture] = gcounts.get(gesture, 0) + 1
                gesture_out.markdown(f'<div class="status-box" style="font-size:0.9rem">{gesture}</div>', unsafe_allow_html=True)
            fc += 1
            prog.progress(min(int(fc / max(total, 1) * 100), 100), text=f"Frame {fc}/{total}")
        cap.release(); os.unlink(tf.name)
        prog.progress(100, text="✅ Done!")
        if gcounts:
            st.markdown("### Gesture Summary")
            for g, c in sorted(gcounts.items(), key=lambda x: -x[1]):
                st.markdown(f'<div class="gesture-card"><span>{g}</span> — {c} frames</div>', unsafe_allow_html=True)

# ── Image ─────────────────────────────────────────────────────
elif libs_ok and source == "🖼️ Upload Image":
    ui = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if ui:
        status_ph.markdown('<div class="status-box">🖼️ IMAGE ANALYSIS</div>', unsafe_allow_html=True)
        frame = cv2.cvtColor(np.array(Image.open(ui).convert("RGB")), cv2.COLOR_RGB2BGR)
        with st.spinner("Detecting..."):
            processed, gesture = process_frame(cv2, mp_python, mp_vision, model_path, frame)
        frame_ph.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if gesture:
            gesture_out.markdown(f'<div class="status-box" style="font-size:0.9rem">{gesture}</div>', unsafe_allow_html=True)
            detection_out.markdown('<span class="info-pill">🟢 Hand Detected</span>', unsafe_allow_html=True)
            st.success(f"Detected: **{gesture}**")
        else:
            gesture_out.markdown('<div class="status-box" style="color:#555;">NO HAND DETECTED</div>', unsafe_allow_html=True)
            st.warning("No hand detected. Try a clearer photo.")

elif libs_ok:
    status_ph.markdown('<div class="status-box" style="color:#555;">— SELECT INPUT SOURCE —</div>', unsafe_allow_html=True)
    st.info("👈 Choose an input source from the sidebar to get started.")
