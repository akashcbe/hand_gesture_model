import streamlit as st
import cv2
import numpy as np
import math
import time
from PIL import Image
import tempfile
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Control",
    page_icon="✋",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #0d0d0d;
    color: #e0e0e0;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp { background-color: #0d0d0d; }

.gesture-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-left: 4px solid #00f5a0;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.85rem;
}
.gesture-card span { color: #00f5a0; font-weight: 700; }

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

# ── MediaPipe lazy loader (avoids module-level AttributeError on Cloud) ───────
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ── Helpers ───────────────────────────────────────────────────
def fingers_up(lm):
    tips = [8, 12, 16, 20]
    return [1 if lm[tip].y < lm[tip - 2].y else 0 for tip in tips]


def detect_gesture(lm, fingers, w, h):
    thumb_tip  = lm[4]
    index_tip  = lm[8]
    middle_tip = lm[12]

    dist = math.hypot(
        (thumb_tip.x - index_tip.x) * w,
        (thumb_tip.y - index_tip.y) * h,
    )

    if fingers == [0, 0, 0, 1]:
        return "📸 SCREENSHOT", (0, 220, 180)
    elif fingers == [1, 0, 0, 0]:
        return "🖱️ MOVE CURSOR", (0, 200, 255)
    elif fingers == [1, 1, 0, 0]:
        return "📜 SCROLL", (255, 200, 0)
    elif fingers == [1, 1, 1, 1]:
        return "🖱️ RIGHT CLICK", (255, 100, 100)
    elif dist < 30:
        return "🖱️ CLICK / DOUBLE CLICK", (180, 100, 255)
    else:
        return "✋ HAND DETECTED", (200, 200, 200)


def process_frame(frame, hands_model):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_model.process(rgb)
    gesture = None
    color = (200, 200, 200)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 245, 160), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=2),
        )

        lm = hand_landmarks.landmark
        fingers = fingers_up(lm)
        gesture, color = detect_gesture(lm, fingers, w, h)

        # Gesture label on frame
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(frame, gesture, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Finger indicators
        labels = ["IDX", "MID", "RNG", "PNK"]
        for i, (f, label) in enumerate(zip(fingers, labels)):
            clr = (0, 245, 160) if f else (80, 80, 80)
            cv2.putText(frame, label, (w - 220 + i * 52, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

    return frame, gesture


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✋ Hand Gesture\n### Control Panel")
    st.markdown("---")

    source = st.radio("Input Source", ["📷 Webcam (Live)", "🎞️ Upload Video", "🖼️ Upload Image"])
    st.markdown("---")

    st.markdown("**Gesture Guide**")
    gestures = [
        ("☝️ Index only",        "Move Cursor"),
        ("👌 Thumb + Index",      "Click / Double Click"),
        ("✌️ Index + Middle",     "Scroll"),
        ("🖐️ All 4 fingers",     "Right Click"),
        ("🤙 Pinky only",         "Screenshot"),
    ]
    for icon_label, action in gestures:
        st.markdown(
            f'<div class="gesture-card"><span>{icon_label}</span><br>{action}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    confidence = st.slider("Detection Confidence", 0.5, 1.0, 0.7, 0.05)
    show_fps = st.checkbox("Show FPS", value=True)


# ── Main UI ───────────────────────────────────────────────────
st.markdown("# ✋ Hand Gesture Recognition")
st.markdown(
    '<div style="color:#666;font-size:0.85rem;margin-bottom:24px;">'
    'Real-time hand gesture detection powered by MediaPipe · '
    'No cursor control in browser mode</div>',
    unsafe_allow_html=True,
)

status_placeholder = st.empty()
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Live Stats")
    gesture_out   = st.empty()
    fps_out       = st.empty()
    detection_out = st.empty()

with col1:
    frame_placeholder = st.empty()

# ── Webcam Mode ───────────────────────────────────────────────
if source == "📷 Webcam (Live)":
    status_placeholder.markdown(
        '<div class="status-box">⚡ WEBCAM LIVE</div>',
        unsafe_allow_html=True,
    )

    img_file = st.camera_input("Point your hand at the camera")

    if img_file:
        pil_img = Image.open(img_file)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5,
        ) as hands_model:
            processed, gesture = process_frame(frame, hands_model)

        result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(result_rgb, channels="RGB", use_container_width=True)

        if gesture:
            gesture_out.markdown(
                f'<div class="status-box" style="font-size:0.9rem">{gesture}</div>',
                unsafe_allow_html=True,
            )
            detection_out.markdown(
                '<span class="info-pill">🟢 Hand Detected</span>',
                unsafe_allow_html=True,
            )
        else:
            gesture_out.markdown(
                '<div class="status-box" style="color:#555;">NO HAND</div>',
                unsafe_allow_html=True,
            )
            detection_out.markdown(
                '<span class="info-pill">🔴 No Detection</span>',
                unsafe_allow_html=True,
            )

# ── Video Upload Mode ─────────────────────────────────────────
elif source == "🎞️ Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        status_placeholder.markdown(
            '<div class="status-box">🎞️ PROCESSING VIDEO</div>',
            unsafe_allow_html=True,
        )

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

        progress = st.progress(0, text="Processing frames…")
        stop_btn  = st.button("⏹ Stop Processing")

        frame_count = 0
        gesture_counts = {}

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5,
        ) as hands_model:
            t0 = time.time()
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                processed, gesture = process_frame(frame, hands_model)

                # FPS overlay
                elapsed = time.time() - t0
                fps_val = frame_count / elapsed if elapsed > 0 else 0
                if show_fps:
                    cv2.putText(processed, f"FPS: {fps_val:.1f}", (10, processed.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(result_rgb, channels="RGB", use_container_width=True)

                if gesture:
                    key = gesture.split(" ", 1)[-1]
                    gesture_counts[key] = gesture_counts.get(key, 0) + 1
                    gesture_out.markdown(
                        f'<div class="status-box" style="font-size:0.9rem">{gesture}</div>',
                        unsafe_allow_html=True,
                    )

                frame_count += 1
                pct = min(int(frame_count / max(total_frames, 1) * 100), 100)
                progress.progress(pct, text=f"Frame {frame_count}/{total_frames}")

                fps_out.markdown(
                    f'<span class="info-pill">🎞️ {fps_val:.1f} FPS</span>',
                    unsafe_allow_html=True,
                )

        cap.release()
        os.unlink(tfile.name)
        progress.progress(100, text="✅ Done!")

        if gesture_counts:
            st.markdown("### Gesture Summary")
            for g, count in sorted(gesture_counts.items(), key=lambda x: -x[1]):
                st.markdown(f'<div class="gesture-card"><span>{g}</span> — {count} frames</div>',
                            unsafe_allow_html=True)

# ── Image Upload Mode ─────────────────────────────────────────
elif source == "🖼️ Upload Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_img:
        status_placeholder.markdown(
            '<div class="status-box">🖼️ IMAGE ANALYSIS</div>',
            unsafe_allow_html=True,
        )

        pil_img = Image.open(uploaded_img).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=confidence,
        ) as hands_model:
            processed, gesture = process_frame(frame, hands_model)

        result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(result_rgb, channels="RGB", use_container_width=True)

        if gesture:
            gesture_out.markdown(
                f'<div class="status-box" style="font-size:0.9rem">{gesture}</div>',
                unsafe_allow_html=True,
            )
            detection_out.markdown(
                '<span class="info-pill">🟢 Hand Detected</span>',
                unsafe_allow_html=True,
            )
            st.success(f"Detected gesture: **{gesture}**")
        else:
            gesture_out.markdown(
                '<div class="status-box" style="color:#555;">NO HAND DETECTED</div>',
                unsafe_allow_html=True,
            )
            st.warning("No hand detected in the image. Try a clearer photo with your hand visible.")

else:
    status_placeholder.markdown(
        '<div class="status-box" style="color:#555;">— SELECT INPUT SOURCE —</div>',
        unsafe_allow_html=True,
    )
    st.info("👈 Choose an input source from the sidebar to get started.")
