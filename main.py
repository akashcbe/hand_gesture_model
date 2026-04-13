import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os
from datetime import datetime

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

click_times = []
freeze_cursor = False
prev_x, prev_y = 0, 0
smooth = 4

# Screenshot setup
screenshot_folder = "screenshots"
os.makedirs(screenshot_folder, exist_ok=True)
screenshot_cooldown = 1.5
last_screenshot_time = 0

print("Controls:")
print("Index finger only     = Move cursor")
print("Thumb + Index pinch   = Click / Double Click")
print("Index + Middle up     = Scroll")
print("All 4 fingers up      = Right Click")
print("Pinky only up         = Screenshot")
print("Q                     = Quit")

def fingers_up(lm):
    tips = [8, 12, 16, 20]
    fingers = []
    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def take_screenshot():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(screenshot_folder, f"screenshot_{timestamp}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"Screenshot saved: {filename}")
    return filename

# Flash effect state
flash_text = ""
flash_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    status = ""

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark
        fingers = fingers_up(lm)

        thumb_tip  = lm[4]
        index_tip  = lm[8]
        middle_tip = lm[12]
        pinky_tip  = lm[20]

        # Thumb up check (for pinky-only gesture)
        thumb_up = lm[4].x > lm[3].x  # for right hand

        # Distance between thumb and index
        dist = math.hypot(
            (thumb_tip.x - index_tip.x) * w,
            (thumb_tip.y - index_tip.y) * h
        )

        # ── SCREENSHOT: only pinky up ──
        if fingers == [0, 0, 0, 1]:
            now = time.time()
            if now - last_screenshot_time > screenshot_cooldown:
                last_screenshot_time = now
                take_screenshot()
                flash_text = "SCREENSHOT!"
                flash_time = now
                status = "SCREENSHOT!"

        # ── MOVE CURSOR: only index up ──
        elif fingers == [1, 0, 0, 0]:
            raw_x = index_tip.x * screen_w
            raw_y = index_tip.y * screen_h
            cur_x = prev_x + (raw_x - prev_x) / smooth
            cur_y = prev_y + (raw_y - prev_y) / smooth
            prev_x, prev_y = cur_x, cur_y
            pyautogui.moveTo(cur_x, cur_y)
            freeze_cursor = False
            status = "MOVE"

        # ── SCROLL: index + middle up ──
        elif fingers == [1, 1, 0, 0]:
            idx_y = index_tip.y
            mid_y = middle_tip.y
            diff = int((idx_y - mid_y) * 500)
            pyautogui.scroll(-diff)
            status = "SCROLL"
            time.sleep(0.05)

        # ── RIGHT CLICK: all fingers up ──
        elif fingers == [1, 1, 1, 1]:
            pyautogui.rightClick()
            status = "RIGHT CLICK"
            time.sleep(0.4)

        # ── PINCH: click / double click ──
        if dist < 30 and fingers != [0, 0, 0, 1]:
            if not freeze_cursor:
                freeze_cursor = True
                now = time.time()
                click_times.append(now)
                click_times = [t for t in click_times if now - t < 0.5]

                if len(click_times) >= 2:
                    pyautogui.doubleClick()
                    click_times = []
                    status = "DOUBLE CLICK"
                else:
                    pyautogui.click()
                    status = "CLICK"
        else:
            freeze_cursor = False

        # Show finger state
        cv2.putText(frame, f"Fingers: {fingers}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Flash screenshot effect (white border for 1 second)
    if time.time() - flash_time < 1.0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 15)
        cv2.putText(frame, "SCREENSHOT!", (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    # Show status
    if status:
        cv2.putText(frame, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()