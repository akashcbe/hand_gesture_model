"""
util.py — Shared helpers for the Hand Gesture Recognition app.
"""

import math
import cv2


def fingers_up(lm):
    """
    Returns a list of 4 booleans (index, middle, ring, pinky) indicating
    which fingers are extended.
    Tips: 8=index, 12=middle, 16=ring, 20=pinky.
    """
    tips = [8, 12, 16, 20]
    return [1 if lm[tip].y < lm[tip - 2].y else 0 for tip in tips]


def pinch_distance(lm, w, h):
    """Euclidean pixel distance between thumb tip (4) and index tip (8)."""
    thumb, index = lm[4], lm[8]
    return math.hypot(
        (thumb.x - index.x) * w,
        (thumb.y - index.y) * h,
    )


def detect_gesture(lm, fingers, w, h):
    """
    Map landmark + finger state to a (label, bgr_color) tuple.

    Returns:
        (str, tuple): Human-readable gesture label and BGR color for overlay.
    """
    dist = pinch_distance(lm, w, h)

    if fingers == [0, 0, 0, 1]:
        return "📸 SCREENSHOT",          (0, 220, 180)
    elif fingers == [1, 0, 0, 0]:
        return "🖱️ MOVE CURSOR",         (0, 200, 255)
    elif fingers == [1, 1, 0, 0]:
        return "📜 SCROLL",              (255, 200, 0)
    elif fingers == [1, 1, 1, 1]:
        return "🖱️ RIGHT CLICK",         (255, 100, 100)
    elif dist < 30:
        return "🖱️ CLICK / DOUBLE CLICK",(180, 100, 255)
    else:
        return "✋ HAND DETECTED",        (200, 200, 200)


def draw_overlay(frame, gesture, color, fingers):
    """Draw gesture label and finger indicators onto a frame in-place."""
    h, w = frame.shape[:2]

    # Dark banner at top
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, gesture, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Finger state indicators at bottom
    labels = ["IDX", "MID", "RNG", "PNK"]
    for i, (f, label) in enumerate(zip(fingers, labels)):
        clr = (0, 245, 160) if f else (80, 80, 80)
        cv2.putText(frame, label, (w - 220 + i * 52, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
