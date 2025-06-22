import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Audio Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Timer and state
start_touch_time = 0
volume_control_active = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2))

    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index tip

        # Always show thumb and index tips
        cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (0, 255, 255), cv2.FILLED)

        # Distance between tips
        length = math.hypot(x2 - x1, y2 - y1)

        if length < 20:
            # Fingers are touching
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "🟢 Holding...", (200, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

            if start_touch_time == 0:
                start_touch_time = time.time()

            elif time.time() - start_touch_time >= 2 and not volume_control_active:
                volume_control_active = True

        else:
            # Reset hold timer if fingers are apart
            start_touch_time = 0

        if volume_control_active:
            # Draw green line to show active state
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # Adjust volume
            vol = np.interp(length, [20, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            vol_bar = np.interp(length, [20, 200], [400, 150])
            vol_percent = np.interp(length, [20, 200], [0, 100])

            # Show volume bar
            cv2.rectangle(img, (50, 150), (90, 400), (50, 50, 50), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (90, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_percent)} %', (45, 440), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

            # Status text
            cv2.putText(img, "✅ Volume Control Activated", (140, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Title
    cv2.putText(img, "Gesture Volume Control", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
