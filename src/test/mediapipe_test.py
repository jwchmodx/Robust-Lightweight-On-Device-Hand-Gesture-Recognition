from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

# ===== 경로 설정 =====
VIDEO_DIR = Path("/home/bae/project/Robust-Lightweight-On-Device-Hand-Gesture-Recognition/data/20bn-jester-v1/58")
OUT_PATH = Path("/home/bae/project/Robust-Lightweight-On-Device-Hand-Gesture-Recognition/data/processed/test_58.npy")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===== MediaPipe Hands =====
mp_hands = mp.solutions.hands

sequence = []

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    frame_paths = sorted(VIDEO_DIR.glob("*.jpg"))

    print(f"frames: {len(frame_paths)}")

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Failed to read: {frame_path}")
            sequence.append(np.zeros(63, dtype=np.float32))
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(np.array(landmarks, dtype=np.float32))
        else:
            sequence.append(np.zeros(63, dtype=np.float32))

sequence = np.stack(sequence)

np.save(OUT_PATH, sequence)

print("saved:", OUT_PATH)
print("shape:", sequence.shape)
print("detected frames:", np.count_nonzero(sequence.sum(axis=1)))