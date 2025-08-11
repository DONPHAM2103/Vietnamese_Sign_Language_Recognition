import cv2
import numpy as np
import os
import time
import mediapipe as mp
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# ========================= CONFIG =============================
DATA_PATH = os.path.join('Assets/Dataset_LSTM')
DEBUG_IMG_PATH = os.path.join('debug_frames')
actions = np.array(['tam_biet'])
no_sequences = 30
sequence_length = 30
image_width = 1280
image_height = 720

# Tạo thư mục
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
os.makedirs(DEBUG_IMG_PATH, exist_ok=True)

# ==================== Mediapipe khởi tạo =======================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) \
        .flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) \
        .flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) \
        .flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) \
        .flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    keypoints = np.concatenate([pose, face, lh, rh])
    if keypoints.shape[0] < 1692:
        keypoints = np.concatenate([keypoints, np.zeros(1692 - keypoints.shape[0])])
    return keypoints

# ===================== Bắt đầu camera ==========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic:

    break_all = False
    for action in actions:
        for sequence in range(no_sequences):
            # Countdown chuẩn bị
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, f'Ready: {action.upper()} in {i}s', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1000)

            print(f"\n[INFO] Start {action.upper()} - Sequence {sequence+1}/{no_sequences}")

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Không thể đọc khung hình từ camera.")
                    continue

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Hiển thị thông tin
                status = f'{action.upper()} - Seq {sequence+1} - Frame {frame_num+1}'
                cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Kiểm tra keypoint
                keypoints = extract_keypoints(results)
                has_hand = results.left_hand_landmarks or results.right_hand_landmarks

                if np.all(keypoints == 0) or not has_hand:
                    print(f"[SKIP] Không có keypoints tay tại frame {frame_num}")
                    cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{action}_{sequence}_{frame_num}.jpg"), image)
                    with open('collect_data_error_log.txt', 'a', encoding='utf-8') as logf:
                        logf.write(f"{action},{sequence},{frame_num},no_hand\n")
                    continue

                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)
                print(f"[OK] Saved: {npy_path}")

                cv2.imshow('OpenCV Feed', image)

                key = cv2.waitKey(10)
                if key & 0xFF == ord('q'):
                    break_all = True
                    break
                elif key & 0xFF == ord('s'):
                    print("[SKIP] Bỏ qua sequence này.")
                    break

            if break_all:
                break
        if break_all:
            break

cap.release()
cv2.destroyAllWindows()
