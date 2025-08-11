import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from gtts import gTTS
import threading
import os
import pickle
import uuid
from playsound import playsound
from tensorflow.keras.models import load_model
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class SignToTextScreen(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.lstm_model = load_model('Training/model.h5')
        self.labels = np.load('Training/actions.npy')
        self.label_vietnamese = {
            "xin_chao": "xin chào",
            "cam_on": "cảm ơn",
            "xin_loi": "xin lỗi",
            "toi_khoe": "tôi khoẻ"
        }

        self.cap = None
        self.sequence = []
        self.frame_count = 0
        self.max_frames = 30
        self.running = False
        self.camera_active = False

        self.result_text = tk.StringVar(value="Nhấn 'Bật Camera' để bắt đầu nhận diện")
        self.camera_label = tk.Label(self)
        self.camera_label.pack(pady=10)

        self.result_label = tk.Label(self, textvariable=self.result_text, font=("Arial", 18), fg="blue")
        self.result_label.pack(pady=10)

        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(pady=10)

        self.toggle_camera_button = tk.Button(self.button_frame, text="Bật Camera", command=self.toggle_camera)
        self.toggle_camera_button.pack(side=tk.LEFT, padx=5)

        self.start_recognition_button = tk.Button(self.button_frame, text="Bắt Đầu Thu Thập", command=self.start_recognition, state=tk.DISABLED)
        self.start_recognition_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_recognition, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION
            )
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

        extra = np.zeros(30)  # THÊM 30 chiều để đủ 1692
        keypoints = np.concatenate([pose, face, lh, rh, extra])

        if keypoints.shape[0] != 1692:
            print("[⚠️] Keypoints không đúng kích thước:", keypoints.shape[0])
        return keypoints

    def toggle_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.result_text.set("Không thể mở camera. Vui lòng kiểm tra.")
                return
            self.camera_active = True
            self.toggle_camera_button.config(text="Tắt Camera")
            self.start_recognition_button.config(state=tk.NORMAL)
            self.result_text.set("Camera đã bật. Nhấn 'Bắt Đầu Thu Thập' để ghi lại cử chỉ.")
            self.update_camera_feed()
        else:
            self.stop_camera_feed()
            self.camera_active = False
            self.toggle_camera_button.config(text="Bật Camera")
            self.start_recognition_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.result_text.set("Camera đã tắt.")
            self.camera_label.config(image='')

    def start_recognition(self):
        if self.camera_active:
            self.sequence = []
            self.frame_count = 0
            self.running = True
            self.result_text.set("Đang thu thập dữ liệu cử chỉ...")
            self.start_recognition_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.NORMAL)

    def reset_recognition(self):
        self.sequence = []
        self.frame_count = 0
        self.result_text.set("Đã reset. Nhấn 'Bắt Đầu Thu Thập' để ghi lại cử chỉ mới.")
        self.reset_button.config(state=tk.DISABLED)
        self.start_recognition_button.config(state=tk.NORMAL)

    def stop_recognition(self):
        if self.running:
            self.running = False
            self.reset_button.config(state=tk.DISABLED)
            self.start_recognition_button.config(state=tk.NORMAL)
            self.result_text.set("Đang xử lý cử chỉ...")
            threading.Thread(target=self.perform_recognition).start()

    def update_camera_feed(self):
        if not self.camera_active or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.camera_label.after(10, self.update_camera_feed)
            return

        image, results = self.mediapipe_detection(frame, self.holistic)
        self.draw_styled_landmarks(image, results)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        if self.running:
            keypoints = self.extract_keypoints(results)
            if keypoints.shape[0] == 1692:
                self.sequence.append(keypoints)
                self.frame_count += 1

            if self.frame_count >= self.max_frames:
                self.stop_recognition()

        self.camera_label.after(33, self.update_camera_feed)

    def stop_camera_feed(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.camera_label.config(image='')

    def perform_recognition(self):
        sequence_array = np.array(self.sequence)
        print("INPUT SHAPE:", sequence_array.shape)
        if sequence_array.shape[0] > 0:
            if sequence_array.shape[0] < self.max_frames:
                padding = np.zeros((self.max_frames - sequence_array.shape[0], sequence_array.shape[1]))
                sequence_array = np.vstack((sequence_array, padding))
            elif sequence_array.shape[0] > self.max_frames:
                sequence_array = sequence_array[:self.max_frames]

            expected_feature_size = 1692
            if sequence_array.shape[1] != expected_feature_size:
                print(f"Kích thước đặc trưng không đúng: {sequence_array.shape[1]} cần {expected_feature_size}")
                text = "Dữ liệu đầu vào không đúng kích thước!"
            else:
                input_data = np.expand_dims(sequence_array, axis=0)  # (1, 30, 1692)
                print("Input data shape for prediction:", input_data.shape)
                try:
                    lstm_pred = self.lstm_model.predict(input_data)
                    lstm_idx = int(np.argmax(lstm_pred))
                    lstm_label = self.labels[lstm_idx] if lstm_idx < len(self.labels) else "Không xác định"
                    text = self.label_vietnamese.get(lstm_label, lstm_label)
                    lstm_confidence = lstm_pred[0][lstm_idx]
                    print(f"LSTM prediction: {lstm_label} ({text}), confidence={lstm_confidence:.3f}")
                    for i, score in enumerate(lstm_pred[0]):
                        print(f"    {self.label_vietnamese.get(self.labels[i], self.labels[i])}: {score:.3f}")
                except Exception as e:
                    text = "Lỗi dự đoán: " + str(e)
                    print(text.encode("utf-8", errors="replace"))
        else:
            text = "Không có dữ liệu cử chỉ để nhận diện."
        # Hiển thị kết quả model
        self.after(100, lambda: self.display_result(text))

    def display_result(self, text):
        self.result_text.set(f"Kết quả: {text}")
        self.start_recognition_button.config(state=tk.NORMAL if self.camera_active else tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL if self.camera_active else tk.DISABLED)
        self.after(800, lambda: self.speak_vietnamese(text))

    def speak_vietnamese(self, text):
        try:
            tts = gTTS(text=text, lang='vi')
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Lỗi đọc tiếng Việt:", e)

    def start(self):
        self.result_text.set("Nhấn 'Bật Camera' để bắt đầu nhận diện")
        self.toggle_camera_button.config(text="Bật Camera", state=tk.NORMAL)
        self.start_recognition_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.camera_label.config(image='')

    def stop(self):
        self.stop_camera_feed()
        self.running = False
        self.camera_active = False
        self.toggle_camera_button.config(text="Bật Camera", state=tk.NORMAL)
        self.start_recognition_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.result_text.set("Nhấn 'Bật Camera' để bắt đầu nhận diện")
        self.camera_label.config(image='')
