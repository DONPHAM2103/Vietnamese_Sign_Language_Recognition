import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import time
from PIL import Image, ImageTk
from test import image_processed, convert_label, put_text_pil
import os
import pickle

class SignLanguageScreen(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        # Load model.pkl ở đây
        with open('Training/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.saved_text = ""
        self.current_prediction = ""
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.cap = None
        self.is_running = False
        self.setup_ui()

    def setup_ui(self):
        self.video_frame = ttk.Frame(self)
        self.video_frame.pack(pady=20)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        result_frame = ttk.Frame(self)
        result_frame.pack(pady=20, fill=tk.X)
        self.result_text = tk.Text(result_frame, height=3, width=50, font=('Arial', 14), wrap=tk.WORD)
        self.result_text.pack(pady=5)

        text_control_frame = ttk.Frame(self)
        text_control_frame.pack(pady=10)

        buttons = [ ("Reset", self.reset_text)]
        for text, command in buttons:
            ttk.Button(text_control_frame, text=text, style="Custom.TButton", command=command).pack(side=tk.LEFT, padx=10)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0) 
            self.update_frame()


    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                data = image_processed(frame)
                y_pred = self.model.predict(np.array(data).reshape(-1, 63))
                prediction = convert_label(str(y_pred[0]))

                if prediction != self.current_prediction:
                    self.current_prediction = prediction
                    self.last_prediction = prediction
                    self.last_prediction_time = time.time()

                self.check_auto_save()
                put_text_pil(frame, f" Prediction: {prediction}", (20, 30), (255, 0, 0))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (480, 360))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.after(10, self.update_frame)

    def check_auto_save(self):
        if self.current_prediction == self.last_prediction and time.time() - self.last_prediction_time >= 3:
            self.save_current_text()
            self.last_prediction_time = time.time()

    def save_current_text(self):
        if self.current_prediction:
            if self.current_prediction.lower() == "space":
                self.result_text.insert('end', ' ')
            else:
                self.result_text.insert('end', self.current_prediction)
            self.current_prediction = ""
            self.saved_text = self.result_text.get('1.0', 'end-1c')

    def reset_text(self):
        self.result_text.delete('1.0', 'end')
        self.saved_text = ""
        self.current_prediction = ""
