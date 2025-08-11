import tkinter as tk
from tkinter import ttk, font
import pickle
from tensorflow.keras.models import load_model
from sign_language_screen import SignLanguageScreen
from sign_to_text_screen import SignToTextScreen

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Vietnamese Sign Language")
        self.window.geometry("1200x800")


        style = ttk.Style()
        style.configure("Custom.TButton", padding=10, font=('Arial', 12))

        self.main_container = ttk.Frame(window)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_font = font.Font(family='Arial', size=24, weight='bold')
        title = ttk.Label(self.main_container, text="Vietnamese Sign Language", font=title_font)
        title.pack(pady=20)

        button_frame = ttk.Frame(self.main_container)
        button_frame.pack(pady=10)

        self.sign_button = ttk.Button(
            button_frame, 
            text="Sign Language (a,b,c,1,2,3)", 
            style="Custom.TButton",
            command=self.show_sign_language
        )
        self.sign_button.pack(side=tk.LEFT, padx=10)

        self.sign_to_text_button = ttk.Button(
            button_frame, 
            text="Basic Sign Language Words", 
            style="Custom.TButton",
            command=self.show_sign_to_text
        )
        self.sign_to_text_button.pack(side=tk.LEFT, padx=10)

        # Truyền đúng model cho từng màn hình
        self.sign_screen = SignLanguageScreen(self.main_container)
        self.sign_to_text_screen = SignToTextScreen(self.main_container)

        self.current_screen = None

    def show_sign_language(self):
        if self.current_screen == self.sign_screen:
            self.sign_screen.stop()
            self.sign_screen.pack_forget()
            self.current_screen = None
            self.sign_button.configure(text="Sign Language (a,b,c,1,2,3)")
        else:
            if self.current_screen:
                self.current_screen.stop()
                self.current_screen.pack_forget()
            self.sign_screen.pack(fill=tk.BOTH, expand=True)
            self.sign_screen.start()
            self.current_screen = self.sign_screen
            self.sign_button.configure(text="Stop Recognition")
            self.sign_to_text_button.configure(text=" Basic Sign Language Words")

    def show_sign_to_text(self):
        if self.current_screen == self.sign_to_text_screen:
            self.sign_to_text_screen.stop()
            self.sign_to_text_screen.pack_forget()
            self.current_screen = None
            self.sign_to_text_button.configure(text="Basic Sign Language Words")
        else:
            if self.current_screen:
                self.current_screen.stop()
                self.current_screen.pack_forget()
            self.sign_to_text_screen.pack(fill=tk.BOTH, expand=True)
            self.sign_to_text_screen.start()
            self.current_screen = self.sign_to_text_screen
            self.sign_to_text_button.configure(text="Back")
            self.sign_button.configure(text="Sign Language (a,b,c,1,2,3)")
