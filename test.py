import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

label_mapping = {
    'DD': 'Đ'  
}

def convert_label(label):
    return label_mapping.get(label, label)

def image_processed(hand_img):
    # Image processing
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    # Accessing MediaPipe solutions
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data).strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        clean = [float(i.strip()[2:]) for i in data if i not in garbage]
        return clean
    except:
        return np.zeros([1, 63], dtype=int)[0]

def put_text_pil(img, text, position, text_color):
    # Chuyển từ BGR (OpenCV) sang RGB (PIL)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Tạo ảnh PIL
    img_pil = Image.fromarray(img_rgb)
    
    # Tạo đối tượng vẽ
    draw = ImageDraw.Draw(img_pil)
    
    # Load font hỗ trợ tiếng Việt
    try:
        # Thử tìm font Arial
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        try:
            # Thử tìm font Times New Roman
            font = ImageFont.truetype("times.ttf", 32)
        except:
            # Sử dụng font mặc định
            font = ImageFont.load_default()
    
    # Vẽ text
    draw.text(position, text, font=font, fill=text_color)
    
    # Chuyển lại sang BGR
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Copy phần đã vẽ vào ảnh gốc
    img[:] = img_bgr[:] 