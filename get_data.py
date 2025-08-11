import cv2
import mediapipe as mp
import os
import numpy as np 

def image_processed(file_path):
    # Đọc hình ảnh tĩnh
    hand_img = cv2.imread(file_path)

    # Kiểm tra xem hình ảnh có được đọc thành công không
    if hand_img is None:
        print(f"Error reading image: {file_path}")
        return np.zeros(63)  # Trả về mảng không có điểm đánh dấu

    # Xử lý hình ảnh
    # 1. Chuyển đổi BGR sang RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Lật hình ảnh theo trục Y
    img_flip = cv2.flip(img_rgb, 1)

    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)  # Giảm độ tin cậy

    # Kết quả
    output = hands.process(img_flip)
    hands.close()

    if output.multi_hand_landmarks:
        # Lấy dữ liệu điểm đánh dấu
        data = output.multi_hand_landmarks[0]
        # Chuyển đổi dữ liệu thành danh sách các tọa độ
        landmarks = []
        for landmark in data.landmark:
            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)

        return landmarks
    else:
        print(f"No hand landmarks detected in image: {file_path}")
        return np.zeros(63)  # Trả về mảng không có điểm đánh dấu

def make_csv():
    mypath = 'Assets/Dataset_SVM'  # Thư mục chứa dữ liệu
    csv_file = 'data_svm.csv'
    
    # Xóa file CSV cũ nếu tồn tại
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    file_name = open(csv_file, 'w', encoding='utf-8')  # Thêm encoding UTF-8

    # Duyệt qua Letters và Numbers
    for category_folder in os.listdir(mypath):
        if '._' in category_folder or not os.path.isdir(os.path.join(mypath, category_folder)):
            continue
            
        category_path = os.path.join(mypath, category_folder)
        print(f"\nProcessing category: {category_folder}")

        # Duyệt qua các thư mục con (A, B, C... hoặc 0, 1, 2...)
        for label_folder in os.listdir(category_path):
            if '._' in label_folder:
                continue

            if not os.path.isdir(os.path.join(category_path, label_folder)):
                continue

            label = label_folder  # Sử dụng tên thư mục làm nhãn
            label_path = os.path.join(category_path, label_folder)
            print(f"\nProcessing label: {label}")

            # Duyệt qua từng ảnh trong thư mục
            try:
                for image_file in os.listdir(label_path):
                    if '._' in image_file:
                        continue
                    
                    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    file_loc = os.path.join(label_path, image_file)
                    
                    try:
                        data = image_processed(file_loc)
                        if isinstance(data, (list, np.ndarray)) and len(data) == 63:  # Kiểm tra dữ liệu hợp lệ
                            file_name.write(','.join(map(str, data)))
                            file_name.write(f',{label}\n')
                            print(f"Processed: {category_folder}/{label}/{image_file}")
                        else:
                            print(f"Invalid data for {file_loc}")
                            file_name.write('0,' * 62 + '0,' + label + '\n')
                    
                    except Exception as e:
                        print(f"Error processing {file_loc}: {e}")
                        file_name.write('0,' * 62 + '0,' + label + '\n')
            except Exception as e:
                print(f"Error accessing directory {label_path}: {e}")

    file_name.close()
    print('\nData Created Successfully!!!')

if __name__ == "__main__":
    make_csv()