import face_recognition
import cv2
import os
import time
import numpy as np
from PIL import Image, ImageOps

def force_load_encoding(path):
    print(f"正在读取样本: {path}...", end="")
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img_array = np.array(img.convert("RGB"))
        
        height, width, _ = img_array.shape
        face_location = [(0, width, height, 0)] 
        
        encodings = face_recognition.face_encodings(img_array, known_face_locations=face_location, num_jitters=10)
        
        if len(encodings) > 0:
            print(" ✅ 成功！")
            return encodings[0]
        else:
            print(" ❌ 无法提取特征。")
            return None
    except Exception as e:
        print(f" 错误: {e}")
        return None

known_encodings = []
known_names = []

# --- 1. 遍历文件夹加载 ---
image_folder = "known_faces"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"请在 {image_folder} 文件夹中放入你的照片")

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(image_folder, filename)
        
        if "low" in filename.lower():
            encoding = force_load_encoding(path)
        else:
            print(f"正在优化加载正脸: {path}...", end="")
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img_array = np.array(img.convert("RGB"))
            h, w, _ = img_array.shape

            encodings = face_recognition.face_encodings(img_array, num_jitters=10)
            
            if len(encodings) > 0:
                encoding = encodings[0]
                print(" ✅ 标准识别成功！")
            else:
                print(" ⚠️ 尝试中心聚焦提取...", end="")
                margin_h = int(h * 0.1)
                margin_w = int(w * 0.1)
                focus_location = [(margin_h, w - margin_w, h - margin_h, margin_w)]
                
                encodings = face_recognition.face_encodings(img_array, known_face_locations=focus_location, num_jitters=10)
                if len(encodings) > 0:
                    encoding = encodings[0]
                    print(" ✅ 聚焦提取成功！")
                else:
                    print(" ❌ 彻底失败。")
                    encoding = None
        
        if encoding is not None:
            known_encodings.append(encoding)
            # 【核心修改 1】：把文件名存入大脑，方便我们抓内鬼！
            known_names.append(f"OwenLi [{filename}]")

print(f"--- 库加载完成，共记录 {len(known_encodings)} 张人脸样本 ---")

# --- 2. 摄像头识别 ---
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    is_forced_mode = False 
    
    if not face_locations:
        h, w, _ = rgb_frame.shape
        margin_h, margin_w = int(h * 0.2), int(w * 0.2)
        face_locations = [(margin_h, w - margin_w, h - margin_h, margin_w)]
        is_forced_mode = True

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    found_known = False
    min_dist = 1.0 
    matched_name = "Unknown"
    
    for face_encoding in face_encodings:
        current_tolerance = 0.45 if is_forced_mode else 0.6
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=current_tolerance)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if len(face_distances) > 0:
            # 【核心修改 2】：找出最像的那一张，并提取它的名字（文件名）
            best_match_index = np.argmin(face_distances)
            min_dist = face_distances[best_match_index]
            
            if matches[best_match_index]:
                found_known = True
                matched_name = known_names[best_match_index]
                break 

    # 3. 状态输出
    if found_known:
        if is_forced_mode:
            print(f"-> 状态: [极限仰角识别] 认识 {matched_name} (距离: {min_dist:.4f} < 0.45)")
        else:
            print(f"-> 状态: [标准识别] 认识 {matched_name} (距离: {min_dist:.4f} < 0.60)")
    else:
        if is_forced_mode:
            print(f"状态: 没找到人脸 (极端角度盲比对失败, 距离: {min_dist:.4f})")
        else:
            print(f"-> 状态: 不认识 (Unknown) (距离: {min_dist:.4f})")

    # 4. 显示与交互
    for (top, right, bottom, left) in face_locations:
        color = (0, 255, 255) if is_forced_mode else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    cv2.imshow('Video', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("\n[操作] 尝试动态保存当前姿势...")
        timestamp = int(time.time())
        
        if not is_forced_mode:
            new_encoding = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=5)[0]
            known_encodings.append(new_encoding)
            filename = f"me_dynamic_{timestamp}.jpg"
            known_names.append(f"OwenLi [{filename}]")
            
            save_path = f"known_faces/{filename}"
            cv2.imwrite(save_path, frame)
            print(f" ✅ 标准动态特征已保存: {save_path}")
            
        else:
            print(" ⚠️ 算法没找到脸！强制截取画面中心区域作为特征...")
            forced_enc = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations, num_jitters=5)
            
            if forced_enc:
                known_encodings.append(forced_enc[0])
                filename = f"me_low_dynamic_{timestamp}.jpg"
                known_names.append(f"OwenLi [{filename}]")
                
                save_path = f"known_faces/{filename}"
                cv2.imwrite(save_path, frame)
                print(f" ✅ 极端特征已保存: {save_path}")
            else:
                print(" ❌ 强制截取也失败了。")

video_capture.release()
cv2.destroyAllWindows()