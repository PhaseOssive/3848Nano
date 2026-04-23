import cv2
import time
import numpy as np
import face_recognition
import serial  # 需要安装: pip3 install pyserial
from face_module import FaceRecognizer

# --- 配置区 ---
WINDOW_DURATION = 2.0  
THRESHOLD_RATE = 0.6   
SERIAL_PORT = '/dev/ttyACM0'  # Nano上Arduino通常的端口，也可能是 /dev/ttyUSB0
BAUD_RATE = 9600

# --- Arduino 连接回退机制 ---
arduino = None
try:
    # 尝试建立串口连接
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✅ 成功连接到 Arduino: {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ 无法连接 Arduino ({e})，进入纯输出模式。")

def send_to_arduino(state_code, log_msg):
    """发送数据到 Arduino 的辅助函数，包含回退逻辑"""
    print(log_msg) # 无论如何都打印输出
    if arduino and arduino.is_open:
        try:
            # 发送状态码并换行，Arduino端用 readStringUntil('\n') 接收比较稳
            arduino.write(f"{state_code}\n".encode()) 
        except Exception as e:
            print(f"❌ 发送失败: {e}")

# --- 初始化识别引擎 ---
face_engine = FaceRecognizer()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- 统计变量 ---
start_time = time.time()
total_frames = 0
face_hits = 0        
unknown_face_hits = 0 

print("--- 系统启动：人脸识别 + Arduino 同步模式 ---")
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    count += 1
    if count % 3 != 0: 
        cv2.imshow("Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue
    
    frame = cv2.flip(frame, 1)
    total_frames += 1
    current_frame_messages = []

    # 1. 人脸检测逻辑
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    is_forced_face = False
    if not face_locations:
        h, w, _ = rgb_frame.shape
        face_locations = [(int(h*0.2), w-int(w*0.2), h-int(h*0.2), int(w*0.2))]
        is_forced_face = True

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    found_known_this_frame = False
    found_unknown_this_frame = False

    if len(face_encodings) > 0:
        for encoding in face_encodings:
            tolerance = 0.38 if is_forced_face else 0.45
            distances = face_recognition.face_distance(face_engine.known_encodings, encoding)
            
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist < tolerance:
                    found_known_this_frame = True
                    name_found = face_engine.known_names[np.argmin(distances)]
                    current_frame_messages.append(f"[人脸] 认识: {name_found}")
                elif min_dist < 0.7: 
                    found_unknown_this_frame = True
    
    if found_known_this_frame: face_hits += 1
    if found_unknown_this_frame and not found_known_this_frame: unknown_face_hits += 1

    # --- 2. 两秒周期判定逻辑 ---
    elapsed = time.time() - start_time
    if elapsed >= WINDOW_DURATION:
        face_rate = face_hits / total_frames if total_frames > 0 else 0
        unknown_rate = unknown_face_hits / total_frames if total_frames > 0 else 0

        # --- 分类发送状态 ---
        if face_rate >= THRESHOLD_RATE:
            msg = f"🌟 【最终确认】我真的看到了 Owen (频率: {(face_rate*100):.1f}%)"
            send_to_arduino("state1", msg)
        
        elif unknown_rate >= THRESHOLD_RATE:
            msg = f"❓ 我不认识这个人 (陌生人频率: {(unknown_rate*100):.1f}%)"
            send_to_arduino("state2", msg)
            
        else:
            msg = "🌑 我什么都没看到..."
            send_to_arduino("state3", msg)

        # 重置统计量
        start_time, total_frames, face_hits, unknown_face_hits = time.time(), 0, 0, 0

    # --- 3. 渲染人脸框 ---
    for (top, right, bottom, left) in face_locations:
        color = (0, 0, 255) if found_unknown_this_frame else (0, 255, 0)
        if is_forced_face: color = (0, 255, 255) 
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    cv2.imshow("Vision - Arduino Sync", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 退出时记得关闭串口
if arduino: arduino.close()
cap.release()
cv2.destroyAllWindows()
