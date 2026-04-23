import cv2
import time
import numpy as np
import face_recognition
import serial
from face_module import FaceRecognizer

# --- 配置区 ---
WINDOW_DURATION = 2.0  # 每次启动检测持续 2 秒
THRESHOLD_RATE = 0.6   
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600
TRIGGER_CMD = "start" # 启动触发命令

# --- Arduino 连接与发送逻辑 ---
arduino = None
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) # 缩短 timeout 提高响应
    print(f"✅ 串口已就绪: {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ 无法连接 Arduino ({e})，进入纯输出模式。")

def send_to_arduino(state_code, log_msg):
    print(log_msg)
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{state_code}\n".encode()) 
        except Exception as e:
            print(f"❌ 发送失败: {e}")

# --- 1. 初始化：程序启动即加载人脸库 ---
# 这样在接收到指令时可以立即开始识别，无需等待加载
face_engine = FaceRecognizer()
print("--- 系统就绪：进入待机模式，等待指令 ---")

def run_detection_session():
    """触发后开启摄像头检测 2 秒的逻辑"""
    print("🚀 收到指令！启动摄像头并开始检测...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    start_time = time.time()
    total_frames = 0
    face_hits = 0        
    unknown_face_hits = 0 
    count = 0

    # 持续运行 WINDOW_DURATION (2秒)
    while (time.time() - start_time) < WINDOW_DURATION:
        ret, frame = cap.read()
        if not ret: break
        
        count += 1
        if count % 3 != 0: continue # 跳帧加速
        
        frame = cv2.flip(frame, 1)
        total_frames += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        is_forced_face = False
        if not face_locations:
            h, w, _ = rgb_frame.shape
            face_locations = [(int(h*0.2), w-int(w*0.2), h-int(h*0.2), int(w*0.2))]
            is_forced_face = True

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        found_known = False
        found_unknown = False

        if face_encodings:
            for encoding in face_encodings:
                tolerance = 0.38 if is_forced_face else 0.45
                distances = face_recognition.face_distance(face_engine.known_encodings, encoding)
                if len(distances) > 0 and np.min(distances) < tolerance:
                    found_known = True
                elif len(distances) > 0 and np.min(distances) < 0.7:
                    found_unknown = True
        
        if found_known: face_hits += 1
        elif found_unknown: unknown_face_hits += 1

        cv2.imshow("Vision - Detecting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- 2秒结束，判定并发送结果 ---
    face_rate = face_hits / total_frames if total_frames > 0 else 0
    unknown_rate = unknown_face_hits / total_frames if total_frames > 0 else 0

    if face_rate >= THRESHOLD_RATE:
        send_to_arduino("state1", f"🌟 确认 Owen ({(face_rate*100):.1f}%)")
    elif unknown_rate >= THRESHOLD_RATE:
        send_to_arduino("state2", f"❓ 陌生人 ({(unknown_rate*100):.1f}%)")
    else:
        send_to_arduino("state3", "🌑 未发现目标")

    # 释放摄像头资源，进入待机
    cap.release()
    cv2.destroyAllWindows()
    print("💤 检测完成，摄像头已关闭，重新进入待机...\n")

# --- 主循环：待机监听串口 ---
try:
    while True:
        if arduino and arduino.in_waiting > 0:
            try:
                # 读取来自 Arduino 的原始指令
                raw_cmd = arduino.readline().decode().strip()
                if raw_cmd == TRIGGER_CMD:
                    run_detection_session()
            except Exception as e:
                print(f"读取指令出错: {e}")
        
        time.sleep(0.1) # 降低 CPU 占用

except KeyboardInterrupt:
    print("程序手动停止")
finally:
    if arduino: arduino.close()
    