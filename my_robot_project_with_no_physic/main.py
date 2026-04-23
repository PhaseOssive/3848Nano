import cv2
import time
import numpy as np
import face_recognition
import serial
from face_module import FaceRecognizer

# --- 配置区 ---
DETECTION_TIMEOUT = 9.0  # 触发后持续检测 9 秒
THRESHOLD_RATE = 0.6    
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600
TRIGGER_CMD = "XXXXXX"  # Arduino 发送的触发指令

# --- Arduino 连接与发送逻辑 ---
arduino = None
try:
    # 设置较短的 timeout 以保证主循环监听指令时的响应速度
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"✅ 串口已就绪: {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ 无法连接 Arduino ({e})，进入纯输出模式。")

def send_to_arduino(state_code, log_msg):
    """打印并向 Arduino 发送状态"""
    print(log_msg)
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{state_code}\n".encode()) 
        except Exception as e:
            print(f"❌ 发送失败: {e}")

# --- 1. 程序启动：预加载人脸图片库 ---
# 先加载完图片，这样触发时可以立即识别，无需等待 IO 加载
print("🚀 正在初始化人脸识别引擎并加载图片库...")
face_engine = FaceRecognizer()
print("--- 系统就绪：进入待机模式，等待指令 'XXXXXX' ---")

def run_9s_detection_session():
    """收到指令后的 9 秒识别逻辑"""
    print("\n🔔 [收到指令] 启动摄像头，开始 9 秒循环检测...")
    
    cap = cv2.VideoCapture(0)
    # 适当降低分辨率以保证 Nano 上的帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    session_start = time.time()
    total_frames = 0
    face_hits = 0        
    unknown_face_hits = 0 
    count = 0

    # 持续运行 9 秒
    while (time.time() - session_start) < DETECTION_TIMEOUT:
        ret, frame = cap.read()
        if not ret: break
        
        count += 1
        if count % 3 != 0: # 跳帧处理，减轻 Nano 负担
            continue
        
        frame = cv2.flip(frame, 1)
        total_frames += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # 强制聚焦模式（当你代码中没检测到脸时的补偿逻辑）
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
                
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    if min_dist < tolerance:
                        found_known = True
                    elif min_dist < 0.7:
                        found_unknown = True
        
        if found_known: 
            face_hits += 1
        elif found_unknown: 
            unknown_face_hits += 1

        # 在 9 秒检测期间实时显示画面（可选）
        cv2.imshow("Detection Session", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # --- 9 秒时间到，计算这段时间的整体频率并反馈 ---
    print(f"⌛ 9 秒检测结束。总有效帧数: {total_frames}")
    
    face_rate = face_hits / total_frames if total_frames > 0 else 0
    unknown_rate = unknown_face_hits / total_frames if total_frames > 0 else 0

    if face_rate >= THRESHOLD_RATE:
        send_to_arduino("state1", f"🌟 最终判定: 看到 Owen (置信频率: {face_rate:.2%})")
    elif unknown_rate >= THRESHOLD_RATE:
        send_to_arduino("state2", f"❓ 最终判定: 发现陌生人 (置信频率: {unknown_rate:.2%})")
    else:
        send_to_arduino("state3", "🌑 最终判定: 未发现有效目标")

    # 释放资源，彻底关闭摄像头进入待机
    cap.release()
    cv2.destroyAllWindows()
    print("💤 任务结束，摄像头已关闭，重新进入待机...\n")

# --- 主循环：串口监听器 ---
try:
    while True:
        # 检查串口是否有数据进来
        if arduino and arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode().strip()
                if line == TRIGGER_CMD:
                    run_9s_detection_session()
            except Exception as e:
                print(f"读取指令时出错: {e}")
        
        # 待机时极低频率检测，节省 CPU
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n程序手动停止")
finally:
    if arduino:
        arduino.close()