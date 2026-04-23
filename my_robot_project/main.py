import cv2
import time
import numpy as np
import face_recognition
from face_module import FaceRecognizer
from object_module import ObjectRecognizer

# --- 配置区 ---
# API_KEY = "OU8ODWSeWojje7T6mzms"
# MODEL_ID = "my-first-project-smgqf/1"
WINDOW_DURATION = 2.0  
THRESHOLD_RATE = 0.6   

# --- 初始化 ---
face_engine = FaceRecognizer()
obj_engine = ObjectRecognizer("best.onnx")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- 统计变量 ---
start_time = time.time()
total_frames = 0
face_hits = 0        # 认识的人脸次数
unknown_face_hits = 0 # 不认识的人脸次数
obj_hits = 0         # 物体检测次数

print("--- 系统启动：多模态同步检测中 ---")
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0: # 每 3 帧才跑一次 AI
        cv2.imshow("Vision", frame)
        continue
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    total_frames += 1
    current_frame_messages = []

    # 1. 人脸检测逻辑
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    is_forced_face = False
    if not face_locations:
        # 强制聚焦模式下，我们视为“疑似有人脸”
        h, w, _ = rgb_frame.shape
        face_locations = [(int(h*0.2), w-int(w*0.2), h-int(h*0.2), int(w*0.2))]
        is_forced_face = True

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    found_known_this_frame = False
    found_unknown_this_frame = False

    if len(face_encodings) > 0:
        for encoding in face_encodings:
            # 这里的逻辑根据你 face_module 的 tolerance 进行匹配
            tolerance = 0.38 if is_forced_face else 0.45
            distances = face_recognition.face_distance(face_engine.known_encodings, encoding)
            
            if len(distances) > 0:
                min_dist = np.min(distances)
                # 严格检查：距离越小越可信
                if min_dist < tolerance:
                    found_known_this_frame = True
                    name_found = face_engine.known_names[np.argmin(distances)]
                    current_frame_messages.append(f"[人脸] 认识: {name_found}")
                elif min_dist < 0.7: # 介于两者之间，说明确实是人脸，但不是 Owen
                    found_unknown_this_frame = True
    
    # 统计计次
    if found_known_this_frame: face_hits += 1
    if found_unknown_this_frame and not found_known_this_frame: unknown_face_hits += 1

    # 2. 物体检测逻辑
    obj_detections, obj_labels = obj_engine.detect(frame)
    if len(obj_detections) > 0:
        obj_hits += 1
        classes = list(set(obj_detections.data['class_name']))
        current_frame_messages.append(f"[物体] 看到: {classes}")

    # --- 实时即时输出 ---
    if current_frame_messages:
        print(" | ".join(current_frame_messages))

    # --- 3. 两秒周期判定逻辑 ---
    elapsed = time.time() - start_time
    if elapsed >= WINDOW_DURATION:
        face_rate = face_hits / total_frames if total_frames > 0 else 0
        obj_rate = obj_hits / total_frames if total_frames > 0 else 0
        unknown_rate = unknown_face_hits / total_frames if total_frames > 0 else 0

        # --- 判定优先级结构 ---
        # 优先级 1: 认识的人 或 物体 达标
        if face_rate >= THRESHOLD_RATE and obj_rate >= THRESHOLD_RATE:
            print(f"🌟 【最终确认】我同时看到了 Owen ({(face_rate*100):.1f}%) 和 目标物体 ({(obj_rate*100):.1f}%)")
        elif face_rate >= THRESHOLD_RATE:
            print(f"🌟 【最终确认】我真的看到了 Owen (频率: {(face_rate*100):.1f}%)")
        elif obj_rate >= THRESHOLD_RATE:
            print(f"🌟 【最终确认】我真的看到了 目标物体 (频率: {(obj_rate*100):.1f}%)")
        
        # 优先级 2: 只有不认识的人脸达标
        elif unknown_rate >= THRESHOLD_RATE:
            print(f"❓ 我不认识这个人 (陌生人频率: {(unknown_rate*100):.1f}%)")
            
        # 优先级 3: 啥也没达标
        else:
            print("🌑 我什么都没看到...")

        # 重置
        start_time = time.time()
        total_frames = 0
        face_hits = 0
        obj_hits = 0
        unknown_face_hits = 0

    # --- 4. 渲染 ---
    frame = obj_engine.box_annotator.annotate(scene=frame, detections=obj_detections)
    frame = obj_engine.label_annotator.annotate(scene=frame, detections=obj_detections, labels=obj_labels)
    for (top, right, bottom, left) in face_locations:
        # 颜色区分：认识绿色，不认识红色
        color = (0, 0, 255) if found_unknown_this_frame else (0, 255, 0)
        if is_forced_face: color = (0, 255, 255) # 聚焦模式黄色
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    cv2.imshow("Integrated Vision - Owen", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
