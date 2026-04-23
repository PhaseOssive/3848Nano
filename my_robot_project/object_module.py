import cv2
import numpy as np
import supervision as sv

class ObjectRecognizer:
    def __init__(self, model_path="best.onnx"):
        # 使用 OpenCV 4.x 自带的推理引擎，无需 ultralytics 库
        self.net = cv2.dnn.readNet(model_path)
        # 类别名称（请务必确认顺序与你 Colab 训练时一致）
        self.classes = ["paper_people", "pink_cat", "test"] 
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame):
        h_orig, w_orig = frame.shape[:2]
        # 1. 预处理：缩放到 320x320 并归一化
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 2. 推理：获得原始张量
        # YOLOv8 ONNX 输出通常是 [1, 7, 2100] (如果是3个类: 4坐标+3类别)
        outputs = self.net.forward() 
        outputs = np.squeeze(outputs)
        outputs = outputs.transpose() # 变为 [2100, 7]

        boxes = []
        confidences = []
        class_ids = []

        # 3. 解析每一行数据
        for row in outputs:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]
            
            # 过滤低置信度
            if confidence > 0.4:
                # YOLO 输出的是中心点格式 [cx, cy, w, h]
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # 缩放回原始图像尺寸
                left = int((cx - w/2) * (w_orig / 320))
                top = int((cy - h/2) * (h_orig / 320))
                width = int(w * (w_orig / 320))
                height = int(h * (h_orig / 320))
                
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # 4. 非极大值抑制 (NMS) 防止重复框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)
        
        # 5. 封装进 Detections 对象以适配你的 main.py
        if len(indices) > 0:
            # 处理不同版本 OpenCV 的 indices 返回格式差异
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            
            final_boxes = np.array([boxes[i] for i in indices])
            # 将 [x, y, w, h] 转为 supervision 的 [x1, y1, x2, y2]
            xyxy = final_boxes.copy()
            xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
            xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]

            detections = sv.Detections(
                xyxy=xyxy.astype(np.float32),
                confidence=np.array([confidences[i] for i in indices]),
                class_id=np.array([class_ids[i] for i in indices]),
                data={'class_name': [self.classes[class_ids[i]] for i in indices]}
            )
            
            # 针对 paper_people 的二次过滤
            mask = []
            for i in range(len(detections)):
                name = detections.data['class_name'][i]
                conf = detections.confidence[i]
                mask.append(conf > 0.75 if name == "paper_people" else conf > 0.5)
            
            detections = detections[np.array(mask)]
            
            labels = [f"{name} {conf:.2f}" for name, conf in zip(detections.data['class_name'], detections.confidence)]
            return detections, labels
        
        return sv.Detections.empty(), []
    