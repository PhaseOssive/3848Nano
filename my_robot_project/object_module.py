import os
from ultralytics import YOLO
import supervision as sv

class ObjectRecognizer:
    def __init__(self, model_path="best.pt"):
        """
        初始化物体识别模块
        :param model_path: 本地 .pt 权重文件的路径
        """
        # 1. 加载本地模型（不再需要 API_KEY 和 MODEL_ID）
        if not os.path.exists(model_path):
            print(f"⚠️ 错误：找不到权重文件 {model_path}，请确认文件已放入项目文件夹。")
        
        self.model = YOLO(model_path)
        
        # 2. 初始化标注器 (supervision 库用于画框)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame):
        """
        执行检测逻辑
        """
        # 3. 使用本地模型进行推理
        # imgsz=320 匹配你 main.py 的摄像头分辨率，提升速度
        results = self.model.predict(frame, conf=0.4, imgsz=320, verbose=False)[0]
        
        # 4. 转换为 supervision 格式以便后续处理
        detections = sv.Detections.from_ultralytics(results)

        # 5. 针对 paper_people 的特殊过滤逻辑 (保持你原有的阈值)
        mask = []
        for i in range(len(detections)):
            # 获取当前目标的类别索引和名称
            class_id = int(detections.class_id[i])
            class_name = self.model.names[class_id]
            conf = detections.confidence[i]
            
            if class_name == "paper_people":
                # 纸片人门槛设为 0.75，防止误报
                mask.append(conf > 0.65)
            else:
                # 其他物体（如 pink_cat）保持 0.6
                mask.append(conf > 0.55)
        
        # 执行过滤
        detections = detections[mask]
        
        # 6. 生成标签文本用于画面显示
        labels = [
            f"{self.model.names[int(class_id)]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
        
        return detections, labels
    