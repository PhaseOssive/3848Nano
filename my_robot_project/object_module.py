from inference import get_model
import supervision as sv
import os

class ObjectRecognizer:
    def __init__(self, model_id, api_key):
        # 屏蔽无关警告
        os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
        os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
        os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
        
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame):
        # 执行推理，阈值逻辑与你之前的 do.py 一致
        results = self.model.infer(frame, confidence=0.4)[0]
        detections = sv.Detections.from_inference(results)

        # 针对 paper_people 的特殊过滤逻辑
        mask = []
        for i in range(len(detections)):
            class_name = detections.data['class_name'][i]
            conf = detections.confidence[i]
            if class_name == "paper_people":
                mask.append(conf > 0.85)
            else:
                mask.append(conf > 0.6)
        
        detections = detections[mask]
        
        # 生成标签文本用于显示
        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(detections.data['class_name'], detections.confidence)
        ]
        return detections, labels
    