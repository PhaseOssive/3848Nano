print('Starting face recognition module...')
import cv2
print('import cv2 successful!')
import os
print('import os successful!')
import numpy as np
print('import numpy successful!')
from PIL import Image, ImageOps
print('import PIL successful!')
import face_recognition
print('import face_recognition successful!')

class FaceRecognizer:
    def __init__(self, folder="known_faces"):
        self.known_encodings = []
        self.known_names = []
        self.folder = folder
        self._load_faces()

    def _force_load_encoding(self, path):
        """针对极端角度或低质量图片的强制提取逻辑"""
        print(f"正在读取样本 (强制模式): {path}...", end="", flush=True)
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            # 缩放以节省 Nano 内存
            if img.width > 640:
                img = img.resize((640, int(640 * img.height / img.width)))
            
            img_array = np.array(img.convert("RGB"))
            height, width, _ = img_array.shape
            # 强制指定全图范围为人脸区域
            face_location = [(0, width, height, 0)] 
            
            encodings = face_recognition.face_encodings(img_array, known_face_locations=face_location, num_jitters=1)
            
            if len(encodings) > 0:
                print(" ✅ 成功！", flush=True)
                return encodings[0]
            else:
                print(" ❌ 无法提取特征。", flush=True)
                return None
        except Exception as e:
            print(f" 错误: {e}", flush=True)
            return None

    def _load_faces(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            print(f"请在 {self.folder} 文件夹中放入照片")

        print('--- 开始加载人脸库 ---')
        for filename in os.listdir(self.folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.folder, filename)
                encoding = None
                
                # 逻辑 1: 如果文件名带 low，直接进入强制提取
                if "low" in filename.lower():
                    encoding = self._force_load_encoding(path)
                else:
                    # 逻辑 2: 标准加载尝试
                    print(f"正在优化加载正脸: {path}...", end="", flush=True)
                    img = Image.open(path)
                    img = ImageOps.exif_transpose(img)
                    img_array = np.array(img.convert("RGB"))
                    
                    encodings = face_recognition.face_encodings(img_array, num_jitters=1)
                    
                    if len(encodings) > 0:
                        encoding = encodings[0]
                        print(" ✅ 标准识别成功！")
                    else:
                        # 逻辑 3: 标准识别失败，尝试中心聚焦提取
                        print(" ⚠️ 标准失败，尝试中心聚焦...", end="", flush=True)
                        h, w, _ = img_array.shape
                        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
                        focus_loc = [(margin_h, w - margin_w, h - margin_h, margin_w)]
                        
                        encodings = face_recognition.face_encodings(img_array, known_face_locations=focus_loc, num_jitters=1)
                        if len(encodings) > 0:
                            encoding = encodings[0]
                            print(" ✅ 聚焦识别成功！")
                        else:
                            # 逻辑 4: 最后的绝招，强制全图提取
                            encoding = self._force_load_encoding(path)
                
                if encoding is not None:
                    self.known_encodings.append(encoding)
                    self.known_names.append(f"OwenLi [{filename}]")

        print(f"--- 库加载完成，共记录 {len(self.known_encodings)} 张人脸样本 ---")
        