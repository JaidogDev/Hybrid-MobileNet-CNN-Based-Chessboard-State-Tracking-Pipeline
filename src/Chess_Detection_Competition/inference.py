# src/inference.py
import cv2
import numpy as np
from collections import deque
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input


from .board import warp_board, split_grid
from .model import load_model

# ลำดับคลาสต้องตรงกับตอนเทรน
CLASSES = [
    "Empty","WP","WN","WB","WR","WQ","WK",
    "BP","BN","BB","BR","BQ","BK"
]

class TemporalBoardPredictor:
    """
    ใช้โมเดล per-cell classifier ทำนายกระดาน 8x8 ต่อเฟรม
    พร้อม smooth ผลลัพธ์แบบ moving average เพื่อกันกระพริบ
    """
    def __init__(self, model_path, img_size=96, smooth_k=5):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.buffers = [[deque(maxlen=smooth_k) for _ in range(8)] for __ in range(8)]

    def predict_labels8x8(self, warped_bgr):
        """
        รับภาพบอร์ดที่ warp แล้ว (สี่เหลี่ยมจัตุรัส)
        คืนเมทริกซ์ 8x8 ของ label ชนิดหมาก
        """
        cells = split_grid(warped_bgr, self.img_size)

        # เตรียม batch และ preprocess ให้ตรงกับ MobileNetV2 (เหมือนตอนเทรน)
        X = []
        for _, patch in cells:
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            x = preprocess_input(rgb.astype(np.float32))  # <-- สำคัญ: preprocess_input
            X.append(x)
        X = np.asarray(X, dtype=np.float32)

        probs = self.model.predict(X, verbose=0)

        mat = [[None]*8 for _ in range(8)]
        k = 0
        for r in range(8):
            for c in range(8):
                # smoothing ด้วยค่าเฉลี่ยของ prob ใน buffer
                self.buffers[r][c].append(probs[k])
                avg = np.mean(np.stack(self.buffers[r][c], axis=0), axis=0)
                mat[r][c] = CLASSES[int(np.argmax(avg))]
                k += 1
        return mat

    def predict_from_frame(self, bgr_frame, cfg):
        """
        เผื่อใช้สะดวก: รับเฟรมดิบ -> warp -> ทำนาย 8x8
        """
        warped, _ = warp_board(bgr_frame, cfg)
        return self.predict_labels8x8(warped)
