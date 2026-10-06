# face_engine.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, det_size=(640, 640)):
        # Khởi động mô hình InsightFace
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, image_bgr):
        return self.app.get(image_bgr)

    @staticmethod
    def l2_normalize(vec):
        # Chuẩn hóa vector độ dài hình học về bằng 1 để thực hiện phép toán so khớp Cosine nhanh hơn
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def best_face(self, image_bgr):
        faces = self.detect_faces(image_bgr)
        if not faces:
            return None
        # Nguyên tắc khoảng cách 3m: Nếu có nhiều mặt trong khung hình, chọn mặt có diện tích hộp bao lớn nhất tiếp cận gần camera nhất
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

    def embedding_from_face(self, face):
        return self.l2_normalize(face.embedding)

    @staticmethod
    def crop_face(image_bgr, face, pad=0.25):
        # Trích xuất và mở rộng vùng cắt khuôn mặt làm dữ liệu bằng chứng (evidence crop)
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        return image_bgr[y1:y2, x1:x2].copy()
