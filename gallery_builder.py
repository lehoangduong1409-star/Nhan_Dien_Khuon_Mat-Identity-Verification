import os
import cv2
import numpy as np
from pathlib import Path
from face_engine import FaceEngine

BASE_DIR = Path(__file__).resolve().parents[1]
EMP_DIR = BASE_DIR / 'data' / 'employees'
OUT = BASE_DIR / 'gallery' / 'face_gallery.npz'
OUT.parent.mkdir(parents=True, exist_ok=True)

engine = FaceEngine()
embeddings = []
employee_codes = []
image_paths = []

print(">>> Hệ thống đang quét thư mục ảnh nhân sự...")

if EMP_DIR.exists():
    for emp_folder in EMP_DIR.iterdir():
        if not emp_folder.is_dir(): continue
        emp_code = emp_folder.name

        for img_path in emp_folder.glob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
            img = cv2.imread(str(img_path))
            if img is None: continue

            face = engine.best_face(img)
            if face is None:
                print(f"X Không phát hiện mặt: {img_path.name}")
                continue

            emb = engine.embedding_from_face(face)
            embeddings.append(emb)
            employee_codes.append(emp_code)
            image_paths.append(str(img_path))
            print(f"V Đã nạp đặc trưng: {emp_code} -> {img_path.name}")

if len(embeddings) > 0:
    np.savez(OUT, embeddings=np.vstack(embeddings), employee_codes=np.array(employee_codes),
             image_paths=np.array(image_paths))
    print(f"\n>>> HOÀN THÀNH: Đã lưu kho mẫu tại: {OUT}")
else:
    print("\n>>> THẤT BẠI: Vui lòng kiểm tra lại ảnh mẫu.")