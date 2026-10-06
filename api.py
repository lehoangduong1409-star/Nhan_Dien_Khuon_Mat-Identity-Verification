# api.py
import os
import cv2
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from config import CFG
from face_engine import FaceEngine
from recognizer import FaceRecognizer

app = FastAPI(
    title="He thong Nhan dien Khuon mat 3m - DATN",
    description="API phuc vu nghiem thu va theo doi thoi gian thuc",
    version="1.0"
)

# --- KHỞI TẠO ĐƯỜNG DẪN CHUẨN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

gallery_path = os.path.join(project_root, 'gallery', 'face_gallery.npz')
csv_report_path = os.path.join(project_root, 'reports', 'recognition_results.csv')
crops_dir = os.path.join(project_root, 'evidence', 'crops')
frames_dir = os.path.join(project_root, 'evidence', 'frames')

# Tạo sẵn thư mục nếu chưa có
os.makedirs(crops_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(os.path.dirname(csv_report_path), exist_ok=True)

# Khởi tạo Engine và Trình nhận diện
engine = FaceEngine()
if os.path.exists(gallery_path):
    recognizer = FaceRecognizer(gallery_path, CFG)
    gallery_loaded = True
else:
    recognizer = None
    gallery_loaded = False


@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra xem hệ thống và cơ sở dữ liệu kho mẫu đang sống hay chết."""
    global gallery_loaded
    if not gallery_loaded and os.path.exists(gallery_path):
        global recognizer
        recognizer = FaceRecognizer(gallery_path, CFG)
        gallery_loaded = True

    return {
        "status": "healthy",
        "gallery_loaded": gallery_loaded,
        "config_threshold": CFG.recognition_threshold
    }


@app.post("/predict", tags=["Core Inference"])
async def predict_face(file: UploadFile = File(...)):
    """Tải một file ảnh lên để kiểm tra nhận diện khuôn mặt, lọc chất lượng và lưu bằng chứng."""
    if not gallery_loaded:
        return {"status": "Error", "message": "Chưa nạp được file kho mẫu .npz!"}

    # Đọc file ảnh từ request dữ liệu tải lên
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_raw is None:
        return {"status": "Error", "message": "File gửi lên không phải là ảnh hợp lệ"}

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    timestamp_file = time.strftime("%Y%m%d_%H%M%S")

    # 1. Phát hiện khuôn mặt tốt nhất
    face = engine.best_face(frame_raw)
    if face is None:
        with open(csv_report_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp},NoFace,,,,,,0,API: Không phát hiện mặt\n")
        return {"status": "NoFace", "message": "Không tìm thấy khuôn mặt nào trong ảnh"}

    x1, y1, x2, y2 = face.bbox.astype(int)
    face_w = int(x2 - x1)
    face_h = int(y2 - y1)

    # 2. Kiểm tra kích thước hình ảnh cận cảnh (khoảng cách 3m)
    if face_w < CFG.min_face_width or face_h < CFG.min_face_height:
        with open(csv_report_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp},Unknown,,,,,0,{face_w},API: Mặt quá nhỏ ở khoảng cách xa\n")
        return {"status": "Rejected", "message": f"Kích thước mặt quá nhỏ ({face_w}px), từ chối nhận diện"}

    # Cắt vùng khuôn mặt (Crop) phục vụ kiểm tra chất lượng
    h_img, w_img = frame_raw.shape[:2]
    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(w_img, x2), min(h_img, y2)
    face_crop = frame_raw[y1_c:y2_c, x1_c:x2_c]

    # 3. Kiểm tra chất lượng ảnh (Buổi 6)
    from quality import is_quality_ok
    q_ok, blur_score, brightness_score = is_quality_ok(face_crop, CFG)
    if not q_ok:
        with open(csv_report_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{timestamp},Unknown,,,,,{face_w},API: Anh mo({round(blur_score, 1)}) hoac sai sang({round(brightness_score, 1)})\n")
        return {
            "status": "Rejected",
            "message": f"Chất lượng ảnh kém. Độ mờ: {round(blur_score, 1)}, Độ sáng: {round(brightness_score, 1)}"
        }

    # 4. Trích xuất đặc trưng và so khớp thông tin nhân sự
    emb = engine.embedding_from_face(face)
    result = recognizer.search(emb)
    status, employee_code, candidate_code = recognizer.decide(result)

    # 5. Tự động lưu bằng chứng ảnh vào thư mục crops và frames (Buổi 4)
    identity = employee_code if employee_code else f"Cand_{candidate_code}"
    if status in ['Recognized', 'LowConfidence']:
        cv2.imwrite(os.path.join(crops_dir, f"{timestamp_file}_{status}_{identity}_crop.jpg"), face_crop)
        cv2.imwrite(os.path.join(frames_dir, f"{timestamp_file}_{status}_{identity}_frame.jpg"), frame_raw)

    # 6. Ghi dữ liệu log xuống file CSV báo cáo nghiệm thu
    with open(csv_report_path, 'a', encoding='utf-8') as f:
        f.write(
            f"{timestamp},{status},{employee_code or ''},{candidate_code or ''},{round(result['bestScore'], 4)},{round(result['secondBestScore'], 4)},{round(result['margin'], 4)},{face_w},API: Thao tac thanh cong\n")

    return {
        "status": status,
        "employee_code": employee_code,
        "candidate_code": candidate_code,
        "confidence_score": round(result['bestScore'], 4),
        "margin": round(result['margin'], 4)
    }


@app.get("/events", tags=["Dashboard"])
def get_recent_events():
    """Lấy danh sách 10 sự kiện nhận diện mới nhất từ file CSV để hiển thị lên bảng điều khiển."""
    if not os.path.exists(csv_report_path):
        return {"events": []}
    try:
        df = pd.read_csv(csv_report_path)
        if df.empty:
            return {"events": []}
        # Lấy 10 dòng cuối cùng (mới nhất) và đảo ngược thứ tự
        recent_df = df.tail(10).fillna("")
        events = recent_df.to_dict(orient="records")
        return {"events": events[::-1]}
    except Exception:
        return {"events": []}