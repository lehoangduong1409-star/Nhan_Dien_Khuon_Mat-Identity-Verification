# video_runner.py
import os
import cv2
import time
from config import CFG
from face_engine import FaceEngine
from recognizer import FaceRecognizer


def run_video_test():
    engine = FaceEngine()

    # Chuẩn hóa đường dẫn theo cấu trúc thư mục thực tế của bạn
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Thư mục src
    project_root = os.path.dirname(current_dir)  # Thư mục face_recognition_3m

    gallery_path = os.path.join(project_root, 'gallery', 'face_gallery.npz')
    video_path = os.path.join(project_root, 'data', 'test_videos', '3m_test_01.mp4')
    csv_report_path = os.path.join(project_root, 'reports', 'recognition_results.csv')

    # Định vị thư mục lưu bằng chứng ảnh (Crops & Frames)
    crops_dir = os.path.join(project_root, 'evidence', 'crops')
    frames_dir = os.path.join(project_root, 'evidence', 'frames')
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.exists(gallery_path):
        print("❌ Lỗi: Chưa có file kho mẫu .npz, hãy chạy gallery_builder.py trước!")
        return

    # Tạo sẵn thư mục chứa báo cáo và file csv tiêu đề nếu chưa có
    os.makedirs(os.path.dirname(csv_report_path), exist_ok=True)
    if not os.path.exists(csv_report_path):
        with open(csv_report_path, 'w', encoding='utf-8') as f:
            f.write(
                "timestamp,recognitionStatus,employeeCode,candidateEmployeeCode,bestScore,secondBestScore,margin,faceBoxWidth,note\n")

    recognizer = FaceRecognizer(gallery_path, CFG)

    # Kiểm tra file video đầu vào, nếu chưa có video thật hệ thống sẽ tự động dùng Camera/Webcam máy tính
    if os.path.exists(video_path):
        print(f"📹 Đang nạp file video thực nghiệm: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("⚠️ Không tìm thấy file video mẫu tại data/test_videos/, hệ thống chuyển sang gọi Webcam...")
        cap = cv2.VideoCapture(0)

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(source_fps / CFG.frame_sample_fps))  # Tính toán bước nhảy khung hình dựa trên FPS cấu hình
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue  # Bỏ qua khung hình thừa để giảm tải cho CPU

        # Tạo chuỗi thời gian không dấu để làm tên file ảnh trùng khớp
        timestamp_file = time.strftime("%Y%m%d_%H%M%S")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        face = engine.best_face(frame)

        if face is None:
            # Ghi nhận trạng thái không tìm thấy khuôn mặt vào CSV
            with open(csv_report_path, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp},NoFace,,,,,,0,Không phát hiện mặt\n")
            cv2.imshow("Nghiem thu He thong", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        x1, y1, x2, y2 = face.bbox.astype(int)
        face_w = int(x2 - x1)
        face_h = int(y2 - y1)

        # Lưu bản sao ảnh gốc (chưa vẽ khung) vào thư mục frames phục vụ lưu trữ gốc
        frame_raw = frame.copy()

        # Vẽ trực quan hóa khung nhận diện lên màn hình hiển thị
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Kiểm tra nhanh kích thước mặt ở khoảng cách 3m
        if face_w < CFG.min_face_width or face_h < CFG.min_face_height:
            with open(csv_report_path, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp},Unknown,,,,,0,{face_w},Mặt quá nhỏ ở khoảng cách xa\n")
            cv2.imshow("Nghiem thu He thong", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        # ================= BUỔI 6: KIỂM TRA CHẤT LƯỢNG ẢNH KHUÔN MẶT =================
        h_img, w_img = frame.shape[:2]
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w_img, x2), min(h_img, y2)
        face_crop = frame_raw[y1_c:y2_c, x1_c:x2_c]  # Lấy từ ảnh gốc sạch

        from quality import is_quality_ok
        q_ok, blur_score, brightness_score = is_quality_ok(face_crop, CFG)

        if not q_ok:
            with open(csv_report_path, 'a', encoding='utf-8') as f:
                f.write(
                    f"{timestamp},Unknown,,,,,{face_w},Anh mo({round(blur_score, 1)}) hoac sai sang({round(brightness_score, 1)})\n")

            cv2.putText(frame, "⚠️ Bad Quality (Blur/Lighting)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.imshow("Nghiem thu He thong", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue
        # ==============================================================================

        emb = engine.embedding_from_face(face)
        result = recognizer.search(emb)
        status, employee_code, candidate_code = recognizer.decide(result)

        # 📸 TỰ ĐỘNG LƯU BẰNG CHỨNG CHO BUỔI 4 VÀ BUỔI 6 (CROPS & FRAMES)
        # Chỉ lưu khi hệ thống nhận diện thành công (Recognized) hoặc nghi vấn (LowConfidence)
        if status in ['Recognized', 'LowConfidence']:
            identity = employee_code if employee_code else f"Cand_{candidate_code}"

            # 1. Lưu ảnh khuôn mặt cận cảnh vào thư mục crops
            crop_name = f"{timestamp_file}_{status}_{identity}_crop.jpg"
            cv2.imwrite(os.path.join(crops_dir, crop_name), face_crop)

            # 2. Lưu ảnh toàn cảnh không gian vào thư mục frames
            frame_name = f"{timestamp_file}_{status}_{identity}_frame.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_name), frame_raw)

        # Ghi log chi tiết dữ liệu nhận diện xuống file CSV
        with open(csv_report_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{timestamp},{status},{employee_code or ''},{candidate_code or ''},{round(result['bestScore'], 4)},{round(result['secondBestScore'], 4)},{round(result['margin'], 4)},{face_w},Chat luong dat chuan. Da luu bang chung.\n")

        # Hiển thị chữ trạng thái lên video
        cv2.putText(frame, f"{status}: {employee_code or 'Unknown'}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.imshow("Nghiem thu He thong", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn nút ESC để thoát luồng video
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Hoàn thành! Báo cáo kết quả thực địa đã xuất ra tại: {csv_report_path}")


if __name__ == '__main__':
    run_video_test()