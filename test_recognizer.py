# test_recognizer.py
import os
import cv2
from config import CFG
from face_engine import FaceEngine
from recognizer import FaceRecognizer


def run_test():
    engine = FaceEngine()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    gallery_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'gallery',
                                'face_gallery.npz')

    if not os.path.exists(gallery_path):
        print(f"❌ Lỗi hệ thống: Chưa tìm thấy kho mẫu tại {gallery_path}. Vui lòng chạy file gallery_builder.py trước!")
        return

    recognizer = FaceRecognizer(gallery_path, CFG)

    # Chỉ định file ảnh test đơn mục tiêu
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'data', 'employees',
                            'EMP001', 'download.jpg')
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Không tìm thấy hoặc không đọc được ảnh test tại: {img_path}")
        return

    face = engine.best_face(img)
    if face is None:
        print("❌ Không tìm thấy khuôn mặt trong ảnh test!")
        return

    emb = engine.embedding_from_face(face)
    result = recognizer.search(emb)
    status, emp_code, cand_code = recognizer.decide(result)

    print("\n" + "=" * 15 + " KẾT QUẢ NGHIỆM THU BUỔI 3 " + "=" * 15)
    print(f"Trạng thái phân loại (Status): {status}")
    print(f"Mã nhân viên chính thức xác định: {emp_code}")
    print(f"Mã nhân viên nằm trong diện nghi vấn: {cand_code}")
    print(f"Điểm số Top-1 tương đồng (bestScore): {round(result['bestScore'], 4)}")
    print(f"Điểm số Top-2 tương đồng (secondBestScore): {round(result['secondBestScore'], 4)}")
    print(f"Khoảng cách biên an toàn biệt lập (Margin): {round(result['margin'], 4)}")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    run_test()