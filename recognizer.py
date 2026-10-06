# face_recognition_3m/src/recognizer.py
import numpy as np


class FaceRecognizer:
    def __init__(self, gallery_npz_path, cfg):
        # Nạp dữ liệu bộ nhớ mẫu đã tạo từ Buổi 2
        data = np.load(gallery_npz_path, allow_pickle=True)
        self.embeddings = data['embeddings'].astype(np.float32)
        self.employee_codes = data['employee_codes']
        self.cfg = cfg

    def search(self, query_embedding):
        """
        So khớp vector khuôn mặt test với kho mẫu bằng Cosine Similarity.
        Vì vector đầu ra của InsightFace đã được chuẩn hóa L2, phép nhân ma trận
        (Dot Product `@`) chính là tính toán điểm tương đồng Cosine.
        """
        # Nhân ma trận để tính điểm tương đồng với toàn bộ kho dữ liệu
        scores = self.embeddings @ query_embedding.astype(np.float32)

        # Sắp xếp chỉ số mảng theo thứ tự điểm số giảm dần (từ cao xuống thấp)
        order = np.argsort(scores)[::-1]

        # Trích xuất vị trí của ứng viên giống nhất (Top-1) và giống nhì (Top-2)
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx

        # Lấy điểm số tương ứng
        best_score = float(scores[best_idx])
        second_score = float(scores[second_idx])

        # Tính độ lệch an toàn (margin) để phân biệt top-1 và top-2
        margin = best_score - second_score

        return {
            'bestEmployeeCode': str(self.employee_codes[best_idx]),
            'bestScore': best_score,
            'secondBestScore': second_score,
            'margin': margin
        }

    def decide(self, search_result):
        """
        Áp dụng quy tắc ép buộc trạng thái (Status) theo yêu cầu tài liệu đề ra.
        """
        best = search_result['bestScore']
        margin = search_result['margin']
        emp = search_result['bestEmployeeCode']

        # Điểm giống cao vượt ngưỡng VÀ khoảng cách an toàn với Top-2 lớn
        if best >= self.cfg.recognized_threshold and margin >= self.cfg.margin_threshold:
            return 'Recognized', emp, None

        # Điểm mấp mé ở vùng tin cậy thấp hoặc hệ thống bị mơ hồ giữa 2 người (margin thấp)
        if best >= self.cfg.low_confidence_threshold:
            return 'LowConfidence', None, emp

        # Các trường hợp còn lại mặc định là người lạ bên ngoài
        return 'Unknown', None, None