
from dataclasses import dataclass

@dataclass
class RecognitionConfig:
    recognized_threshold: float = 0.70     # Ngưỡng nhận diện đúng
    low_confidence_threshold: float = 0.55   # Ngưỡng nghi vấn
    margin_threshold: float = 0.06           # Độ lệch an toàn giữa Top-1 và Top-2
    min_face_width: int = 80                 # Kích thước mặt tối thiểu tại 3m (pixel)
    min_face_height: int = 80
    blur_threshold: float = 80.0             # Ngưỡng lọc ảnh mờ
    min_brightness: float = 45.0             # Ngưỡng lọc ảnh quá tối
    max_brightness: float = 220.0            # Ngưỡng lọc ảnh lóa sáng
    frame_sample_fps: int = 5                # Số khung hình xử lý/giây

CFG = RecognitionConfig()