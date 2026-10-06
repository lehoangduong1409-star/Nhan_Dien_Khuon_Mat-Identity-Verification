
# src/quality.py
import cv2
import numpy as np
def blur_score_bgr(image_bgr) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
def brightness_bgr(image_bgr) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
def is_quality_ok(face_bgr, cfg):
    if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] == 0 or face_bgr.shape[1] == 0:
        return False, 0.0, 0.0# thêm phân này dẻ k bị báo lỗi
    blur = blur_score_bgr(face_bgr)
    bright = brightness_bgr(face_bgr)
    ok = blur >= cfg.blur_threshold
    ok = ok and cfg.min_brightness <= bright <= cfg.max_brightness
    return ok, blur, bright