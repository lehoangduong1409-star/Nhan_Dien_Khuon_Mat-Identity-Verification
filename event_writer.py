#Tạo file bổ trợ ghi bản ghi sự kiện ĐỂ CHẠY FILE VIEO-RUNNER.PY
# face_recognition_3m/src/event_writer.py
import csv
from pathlib import Path

# Cấu trúc gói dữ liệu sự kiện (Event Schema) bắt buộc của tài liệu
FIELDS = [
    'eventId', 'timestamp', 'camerald', 'distanceMeters',
    'recognitionStatus', 'employeeCode', 'candidateEmployeeCode',
    'bestScore', 'secondBestScore', 'margin',
    'faceBoxWidth', 'faceBoxHeight', 'blurScore', 'brightness',
    'evidenceImagePath', 'note'
]

def append_event_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader() # Ghi hàng tiêu đề cột nếu file mới được tạo lần đầu
        writer.writerow({k: row.get(k) for k in FIELDS})