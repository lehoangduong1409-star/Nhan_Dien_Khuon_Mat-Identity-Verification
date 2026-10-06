# 📑 MÔ-ĐUN NHẬN DIỆN KHUÔN MẶT Ở KHOẢNG CÁCH KHÔNG GIAN ĐẠT 3M
> **DATN-MH02** | Mã nguồn hệ thống kiểm soát và nhận diện nhân sự thời gian thực.

---

## 🛠️ 1. Tổng quan Công nghệ & Tính năng Hệ thống
Mô-đun được phát triển nhằm mục đích nhận diện và định danh khuôn mặt nhân viên ở khoảng cách xa (lên tới 3m) với độ chính xác cao nhờ tích hợp các công nghệ AI tiên tiến:
*   **Face Detection:** Sử dụng mô hình RetinaFace (thư viện InsightFace) trích xuất khuôn mặt tối ưu nhất trong khung hình.
*   **Face Recognition:** Trích xuất vector đặc trưng (Embedding) 512-chiều và so khớp độ tương đồng Cosine (Cosine Similarity).
*   **Quality Filter:** Bộ lọc ảnh thông minh tự động đo độ mờ (Blur) bằng thuật toán Laplacian và đo độ sáng (Brightness) để từ chối ảnh kém chất lượng trước khi đưa vào nhận diện.
*   **Web API Gateway:** Cung cấp cổng giao tiếp FastAPI chuẩn OpenAPI 3.1 phục vụ tích hợp hệ thống ngoại vi và Dashboard.

---

## 📂 2. Cấu trúc Thư mục Dự án (Project Architecture)
Để hệ thống vận hành ổn định, cấu trúc thư mục của đồ án được tổ chức quy chuẩn như sau:

```text
face_recognition_3m/
├── .venv/                      # Môi trường ảo chứa các thư viện Python cách ly
├── data/                       # Thư mục lưu trữ dữ liệu đầu vào
│   └── employees/              # Kho ảnh gốc nhân viên (Chia theo mã EMPXXX)
├── evidence/                   # Lưu trữ bằng chứng khi nhận diện thành công
│   ├── crops/                  # Ảnh cắt cận cảnh khuôn mặt (phục vụ đối chứng)
│   └── frames/                 # Ảnh toàn cảnh chụp từ camera/video gốc
├── gallery/                    # Cơ sở dữ liệu mẫu dạng nén
│   └── face_gallery.npz        # File chứa các vector đặc trưng đã đóng gói
├── reports/                    # Thư mục xuất báo cáo kỹ thuật
│   └── recognition_results.csv # Nhật ký log nhận diện thực địa (Mã số, thời gian, score)
├── src/                        # Thư mục chứa toàn bộ mã nguồn chính
│   ├── api.py                  # Dịch vụ Backend Web API (FastAPI)
│   ├── config.py               # File cấu hình tham số hệ thống (Ngưỡng, kích thước mặt)
│   ├── face_engine.py          # Lớp xử lý AI trích xuất khuôn mặt (InsightFace)
│   ├── quality.py              # Bộ thuật toán kiểm tra chất lượng ảnh cận cảnh
│   ├── recognizer.py           # Bộ logic so khớp và ra quyết định định danh
│   ├── gallery_builder.py      # Script quét ảnh gốc để đóng gói kho mẫu .npz
│   ├── test_recognizer.py      # Script kiểm thử nhận diện trên ảnh tĩnh đơn lẻ
│   └── video_runner.py         # Script chạy thực nghiệm luồng Video/Camera trực tiếp
├── README.md                   # Tài liệu hướng dẫn triển khai (File này)
└── requirements.txt            # Danh sách các thư viện đóng gói để cài tự động
#------------------------------------------------------------------------------------------------------
CÁC BƯỚC TRIỂN KHAI DỰ ÁN 


# Bước 1: Khởi tạo môi trường ảo độc lập để tránh xung đột thư viện máy
python -m venv .venv

# Bước 2: Kích hoạt môi trường ảo vừa tạo
.venv\Scripts\activate

# Bước 3: Dùng 1 lệnh Terminal duy nhất để tải tự động toàn bộ thư viện gộp
pip install -r requirements.txt