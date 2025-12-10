# Phát Hiện Vi Biểu Cảm Khuôn Mặt

Hệ thống machine learning phát hiện vi biểu cảm khuôn mặt sử dụng Random Forest để nhận diện nói dối/nói thật.

## Tính Năng

- **Phát hiện thời gian thực**: Xử lý video webcam trực tiếp
- **Phân loại nhị phân**: Phân biệt nói thật vs nói dối
- **Độ chính xác cao**: Thuật toán Random Forest tối ưu
- **Giao diện web**: Dashboard Flask real-time
- **Thống kê**: Phân tích kết quả theo phiên

## Hướng Dẫn Sử Dụng

1. **Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

2. **Chuẩn bị dữ liệu**
- Tạo cấu trúc thư mục `data/micro/train/truth/` và `data/micro/train/lie/`
- Copy ảnh vào `data/micro/train/truth/` (happy, neutral, surprise)
- Copy ảnh vào `data/micro/train/lie/` (angry, sad, fear, disgust)

3. **Huấn luyện mô hình**
```bash
python simple_train.py
```

4. **Chạy ứng dụng**
```bash
python trained_app.py
```

## Kiến Trúc

- **Đầu vào**: Ảnh khuôn mặt grayscale 48x48
- **Mô hình**: Random Forest Classifier
- **Đầu ra**: Phân loại nhị phân (Thật/Dối) với độ tin cậy

## Ứng Dụng

- Kiểm tra an ninh
- Phân tích phỏng vấn
- Đánh giá hành vi
- Tương tác người-máy

## Hiệu Suất

Hệ thống sử dụng phân tích thời gian và ngưỡng tin cậy để cải thiện độ chính xác trong việc phát hiện vi biểu cảm tinh tế cho thấy sự lừa dối.
