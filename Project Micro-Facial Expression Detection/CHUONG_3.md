# CHƯƠNG III: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG

## 3.1. Kiến Trúc Hệ Thống

Hệ thống phát hiện vi biểu cảm khuôn mặt để nhận diện nói dối/nói thật được triển khai trên nền tảng Flask, sử dụng mô hình Random Forest Classifier để phân loại vi biểu cảm từ khuôn mặt người dùng thời gian thực. Kiến trúc hệ thống chia thành các thành phần chính sau:

• **Frontend (Giao diện người dùng)**: Được xây dựng bằng HTML, CSS, JavaScript với Bootstrap framework để tạo giao diện tương tác real-time. Giao diện bao gồm các chức năng chính: hiển thị video stream từ webcam, kết quả phân loại Truth/Lie với độ tin cậy, và thống kê phiên làm việc.

• **Backend (Xử lý dữ liệu và mô hình)**: Flask được sử dụng để điều phối dữ liệu giữa frontend và mô hình machine learning. Backend sẽ nhận video stream từ webcam, xử lý qua mô hình phát hiện vi biểu cảm và trả kết quả phân loại nhị phân với độ tin cậy.

• **Mô hình Random Forest**: Mô hình được huấn luyện để phân loại vi biểu cảm thành hai nhóm: Truth (happy, neutral, surprise) và Lie (angry, sad, fear, disgust) từ ảnh khuôn mặt grayscale 48x48 pixels.

• **Computer Vision Module**: Sử dụng OpenCV và Haar Cascade Classifier để phát hiện khuôn mặt real-time, tiền xử lý ảnh và tối ưu hóa hiệu suất detection.

## 3.2. Mô Tả Các Thành Phần Hệ Thống

### Thu Thập Dữ Liệu Real-time
Dữ liệu khuôn mặt được thu thập từ webcam người dùng thông qua OpenCV VideoCapture. Hệ thống được cấu hình với độ phân giải 640x480 và xử lý mỗi 2 frames để tối ưu hiệu suất:

```python
self.cap = cv2.VideoCapture(0)
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Phát Hiện Khuôn Mặt
Sử dụng Haar Cascade Classifier để phát hiện khuôn mặt với các tham số được tối ưu:
- scaleFactor=1.15: Tỷ lệ thu nhỏ ảnh qua mỗi scale
- minNeighbors=3: Số lượng neighbors tối thiểu để xác nhận detection
- minSize=(40,40): Kích thước khuôn mặt tối thiểu

### Phân Loại Vi Biểu Cảm
Mô hình Random Forest nhận đầu vào là ảnh grayscale 48x48 được flatten thành vector 2304 chiều. Quá trình prediction bao gồm:
1. Resize ảnh khuôn mặt về 48x48
2. Flatten và normalize về [0,1]
3. Predict với Random Forest
4. Áp dụng logic cân bằng để giảm false positive

### Hiển Thị Kết Quả Real-time
Kết quả được hiển thị trực tiếp trên video stream với:
- Khung màu xanh (Truth) hoặc đỏ (Lie) quanh khuôn mặt
- Label và độ tin cậy hiển thị trên khung
- Cập nhật thống kê phiên làm việc theo thời gian thực

## 3.3. Xử Lý Dữ Liệu Và Huấn Luyện Mô Hình

### Chuẩn Bị Dữ Liệu
Dữ liệu từ FER-2013 được tái tổ chức thành cấu trúc phù hợp:
```
data/micro/train/
├── truth/          # Happy, Neutral, Surprise
└── lie/            # Angry, Sad, Fear, Disgust
```

### Data Augmentation
Áp dụng các kỹ thuật tăng cường dữ liệu:
```python
def augment_image(img):
    augmented = []
    augmented.append(img)                    # Original
    augmented.append(cv2.flip(img, 1))       # Horizontal flip
    
    # Rotation 5 degrees
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    augmented.append(rotated)
    
    return augmented
```

### Huấn Luyện Mô Hình
Random Forest được cấu hình để tránh overfitting:
```python
model = RandomForestClassifier(
    n_estimators=50,      # Số cây quyết định
    max_depth=10,         # Độ sâu tối đa
    min_samples_split=10, # Min samples để split
    min_samples_leaf=5,   # Min samples tại leaf
    random_state=42
)
```

## 3.4. Giao Diện Ứng Dụng Web

Giao diện được thiết kế với Bootstrap framework, bao gồm:

### Layout Chính
- **Video Container**: Hiển thị live stream từ webcam với detection results
- **Results Panel**: Hiển thị trạng thái detection và thống kê phiên

### Tính Năng Giao Diện
• **Live Video Feed**: Stream video real-time với overlay detection results
• **Detection Status**: Hiển thị label (Truth/Lie) và confidence score
• **Session Statistics**: Progress bars cho tỷ lệ Truth/Lie trong phiên
• **Responsive Design**: Tương thích với các kích thước màn hình khác nhau

### JavaScript Real-time Updates
```javascript
setInterval(function() {
    $.getJSON('/detections', function(data) {
        updateResults(data);
    });
}, 500);
```

## 3.5. Quy Trình Hoạt Động Real-time Detection

### Bước 1: Khởi Tạo Hệ Thống
- Flask server khởi động trên port 5000
- Load mô hình Random Forest đã huấn luyện
- Khởi tạo VideoCapture và Haar Cascade
- Cung cấp giao diện web cho người dùng

### Bước 2: Video Streaming
- Webcam capture frames liên tục
- Chuyển đổi sang grayscale cho face detection
- Stream video qua Flask Response với multipart format

### Bước 3: Face Detection và Preprocessing
- Phát hiện khuôn mặt với detectMultiScale
- Chọn khuôn mặt lớn nhất (loại bỏ false detection)
- Extract face ROI và resize về 48x48

### Bước 4: Prediction và Logic Cân Bằng
- Flatten ảnh thành vector và normalize
- Predict với Random Forest model
- Áp dụng logic cân bằng:
  ```python
  if prediction == 1 and probabilities[1] < 0.75:
      if probabilities[0] > 0.3:
          prediction = 0
          confidence = probabilities[0]
  ```

### Bước 5: Hiển Thị Kết Quả
- Vẽ bounding box quanh khuôn mặt
- Hiển thị label và confidence score
- Cập nhật detection results cho frontend
- Lưu thống kê phiên làm việc

## 3.6. Tối Ưu Hóa Hiệu Suất Và Độ Chính Xác

### Tối Ưu Hiệu Suất
• **Frame Skipping**: Xử lý mỗi 2 frames thay vì mọi frame
• **Face Caching**: Lưu tọa độ khuôn mặt để tránh detection liên tục
• **Optimized Parameters**: Điều chỉnh scaleFactor và minNeighbors cho tốc độ
• **JPEG Compression**: Sử dụng quality=85 cho video streaming

### Cải Thiện Độ Chính Xác
• **Confidence Threshold**: Chỉ hiển thị kết quả khi confidence > 0.55
• **Prediction Balancing**: Logic cân bằng để giảm false positive cho class "Lie"
• **Largest Face Selection**: Chọn khuôn mặt lớn nhất để tránh noise
• **Data Augmentation**: Tăng cường dữ liệu để cải thiện generalization

### Xử Lý Lỗi và Stability
• **Model Loading Check**: Kiểm tra model availability trước khi predict
• **Exception Handling**: Try-catch cho các operations có thể fail
• **Graceful Degradation**: Hệ thống vẫn hoạt động khi model không available
• **Resource Management**: Proper cleanup cho VideoCapture resources