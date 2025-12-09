# HỆ THỐNG PHÁT HIỆN VI BIỂU CẢM KHUÔN MẶT - DAXLA
## Hướng Dẫn Chi Tiết Về Cách Vận Hành

---

## 📋 TỔNG QUAN HỆ THỐNG

### Mục Đích
Hệ thống DAXLA được thiết kế để phát hiện vi biểu cảm khuôn mặt nhằm nhận diện sự lừa dối thông qua phân tích biểu cảm trong thời gian thực. Hệ thống sử dụng thuật toán Random Forest để phân loại nhị phân giữa "Nói thật" và "Nói dối".

### Nguyên Lý Hoạt Động
- **Đầu vào**: Video webcam thời gian thực
- **Xử lý**: Phát hiện khuôn mặt → Trích xuất đặc trưng → Phân loại ML
- **Đầu ra**: Kết quả phân loại với độ tin cậy

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### Cấu Trúc Thư Mục
```
DAXLA/
├── Project Micro-Facial Expression Detection/
│   ├── data/micro/
│   │   ├── train/
│   │   │   ├── truth/ (happy, neutral, surprise)
│   │   │   └── lie/ (angry, sad, fear, disgust)
│   │   └── test/ (cấu trúc tương tự)
│   ├── templates/micro_index.html
│   ├── simple_train.py (Huấn luyện mô hình)
│   ├── trained_app.py (Ứng dụng web)
│   ├── evaluate_simple.py (Đánh giá mô hình)
│   ├── micro_model_simple.pkl (Mô hình đã huấn luyện)
│   ├── haarcascade_frontalface_default.xml (Phát hiện khuôn mặt)
│   └── requirements.txt
└── README.md
```

### Thành Phần Chính

#### 1. **Module Huấn Luyện (simple_train.py)**
- **Chức năng**: Huấn luyện mô hình Random Forest
- **Đầu vào**: Ảnh khuôn mặt 48x48 grayscale
- **Xử lý**: Data augmentation + Feature extraction
- **Đầu ra**: Mô hình .pkl

#### 2. **Module Ứng Dụng Web (trained_app.py)**
- **Chức năng**: Giao diện web real-time
- **Framework**: Flask
- **Tính năng**: Video streaming + Detection + Statistics

#### 3. **Module Đánh Giá (evaluate_simple.py)**
- **Chức năng**: Đánh giá hiệu suất mô hình
- **Đầu ra**: Confusion matrix + Metrics

---

## 🔧 CHI TIẾT CÁCH VẬN HÀNH

### BƯỚC 1: CÀI ĐẶT MÔI TRƯỜNG

```bash
# Cài đặt thư viện
pip install -r requirements.txt
```

**Thư viện cần thiết:**
- Flask==2.3.3: Web framework
- opencv-python==4.8.1.78: Computer vision
- numpy==1.24.3: Tính toán số học
- scikit-learn==1.3.0: Machine learning
- matplotlib==3.7.2: Visualization

### BƯỚC 2: CHUẨN BỊ DỮ LIỆU

#### Cấu Trúc Dữ Liệu
```
data/micro/train/
├── truth/ (Biểu cảm thật)
│   ├── happy_*.jpg (Vui vẻ)
│   ├── neutral_*.jpg (Trung tính)
│   └── surprise_*.jpg (Ngạc nhiên)
└── lie/ (Biểu cảm dối trá)
    ├── angry_*.jpg (Tức giận)
    ├── sad_*.jpg (Buồn bã)
    ├── fear_*.jpg (Sợ hãi)
    └── disgust_*.jpg (Ghê tởm)
```

#### Phân Loại Biểu Cảm
- **Truth (0)**: Happy, Neutral, Surprise
- **Lie (1)**: Angry, Sad, Fear, Disgust

### BƯỚC 3: HUẤN LUYỆN MÔ HÌNH

```bash
python simple_train.py
```

#### Quy Trình Huấn Luyện Chi Tiết

**3.1. Tải và Tiền Xử Lý Dữ Liệu**
```python
def load_data():
    # Đọc ảnh từ thư mục
    # Resize về 48x48 pixels
    # Chuyển đổi sang grayscale
    # Flatten thành vector 1D (2304 features)
```

**3.2. Data Augmentation**
```python
def augment_image(img):
    # Ảnh gốc
    # Lật ngang (horizontal flip)
    # Xoay nhẹ 5 độ
    # Tăng gấp 3 lần dữ liệu
```

**3.3. Cấu Hình Random Forest**
```python
RandomForestClassifier(
    n_estimators=50,      # 50 cây quyết định
    max_depth=10,         # Độ sâu tối đa 10
    min_samples_split=10, # Tối thiểu 10 mẫu để split
    min_samples_leaf=5,   # Tối thiểu 5 mẫu ở leaf
    random_state=42
)
```

**3.4. Chia Dữ Liệu**
- Training: 80%
- Testing: 20%
- Stratified split để cân bằng classes

### BƯỚC 4: CHẠY ỨNG DỤNG WEB

```bash
python trained_app.py
```

#### Luồng Xử Lý Real-time

**4.1. Khởi Tạo Camera**
```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**4.2. Phát Hiện Khuôn Mặt**
```python
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.15,    # Tỷ lệ thu nhỏ
    minNeighbors=3,      # Số lượng neighbor tối thiểu
    minSize=(40, 40)     # Kích thước khuôn mặt tối thiểu
)
```

**4.3. Trích Xuất và Dự Đoán**
```python
def predict_micro_expression(face_roi):
    # Resize về 48x48
    face_resized = cv2.resize(face_roi, (48, 48))
    # Flatten và normalize
    face_flattened = face_resized.flatten().astype('float32') / 255.0
    # Dự đoán
    prediction = model.predict(face_input)[0]
    probabilities = model.predict_proba(face_input)[0]
    confidence = np.max(probabilities)
```

**4.4. Logic Cân Bằng**
```python
# Giảm false positive cho "Lie"
if prediction == 1 and probabilities[1] < 0.75:
    if probabilities[0] > 0.3:
        prediction = 0  # Chuyển về Truth
        confidence = probabilities[0]
```

### BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH

```bash
python evaluate_simple.py
```

#### Metrics Đánh Giá
- **Confusion Matrix**: Hiển thị chi tiết phân loại
- **Precision**: Độ chính xác cho từng class
- **Recall**: Độ nhạy cho từng class  
- **F1-Score**: Điểm số cân bằng
- **Overall Accuracy**: Độ chính xác tổng thể

---

## 🎯 GIAO DIỆN WEB

### Tính Năng Chính

#### 1. **Live Video Feed**
- Stream webcam real-time
- Hiển thị khung bao quanh khuôn mặt
- Màu sắc: Xanh (Truth), Đỏ (Lie)

#### 2. **Detection Results Panel**
- Trạng thái hiện tại
- Độ tin cậy (%)
- Thống kê phiên làm việc

#### 3. **Session Statistics**
- Progress bar cho Truth/Lie ratio
- Tổng số detections
- Phần trăm cho mỗi loại

### Cập Nhật Real-time
```javascript
setInterval(function() {
    $.getJSON('/detections', function(data) {
        updateResults(data);
    });
}, 500); // Cập nhật mỗi 500ms
```

---

## ⚙️ TỐI ƯU HÓA HIỆU SUẤT

### Tối Ưu Tốc Độ
1. **Frame Processing**: Xử lý mỗi 2 frames
2. **Face Detection**: Tham số tối ưu cho tốc độ
3. **Model Inference**: Cache kết quả gần nhất
4. **Video Streaming**: JPEG compression 85%

### Tối Ưu Độ Chính Xác
1. **Data Augmentation**: Tăng đa dạng dữ liệu
2. **Feature Engineering**: Normalize pixel values
3. **Model Tuning**: Giảm overfitting
4. **Threshold Adjustment**: Cân bằng precision/recall

---

## 🔍 THUẬT TOÁN MACHINE LEARNING

### Random Forest Classifier

#### Ưu Điểm
- **Robust**: Ít bị overfitting
- **Fast**: Inference nhanh
- **Interpretable**: Dễ hiểu và debug
- **No Feature Scaling**: Không cần chuẩn hóa đặc trưng

#### Cách Hoạt Động
1. **Bootstrap Sampling**: Tạo nhiều subset từ training data
2. **Tree Building**: Xây dựng decision tree cho mỗi subset
3. **Feature Randomness**: Random chọn features tại mỗi split
4. **Voting**: Kết hợp kết quả từ tất cả trees

#### Hyperparameters
- `n_estimators=50`: Số lượng cây
- `max_depth=10`: Độ sâu tối đa
- `min_samples_split=10`: Mẫu tối thiểu để split
- `min_samples_leaf=5`: Mẫu tối thiểu ở leaf node

---

## 📊 PHÂN TÍCH DỮ LIỆU

### Thống Kê Dataset
```
Training Data:
├── Truth: ~300 images (after augmentation: ~900)
├── Lie: ~400 images (after augmentation: ~1200)
└── Total: ~2100 augmented samples

Test Data:
├── Truth: ~150 images
├── Lie: ~200 images  
└── Total: ~350 samples
```

### Feature Engineering
- **Input Size**: 48x48 = 2304 features
- **Normalization**: Pixel values / 255.0
- **Data Type**: float32 (memory efficient)

---

## 🚀 DEPLOYMENT & PRODUCTION

### Yêu Cầu Hệ Thống
- **CPU**: Multi-core (tối thiểu dual-core)
- **RAM**: 4GB+ (cho model loading)
- **Camera**: Webcam với resolution 640x480+
- **OS**: Windows/Linux/macOS

### Cấu Hình Production
```python
# Flask production settings
app.run(
    debug=False,        # Tắt debug mode
    host='0.0.0.0',     # Listen trên tất cả interfaces
    port=5000,          # Port mặc định
    threaded=True       # Enable threading
)
```

### Monitoring & Logging
- **Performance Metrics**: FPS, latency, accuracy
- **Error Handling**: Try-catch cho model inference
- **Resource Usage**: CPU, memory monitoring

---

## 🔧 TROUBLESHOOTING

### Lỗi Thường Gặp

#### 1. **Camera không hoạt động**
```python
# Kiểm tra camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
```

#### 2. **Model không load được**
```python
# Kiểm tra file model
if not os.path.exists('micro_model_simple.pkl'):
    print("Run simple_train.py first")
```

#### 3. **Face detection kém**
- Đảm bảo ánh sáng đủ
- Khuôn mặt thẳng với camera
- Khoảng cách phù hợp (50-100cm)

#### 4. **Độ chính xác thấp**
- Tăng dữ liệu training
- Điều chỉnh hyperparameters
- Cải thiện chất lượng ảnh

---

## 📈 HƯỚNG PHÁT TRIỂN

### Cải Tiến Ngắn Hạn
1. **Deep Learning**: Chuyển sang CNN/ResNet
2. **Multi-class**: Phân loại 7 emotions
3. **Temporal Analysis**: Phân tích chuỗi thời gian
4. **Mobile App**: Ứng dụng di động

### Cải Tiến Dài Hạn
1. **Real-time Optimization**: GPU acceleration
2. **Cloud Deployment**: Scalable architecture
3. **Advanced Features**: Eye tracking, micro-gestures
4. **Integration**: API cho các hệ thống khác

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers & Research
- Facial Expression Recognition using Random Forest
- Micro-expression Detection in Real-time
- Computer Vision for Deception Detection

### Libraries Documentation
- OpenCV: https://docs.opencv.org/
- Scikit-learn: https://scikit-learn.org/
- Flask: https://flask.palletsprojects.com/

### Datasets
- FER2013: Facial Expression Recognition
- CK+: Extended Cohn-Kanade Dataset
- JAFFE: Japanese Female Facial Expression

---

## 🎯 KẾT LUẬN

Hệ thống DAXLA cung cấp một giải pháp hoàn chỉnh cho việc phát hiện vi biểu cảm khuôn mặt trong thời gian thực. Với kiến trúc đơn giản nhưng hiệu quả, hệ thống có thể được triển khai trong nhiều ứng dụng thực tế như:

- **An ninh**: Phát hiện hành vi đáng ngờ
- **Phỏng vấn**: Hỗ trợ đánh giá ứng viên  
- **Giáo dục**: Phân tích phản ứng học sinh
- **Y tế**: Đánh giá tâm lý bệnh nhân

Hệ thống được thiết kế với tính mở rộng cao, cho phép dễ dàng cải tiến và tích hợp với các công nghệ mới.