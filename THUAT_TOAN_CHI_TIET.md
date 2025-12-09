# THUẬT TOÁN CHI TIẾT - HỆ THỐNG PHÁT HIỆN VI BIỂU CẢM KHUÔN MẶT DAXLA

---

## 📊 TỔNG QUAN THUẬT TOÁN

### Pipeline Xử Lý Chính
```
Input Video → Face Detection → Feature Extraction → Classification → Output Result
     ↓              ↓                ↓                  ↓              ↓
  Webcam      Haar Cascade      Pixel Features    Random Forest    Truth/Lie
```

---

## 🎯 1. THUẬT TOÁN PHÁT HIỆN KHUÔN MẶT

### Haar Cascade Classifier

#### Nguyên Lý Hoạt Động
```python
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(
    gray,                # Ảnh grayscale
    scaleFactor=1.15,    # Tỷ lệ thu nhỏ mỗi lần quét
    minNeighbors=3,      # Số lượng neighbor tối thiểu
    minSize=(40, 40)     # Kích thước khuôn mặt tối thiểu
)
```

#### Chi Tiết Thuật Toán
**Bước 1: Haar Features**
- Sử dụng các pattern hình chữ nhật để phát hiện đặc trưng
- Tính toán sự khác biệt cường độ sáng giữa các vùng
- Ví dụ: Vùng mắt thường tối hơn vùng má

**Bước 2: Integral Image**
```
Tính toán nhanh tổng pixel trong hình chữ nhật:
sum(x,y) = I(x,y) + sum(x-1,y) + sum(x,y-1) - sum(x-1,y-1)
```

**Bước 3: AdaBoost Learning**
- Kết hợp nhiều weak classifier thành strong classifier
- Chọn các Haar features quan trọng nhất
- Tạo cascade structure để tăng tốc độ

**Bước 4: Multi-scale Detection**
```python
# Quét ảnh ở nhiều kích thước khác nhau
for scale in [1.0, 1.15, 1.32, ...]:
    resized_image = resize(image, scale)
    detect_faces(resized_image)
```

#### Tối Ưu Hóa
```python
# Chỉ lấy khuôn mặt lớn nhất để tránh false detection
largest_face = max(faces, key=lambda f: f[2] * f[3])
```

---

## 🧠 2. THUẬT TOÁN RANDOM FOREST

### Cấu Trúc Tổng Thể

#### Hyperparameters
```python
RandomForestClassifier(
    n_estimators=50,      # 50 cây quyết định
    max_depth=10,         # Độ sâu tối đa 10 levels
    min_samples_split=10, # Tối thiểu 10 mẫu để split node
    min_samples_leaf=5,   # Tối thiểu 5 mẫu ở leaf node
    random_state=42,      # Seed cho reproducibility
    n_jobs=-1            # Sử dụng tất cả CPU cores
)
```

### Chi Tiết Thuật Toán

#### Bước 1: Bootstrap Sampling
```python
# Tạo n_estimators datasets con từ training data
for i in range(50):
    bootstrap_sample = random_sample_with_replacement(X_train, len(X_train))
    trees[i] = build_tree(bootstrap_sample)
```

#### Bước 2: Feature Randomness
```python
# Tại mỗi node, chỉ xem xét sqrt(n_features) features ngẫu nhiên
n_features_per_split = int(sqrt(2304))  # sqrt(48*48) ≈ 48 features
selected_features = random.choice(all_features, n_features_per_split)
```

#### Bước 3: Decision Tree Construction
```
Node Splitting Criteria:
├── Gini Impurity: Gini = 1 - Σ(p_i²)
├── Information Gain: IG = H(parent) - Σ(w_i * H(child_i))
└── Best Split: argmax(Information_Gain)

Stopping Conditions:
├── max_depth = 10
├── min_samples_split = 10
├── min_samples_leaf = 5
└── Pure node (Gini = 0)
```

#### Bước 4: Prediction Aggregation
```python
def predict(X):
    predictions = []
    for tree in trees:
        pred = tree.predict(X)
        predictions.append(pred)
    
    # Voting cho classification
    final_prediction = majority_vote(predictions)
    
    # Probability từ tỷ lệ votes
    probability = count(predictions == final_prediction) / len(trees)
    
    return final_prediction, probability
```

### Ưu Điểm Random Forest
1. **Giảm Overfitting**: Bootstrap + Feature randomness
2. **Robust**: Ít nhạy cảm với noise và outliers
3. **Fast Inference**: Parallel prediction trên nhiều trees
4. **Feature Importance**: Đánh giá tầm quan trọng của từng pixel

---

## 🖼️ 3. THUẬT TOÁN XỬ LÝ ẢNH

### Data Preprocessing Pipeline

#### Bước 1: Image Loading & Conversion
```python
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
img_resized = cv2.resize(img, (48, 48))           # Resize to 48x48
```

#### Bước 2: Data Augmentation
```python
def augment_image(img):
    augmented = []
    
    # 1. Original image
    augmented.append(img)
    
    # 2. Horizontal flip (mirror effect)
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    # 3. Rotation (5 degrees)
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    augmented.append(rotated)
    
    return augmented  # 3x data increase
```

#### Bước 3: Feature Extraction
```python
# Flatten 2D image to 1D feature vector
feature_vector = img_resized.flatten()  # 48x48 = 2304 features

# Normalization
normalized_features = feature_vector.astype('float32') / 255.0
```

### Geometric Transformations

#### Rotation Matrix
```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]

For θ = 5°:
R(5°) = [0.996  -0.087]
        [0.087   0.996]
```

#### Affine Transformation
```python
# Warp image using transformation matrix
cv2.warpAffine(src, M, (width, height))
# M: 2x3 transformation matrix
# Preserves parallel lines and ratios
```

---

## ⚡ 4. THUẬT TOÁN TỐI ƯU HÓA REAL-TIME

### Frame Processing Optimization

#### Temporal Sampling
```python
# Xử lý mỗi 2 frames để tăng tốc độ
if self.frame_count % 2 == 0:
    process_frame(frame)
else:
    skip_frame()
```

#### Face Tracking
```python
# Cache tọa độ khuôn mặt để tránh detect lại
if faces_detected:
    self.last_face_coords = (x, y, w, h)
else:
    # Sử dụng tọa độ cũ nếu không detect được
    use_cached_coordinates()
```

### Confidence Balancing Algorithm

#### Logic Cân Bằng
```python
def balance_prediction(prediction, probabilities):
    # Giảm false positive cho "Lie" class
    if prediction == 1 and probabilities[1] < 0.75:
        if probabilities[0] > 0.3:
            # Chuyển về "Truth" nếu confidence không đủ cao
            prediction = 0
            confidence = probabilities[0]
    
    return prediction, confidence
```

#### Threshold Strategy
```
Confidence Thresholds:
├── Minimum Detection: 0.55 (55%)
├── Lie Confirmation: 0.75 (75%)
├── Truth Fallback: 0.30 (30%)
└── High Confidence: 0.85+ (85%+)
```

---

## 📈 5. THUẬT TOÁN ĐÁNH GIÁ HIỆU SUẤT

### Confusion Matrix Calculation

#### Metrics Computation
```python
# True/False Positives & Negatives
TP = sum((y_true == 1) & (y_pred == 1))  # Correctly predicted Lie
TN = sum((y_true == 0) & (y_pred == 0))  # Correctly predicted Truth
FP = sum((y_true == 0) & (y_pred == 1))  # False Lie detection
FN = sum((y_true == 1) & (y_pred == 0))  # Missed Lie detection

# Performance Metrics
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### Cross-Validation Strategy
```python
# Stratified split để cân bằng classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Đảm bảo tỷ lệ Truth/Lie giống nhau
)
```

---

## 🔄 6. THUẬT TOÁN STREAMING & WEB

### Video Streaming Algorithm

#### MJPEG Streaming
```python
def generate_frames():
    while True:
        frame, results = detector.get_frame()
        
        # JPEG compression
        ret, jpeg = cv2.imencode('.jpg', frame, 
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # HTTP multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               jpeg.tobytes() + b'\r\n\r\n')
        
        time.sleep(0.033)  # ~30 FPS
```

#### Asynchronous Detection Updates
```javascript
// Client-side polling for results
setInterval(function() {
    $.getJSON('/detections', function(data) {
        updateUI(data);
    });
}, 500);  // Update every 500ms
```

### Memory Management
```python
# Efficient memory usage
frame_buffer = collections.deque(maxlen=5)  # Keep only 5 recent frames
result_cache = {}  # Cache recent predictions

# Garbage collection for long-running sessions
if frame_count % 1000 == 0:
    gc.collect()
```

---

## 🎯 7. THUẬT TOÁN PHÂN LOẠI BIỂU CẢM

### Emotion-to-Class Mapping

#### Binary Classification Strategy
```python
# Truth Class (Label 0)
truth_emotions = ['happy', 'neutral', 'surprise']
# Reasoning: Positive/neutral emotions indicate honesty

# Lie Class (Label 1)  
lie_emotions = ['angry', 'sad', 'fear', 'disgust']
# Reasoning: Negative emotions may indicate deception
```

#### Feature Space Analysis
```
48x48 Grayscale Image → 2304-dimensional feature space

Key Facial Regions:
├── Eyes: pixels [10:20, 15:35] → Micro-expressions
├── Mouth: pixels [25:35, 15:35] → Smile/frown detection  
├── Eyebrows: pixels [5:15, 10:40] → Tension indicators
└── Cheeks: pixels [15:30, 5:15, 35:45] → Muscle movement
```

### Decision Boundary Optimization
```python
# Random Forest creates non-linear decision boundaries
# Each tree contributes to final decision surface
# Ensemble voting smooths decision boundaries
# Reduces overfitting to specific facial features
```

---

## 🚀 8. THUẬT TOÁN DEPLOYMENT

### Model Serialization
```python
# Save trained model
with open('micro_model_simple.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load for inference
with open('micro_model_simple.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Production Optimizations
```python
# Model loading optimization
@lru_cache(maxsize=1)
def load_model():
    return pickle.load(open('micro_model_simple.pkl', 'rb'))

# Batch prediction for multiple faces
def batch_predict(face_regions):
    features = [preprocess(face) for face in face_regions]
    return model.predict_proba(np.array(features))
```

---

## 📊 9. PHÂN TÍCH COMPLEXITY

### Time Complexity
```
Face Detection: O(n * m * k)  # n=scales, m=positions, k=features
Feature Extraction: O(1)      # Fixed 48x48 → 2304
Random Forest: O(log d * t)   # d=depth, t=trees
Total per frame: O(n * m * k + log d * t)
```

### Space Complexity
```
Model Storage: O(t * d * f)   # trees * depth * features
Runtime Memory: O(w * h * c)  # width * height * channels
Feature Vector: O(2304)       # Fixed size
```

### Performance Benchmarks
```
Typical Performance:
├── Face Detection: ~10-15ms
├── Feature Extraction: ~1-2ms  
├── ML Prediction: ~2-3ms
├── Total Latency: ~15-20ms
└── Throughput: ~50-60 FPS
```

---

## 🔧 10. THUẬT TOÁN ERROR HANDLING

### Robust Prediction Pipeline
```python
def safe_predict(face_roi):
    try:
        # Preprocessing validation
        if face_roi is None or face_roi.size == 0:
            return None, 0.5
            
        # Size validation
        if min(face_roi.shape) < 20:
            return None, 0.5
            
        # Model prediction
        prediction, confidence = model_predict(face_roi)
        
        # Confidence validation
        if confidence < 0.55:
            return None, 0.5
            
        return prediction, confidence
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, 0.5
```

### Fallback Mechanisms
```python
# Graceful degradation
if not MODEL_AVAILABLE:
    return random_baseline_prediction()

if face_detection_fails():
    use_previous_face_coordinates()

if prediction_confidence_low():
    return_neutral_result()
```

---

## 🎯 KẾT LUẬN THUẬT TOÁN

### Điểm Mạnh
1. **Hiệu Quả**: Random Forest cân bằng tốt giữa accuracy và speed
2. **Robust**: Ít bị overfitting nhờ ensemble method
3. **Real-time**: Tối ưu hóa cho xử lý video streaming
4. **Scalable**: Dễ dàng thêm features hoặc classes mới

### Điểm Cần Cải Thiện
1. **Feature Engineering**: Có thể sử dụng deep features thay vì raw pixels
2. **Temporal Modeling**: Thêm phân tích chuỗi thời gian
3. **Multi-modal**: Kết hợp với audio hoặc physiological signals
4. **Personalization**: Adapt model cho từng người dùng

### Hướng Phát Triển
1. **Deep Learning**: CNN/ResNet cho feature extraction
2. **Attention Mechanism**: Focus vào vùng quan trọng của khuôn mặt
3. **Sequence Modeling**: LSTM/Transformer cho temporal analysis
4. **Multi-task Learning**: Đồng thời detect emotion và deception