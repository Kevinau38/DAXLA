# CHƯƠNG II: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ ÁP DỤNG

## 2.1. Tổng Quan Về Machine Learning Và Ứng Dụng

Machine Learning (ML) là một nhánh của trí tuệ nhân tạo (AI), cho phép máy tính học từ dữ liệu đầu vào để thực hiện phân tích, dự đoán và đưa ra quyết định mà không cần lập trình rõ ràng từng bước. Trong đề tài này, machine learning đóng vai trò trung tâm trong việc huấn luyện mô hình phát hiện vi biểu cảm khuôn mặt để phân biệt trạng thái nói thật và nói dối.

Khi áp dụng vào phát hiện vi biểu cảm, các mô hình machine learning sẽ được huấn luyện từ các hình ảnh gắn nhãn cảm xúc cụ thể được phân loại thành hai nhóm: Truth (happy, neutral, surprise) và Lie (angry, sad, fear, disgust) để máy học cách phân biệt từng trạng thái qua đặc trưng khuôn mặt.

Một số lợi ích khi sử dụng machine learning:
• Tự động hóa quy trình phân loại vi biểu cảm
• Khả năng xử lý dữ liệu hình ảnh quy mô lớn
• Nâng cao độ chính xác trong phân biệt nói thật/nói dối
• Phát hiện và phân loại hành vi nhanh chóng trong thời gian thực

## 2.2. Thuật Toán Random Forest Classifier

Trong nghiên cứu này, mô hình Random Forest Classifier được sử dụng để xử lý và phân loại vi biểu cảm khuôn mặt. Random Forest nổi bật với khả năng xử lý dữ liệu phức tạp, tránh overfitting và đạt độ chính xác cao trong các tác vụ phân loại.

Các đặc điểm chính của Random Forest:
• **Ensemble Learning**: Kết hợp nhiều cây quyết định (Decision Trees) để đưa ra dự đoán cuối cùng
• **Bootstrap Aggregating**: Sử dụng kỹ thuật bagging để tạo ra các tập con dữ liệu khác nhau
• **Feature Randomness**: Chọn ngẫu nhiên một tập con các đặc trưng tại mỗi node
• **Voting Mechanism**: Kết quả cuối cùng được quyết định bằng cách bỏ phiếu từ tất cả các cây

**Cấu hình mô hình trong dự án:**
- n_estimators=50: Số lượng cây quyết định
- max_depth=10: Độ sâu tối đa của mỗi cây
- min_samples_split=10: Số mẫu tối thiểu để chia node
- min_samples_leaf=5: Số mẫu tối thiểu tại mỗi lá

Mô hình được huấn luyện trên tập dữ liệu FER-2013 với hơn 35.000 ảnh khuôn mặt được tái phân loại, đạt hiệu suất tốt trong phân biệt nói thật/nói dối thời gian thực.

## 2.3. Giới Thiệu Về Flask và OpenCV

**Flask** là framework web nhẹ của Python, đóng vai trò là backend xử lý và hiển thị luồng video camera kết hợp với kết quả phân loại vi biểu cảm đầu ra. Flask cung cấp:
• Routing system đơn giản cho các endpoint API
• Template engine để render giao diện HTML
• Session management cho tracking kết quả
• Real-time streaming với multipart response

**OpenCV** là thư viện xử lý ảnh mã nguồn mở, được sử dụng để:
• Thu thập ảnh từ webcam với VideoCapture
• Phát hiện khuôn mặt sử dụng Haar Cascade Classifier
• Tiền xử lý ảnh (resize, grayscale conversion)
• Xử lý real-time video stream với tối ưu hóa hiệu suất

**Tích hợp Flask + OpenCV:**
```python
# Video streaming endpoint
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Face detection với OpenCV
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3)
```

Hai công cụ này kết hợp giúp xây dựng hệ thống trực quan, đơn giản và hiệu quả khi triển khai thực tế trên môi trường web.

## 2.4. Kỹ Thuật Data Augmentation

Data Augmentation là kỹ thuật tăng cường dữ liệu nhằm mở rộng tập dữ liệu huấn luyện mà không cần thu thập thêm dữ liệu mới. Trong dự án phát hiện vi biểu cảm, kỹ thuật này đặc biệt quan trọng để cải thiện độ chính xác và tránh overfitting.

**Các kỹ thuật được áp dụng:**

• **Horizontal Flip**: Lật ảnh theo chiều ngang
```python
augmented.append(cv2.flip(img, 1))
```

• **Rotation**: Xoay ảnh với góc nhỏ (5 độ)
```python
M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
rotated = cv2.warpAffine(img, M, (cols, rows))
```

• **Normalization**: Chuẩn hóa pixel values về khoảng [0,1]
```python
X = X.astype('float32') / 255.0
```

**Lợi ích của Data Augmentation:**
• Tăng kích thước tập dữ liệu từ ~400 ảnh lên ~1200 ảnh
• Cải thiện khả năng tổng quát hóa của mô hình
• Giảm thiểu overfitting trên tập huấn luyện nhỏ
• Tăng độ robust của mô hình với các biến thể ảnh đầu vào

## 2.5. Quy Trình Xây Dựng Mô Hình Machine Learning

Trong dự án phát hiện vi biểu cảm, quy trình xây dựng mô hình tuân theo các bước chuẩn của machine learning với một số điều chỉnh phù hợp với bài toán phân loại nhị phân.

**7 bước chính trong quy trình:**

• **Bước 1: Thu thập dữ liệu** – Sử dụng tập FER-2013 với 35.000+ ảnh khuôn mặt được gắn nhãn 7 cảm xúc cơ bản

• **Bước 2: Tiền xử lý và tái phân loại dữ liệu** – Chuyển đổi từ 7 classes thành 2 classes (Truth/Lie), resize ảnh về 48x48 grayscale

• **Bước 3: Data Augmentation** – Áp dụng flip, rotation để tăng cường dữ liệu và cân bằng tập huấn luyện

• **Bước 4: Lựa chọn mô hình** – Chọn Random Forest Classifier phù hợp với dữ liệu tabular (flattened images)

• **Bước 5: Huấn luyện mô hình** – Train với hyperparameters được tối ưu để tránh overfitting

• **Bước 6: Đánh giá mô hình** – Kiểm tra accuracy trên tập train/test, phân tích confusion matrix

• **Bước 7: Triển khai real-time** – Tích hợp mô hình vào Flask app với OpenCV để detection thời gian thực

## 2.6. Ứng Dụng Của Phát Hiện Vi Biểu Cảm Trong Đời Sống

Phát hiện vi biểu cảm khuôn mặt có nhiều ứng dụng thực tiễn trong các lĩnh vực khác nhau:

**An ninh và Pháp luật:**
• Hỗ trợ thẩm vấn và điều tra
• Phát hiện hành vi đáng ngờ tại các khu vực nhạy cảm
• Kiểm tra tính trung thực trong lời khai

**Tuyển dụng và Nhân sự:**
• Đánh giá ứng viên trong quá trình phỏng vấn
• Phân tích mức độ tự tin và chân thành
• Hỗ trợ quyết định tuyển dụng

**Y tế và Tâm lý:**
• Chẩn đoán các rối loạn tâm lý
• Theo dõi trạng thái cảm xúc bệnh nhân
• Hỗ trợ liệu pháp tâm lý

**Giáo dục:**
• Đánh giá mức độ hiểu bài của học sinh
• Phát hiện gian lận trong thi cử
• Cải thiện phương pháp giảng dạy

**Thương mại:**
• Phân tích phản ứng khách hàng với sản phẩm
• Tối ưu hóa trải nghiệm mua sắm
• Đánh giá hiệu quả quảng cáo

## 2.7. Phân Tích Vi Biểu Cảm Và Tâm Lý Học Hành Vi

Vi biểu cảm (Micro-expressions) là những biểu hiện cảm xúc thoáng qua, thường kéo dài từ 1/25 đến 1/5 giây, xuất hiện khi con người cố gắng che giấu cảm xúc thật sự. Khái niệm này được phát triển bởi nhà tâm lý học Paul Ekman.

**Đặc điểm của vi biểu cảm:**
• **Thời gian ngắn**: Xuất hiện và biến mất rất nhanh
• **Không tự nguyện**: Khó kiểm soát bằng ý thức
• **Phản ánh cảm xúc thật**: Tiết lộ cảm xúc đang cố che giấu
• **Phổ quát**: Giống nhau ở mọi nền văn hóa

**Cơ sở tâm lý học:**

**Lý thuyết cảm xúc cơ bản của Ekman:**
- 7 cảm xúc cơ bản: Happy, Sad, Angry, Fear, Surprise, Disgust, Contempt
- Mỗi cảm xúc có biểu hiện đặc trưng trên khuôn mặt
- Vi biểu cảm xuất hiện khi có xung đột giữa cảm xúc thật và cảm xúc muốn thể hiện

**Ứng dụng trong phát hiện nói dối:**
• **Truth indicators**: Happy, Neutral, Surprise - thể hiện trạng thái thoải mái, tự nhiên
• **Deception indicators**: Angry, Sad, Fear, Disgust - thể hiện căng thẳng, bất tự nhiên

**Hạn chế và thách thức:**
• Độ chính xác phụ thuộc vào chất lượng dữ liệu huấn luyện
• Sự khác biệt cá nhân trong biểu hiện cảm xúc
• Ảnh hưởng của yếu tố văn hóa và bối cảnh
• Cần kết hợp với các chỉ số khác để đạt độ tin cậy cao