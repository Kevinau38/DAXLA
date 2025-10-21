# CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI

## 1.1. Lý Do Lựa Chọn Đề Tài Nghiên Cứu

Trong thời đại công nghệ 4.0, việc ứng dụng trí tuệ nhân tạo vào phân tích hành vi con người đang trở thành một xu hướng quan trọng. Một trong những lĩnh vực ứng dụng đầy tiềm năng là phát hiện vi biểu cảm khuôn mặt – nơi mà hệ thống có thể phân tích những thay đổi tinh tế trên gương mặt để suy luận tính chân thực trong lời nói.

Vi biểu cảm là những biểu hiện cảm xúc thoáng qua, thường kéo dài chỉ vài phần trăm giây, nhưng có thể tiết lộ cảm xúc thật sự mà con người đang cố gắng che giấu. Khác với nhận diện cảm xúc thông thường, việc phát hiện vi biểu cảm đòi hỏi độ chính xác cao hơn để phân biệt giữa trạng thái nói thật và nói dối thông qua các dấu hiệu như căng thẳng, bất tự nhiên hoặc mâu thuẫn trong biểu cảm.

Đề tài này hướng đến việc kết hợp công nghệ xử lý hình ảnh (Computer Vision), machine learning (Random Forest) và web framework (Flask) để xây dựng một hệ thống thông minh, có khả năng:

• Phát hiện vi biểu cảm khuôn mặt thời gian thực thông qua webcam;
• Phân loại nhị phân giữa trạng thái nói thật (happy, neutral, surprise) và nói dối (angry, sad, fear, disgust);
• Hiển thị kết quả với độ tin cậy và thống kê phiên làm việc;
• Tạo ra một công cụ hỗ trợ trong các lĩnh vực an ninh, phỏng vấn và đánh giá tâm lý.

Bên cạnh tính thực tiễn, đề tài còn có ý nghĩa trong việc ứng dụng công nghệ hiện đại để hỗ trợ các hoạt động kiểm tra tính trung thực, đặc biệt trong bối cảnh nhu cầu về an ninh và đánh giá hành vi ngày càng tăng cao. Tại Việt Nam, hướng nghiên cứu này vẫn còn mới mẻ, do đó việc triển khai đề tài sẽ đóng góp thêm một góc nhìn thực tế về tiềm năng ứng dụng AI trong lĩnh vực phân tích hành vi con người.

## 1.2. Mục Tiêu Đề Tài

Mục tiêu chính của đề tài là xây dựng một hệ thống thông minh có khả năng phát hiện vi biểu cảm khuôn mặt của người dùng thông qua hình ảnh từ webcam và phân loại trạng thái nói thật/nói dối theo thời gian thực.

Cụ thể, hệ thống hướng đến các mục tiêu sau:

• Ứng dụng thuật toán Random Forest để phân loại chính xác vi biểu cảm từ ảnh khuôn mặt grayscale 48x48;
• Sử dụng OpenCV để xử lý hình ảnh real-time và phát hiện khuôn mặt với Haar Cascade;
• Xây dựng giao diện web đơn giản với Flask, cho phép người dùng tương tác trực tiếp qua trình duyệt;
• Tối ưu hóa hiệu suất với data augmentation và logic cân bằng prediction;
• Đáp ứng nhu cầu ứng dụng trong các lĩnh vực an ninh, phỏng vấn tuyển dụng và đánh giá tâm lý.

## 1.3. Đối Tượng Nghiên Cứu

**Nguồn dữ liệu:** Dữ liệu biểu cảm khuôn mặt được thu thập từ bộ dữ liệu công khai FER-2013 – một tập dữ liệu nổi tiếng được sử dụng rộng rãi trong các nghiên cứu về nhận diện cảm xúc. Tập dữ liệu được cung cấp bởi tổ chức Kaggle, bao gồm hơn 35.000 hình ảnh đen trắng 48x48 pixel gắn nhãn theo 7 loại cảm xúc cơ bản: Angry, Disgust, Fear, Happy, Sad, Surprise, và Neutral.

**Phân loại dữ liệu:** Dữ liệu được tái tổ chức thành hai nhóm chính:
- **Truth (Nói thật):** Happy, Neutral, Surprise - những cảm xúc thể hiện trạng thái tự nhiên, thoải mái
- **Lie (Nói dối):** Angry, Sad, Fear, Disgust - những cảm xúc thể hiện căng thẳng, bất tự nhiên

**Mô tả tập dữ liệu:** Mỗi ảnh trong tập FER-2013 là ảnh khuôn mặt đơn lẻ với định dạng grayscale, được chuẩn hóa để có kích thước và định dạng đồng nhất. Dữ liệu được chia thành các tập huấn luyện và kiểm tra, phục vụ cho việc đào tạo và đánh giá mô hình Random Forest.

**Xử lý dữ liệu:** Do ảnh có độ phân giải thấp và chứa nhiều biểu cảm khác nhau, quá trình tiền xử lý như chuẩn hóa, data augmentation (flip, rotation) và cân bằng dữ liệu là cần thiết để cải thiện độ chính xác của mô hình.

## 1.4. Phương Pháp Nghiên Cứu

**Phương Pháp Thu Thập Dữ Liệu**

Toàn bộ dữ liệu được thu thập từ tập FER-2013, công khai trên nền tảng Kaggle. Dữ liệu được tiền xử lý và huấn luyện mô hình thông qua môi trường Python với sự hỗ trợ của các thư viện như scikit-learn, OpenCV và Flask. Đồng thời, việc trực quan hóa dữ liệu và kết quả mô hình cũng được thực hiện bằng thư viện matplotlib.

**Phương Pháp Nghiên Cứu Lý Luận**

Nghiên cứu tiến hành khảo sát, tổng hợp và phân tích các tài liệu liên quan đến lĩnh vực machine learning, phát hiện vi biểu cảm khuôn mặt và ứng dụng trong phân tích hành vi. Bên cạnh việc tìm hiểu lý thuyết, nhóm nghiên cứu sử dụng hai hướng tiếp cận chính:

• **Phương pháp mô hình hóa:** Xây dựng mô hình Random Forest để phân loại trạng thái nói thật/nói dối từ ảnh khuôn mặt và áp dụng các kỹ thuật như data augmentation, feature engineering để cải thiện hiệu suất.

• **Phương pháp thực nghiệm:** Thử nghiệm mô hình trên dữ liệu huấn luyện, hiệu chỉnh hyperparameters, và đánh giá độ chính xác qua dữ liệu kiểm thử. Đồng thời, tích hợp hệ thống real-time detection sử dụng Flask và OpenCV để hiển thị kết quả phân loại với độ tin cậy.

## 1.5. Cấu Trúc Của Đề Tài

Cấu trúc của đề tài này gồm có các nội dung như sau:

**CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI**

**CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ ÁP DỤNG**

**CHƯƠNG 3: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG**

**CHƯƠNG 4: KẾT QUẢ VÀ ĐÁNH GIÁ**

**KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**

**DANH MỤC THAM KHẢO**