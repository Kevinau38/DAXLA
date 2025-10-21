# LỜI MỞ ĐẦU

Công nghệ trí tuệ nhân tạo (AI) đang ngày càng phát triển mạnh mẽ và trở thành một phần không thể thiếu trong nhiều lĩnh vực của đời sống hiện đại. Trong đó, lĩnh vực nhận diện hình ảnh và phân tích hành vi con người đang thu hút sự quan tâm lớn từ cộng đồng nghiên cứu và doanh nghiệp, đặc biệt là khi ứng dụng vào các hệ thống an ninh, phỏng vấn tuyển dụng và đánh giá tâm lý.

Một trong những hướng nghiên cứu đầy thách thức là phát hiện vi biểu cảm khuôn mặt – một kỹ thuật cho phép máy tính phân tích những thay đổi tinh tế trên gương mặt để suy luận tính chân thực trong lời nói của con người. Khác với nhận diện cảm xúc cơ bản, việc phát hiện vi biểu cảm đòi hỏi độ chính xác cao hơn để phân biệt giữa trạng thái nói thật và nói dối thông qua các dấu hiệu biểu cảm không tự nhiên, căng thẳng hoặc che giấu.

Đề tài này tập trung vào việc xây dựng một hệ thống phát hiện vi biểu cảm khuôn mặt thời gian thực thông qua webcam, sử dụng thuật toán Random Forest được huấn luyện với dữ liệu hình ảnh biểu cảm được phân loại thành hai nhóm: nói thật (happy, neutral, surprise) và nói dối (angry, sad, fear, disgust). Hệ thống được triển khai bằng Flask – một framework web nhẹ, và sử dụng OpenCV cho xử lý hình ảnh real-time.

Ngoài việc giới thiệu kiến trúc tổng thể, đề tài cũng phân tích các bước tiền xử lý ảnh grayscale 48x48, cấu trúc mô hình Random Forest với data augmentation, cũng như cách tối ưu hóa hiệu suất phát hiện thông qua ngưỡng tin cậy và logic cân bằng prediction để giảm thiểu false positive.

Khi hoàn thành đề tài, em xin gửi lời cảm ơn chân thành đến quý thầy cô đã tận tình giảng dạy và hướng dẫn. Đặc biệt, chúng em xin tri ân cô Võ Thị Hồng Thắm đã trực tiếp hướng dẫn và hỗ trợ em trong suốt quá trình nghiên cứu và thực hiện đề tài. Những kiến thức và kinh nghiệm quý báu mà thầy cô chia sẻ là nguồn động lực to lớn giúp em hoàn thiện bài báo cáo này.