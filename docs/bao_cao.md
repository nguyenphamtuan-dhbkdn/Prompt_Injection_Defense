# Báo cáo nghiên cứu PI-DEFENSE

Khung báo cáo theo Phần 12 của SPEC.md. M0 chưa có dữ liệu hoặc kết quả
thực nghiệm; các phần kết quả sẽ đọc dữ liệu do mã thí nghiệm tạo trong results/.

## Giới thiệu và động lực

Chờ hoàn thiện ở mốc báo cáo.

## Dữ liệu và phương pháp

Chờ triển khai bộ nạp dữ liệu và phương pháp đánh giá.

## Mô hình phát hiện và kết quả

Chưa huấn luyện hoặc đánh giá mô hình.

## Tỷ lệ bị vượt qua và báo sai

Chưa chạy thí nghiệm.

## Kiểm thử chatbot Ollama

Chưa chạy kiểm thử trên demo cục bộ.

## Hiệu năng và đánh đổi

Chưa đo độ trễ.

## Độ bền vững trước nhiễu loạn đối kháng

Chưa chạy tìm kiếm nhiễu loạn.

## Kiến trúc phòng thủ nhiều lớp

Chờ triển khai middleware.

## Hạn chế và hướng mở rộng

- Dữ liệu dự kiến là ảnh chụp cố định, tương đối nhỏ và thiên về tiếng Anh;
  kết quả không khái quát cho mọi ngôn ngữ, mô hình hoặc cách diễn đạt mới.
- F1 tổng hợp cao vẫn có thể đi cùng tỷ lệ vượt qua đáng kể ở từng kỹ thuật;
  độ chính xác cao không bảo đảm an toàn.
- Bộ lọc đầu ra dự kiến dùng từ khóa, mẫu và kiểm tra ngữ nghĩa nhẹ; đây
  không phải bộ kiểm chứng hình thức và có thể bị né tránh.
- Phòng thủ nhiều lớp giảm nhưng không loại bỏ rủi ro. Dự án nghiên cứu này
  không tuyên bố sẵn sàng triển khai sản phẩm hoặc bảo đảm an toàn.
- Phạm vi red-team chỉ là mô hình instruct nhỏ trên demo Ollama cục bộ;
  mô hình lớn hoặc dịch vụ sản phẩm có thể phản ứng khác.
- Tập nhiễu loạn đối kháng dự kiến nhỏ, không phải tìm kiếm toàn diện;
  không tìm thấy trường hợp vượt qua không chứng minh độ bền vững.
- Chưa đánh giá hội thoại nhiều lượt, tool-use hay tác tử ngoài trường hợp
  tách chỉ dẫn đơn giản dự kiến; các trường hợp này là hướng mở rộng.

Tích hợp guard vào AI Pentest Copilot là hướng mở rộng độc lập, chưa triển khai.

## Kết luận

Chờ kết quả thực nghiệm.

## Phụ lục

Chờ cấu hình và lệnh tái lập thí nghiệm. Payload và phản hồi thô chỉ lưu trong
data/ hoặc results/ theo quy tắc bảo vệ log; báo cáo chỉ dùng ví dụ đã làm sạch.
