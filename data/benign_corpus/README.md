# Câu hỏi lành tự biên soạn

`benign_queries.csv` dùng nguồn **student-curated, not scraped** ở mỗi hàng.
Nội dung được trợ lý lập trình viết mới cho dự án sinh viên theo yêu cầu;
không lấy từ web, tập dữ liệu hay dịch vụ chatbot. Chưa có xác nhận duyệt
thủ công độc lập của sinh viên. Trường `topic` ghi nhóm chủ đề, `source_id`
là định danh ổn định, nhãn `0` biểu thị câu hỏi lành.

Các câu hỏi bao gồm nấu ăn, làm vườn, khoa học, toán, lập trình, viết, ngôn ngữ,
lịch sử, địa lý, nghệ thuật, học tập, việc nhà, du lịch, môi trường và kế hoạch
hằng ngày. Có nội dung Unicode và câu hỏi kỹ thuật thông thường để tránh
chỉ kiểm tra những câu giao tiếp rất ngắn.

Giữ corpus riêng để đánh giá báo sai ở mốc sau; không trộn vào train/val/test.
Đây là tập tiện dụng tự biên soạn, không đại diện cho phân phối người dùng
thực tế và không chứng minh an toàn trong môi trường sản phẩm.
