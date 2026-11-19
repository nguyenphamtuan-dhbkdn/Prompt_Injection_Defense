# Lựa chọn ngưỡng vận hành M5

Tự sinh từ E03; không sửa số đo bằng tay. Chọn riêng cho từng mô hình bằng F1 lớn nhất trên validation; hòa thì ưu tiên recall cao hơn rồi ngưỡng nhỏ hơn. F1 cân bằng precision/recall khi chưa có chi phí vận hành được xác định. Đây là lựa chọn nghiên cứu, không phải cam kết giới hạn FPR trong sản phẩm.

- **logreg**: ngưỡng `0.5`; F1 validation `1.000000`; recall validation `1.000000`. FPR corpus lành `0.340000` (51/150); recall test `1.000000`.
- **svm**: ngưỡng `0.5`; F1 validation `1.000000`; recall validation `1.000000`. FPR corpus lành `0.353333` (53/150); recall test `1.000000`.
- **rf**: ngưỡng `0.27`; F1 validation `0.857143`; recall validation `1.000000`. FPR corpus lành `0.426667` (64/150); recall test `1.000000`.

Xem `e03_thresholds_fixture.csv` để kiểm tra toàn bộ ứng viên, `e03_fpr_fixture.csv` để so sánh với ngưỡng mặc định và xem CI. `m5_predictions_fixture.csv` giữ score/ID để tái tính. Test và corpus lành chỉ dùng đánh giá sau khi khóa ngưỡng.

CI percentile có thể suy biến khi không quan sát lỗi; không chứng minh tỷ lệ lỗi thật bằng không. Corpus tự biên soạn chưa được duyệt độc lập, không đại diện lưu lượng thực tế. Bypass chỉ là bỏ lọt qua bộ phân loại, không đo việc chatbot làm theo chỉ dẫn. Dịch bị bỏ qua khi thiếu backend; multi-turn được chặn nếu bất kỳ lượt nào có score đạt ngưỡng. F1 cao vẫn có thể bỏ lọt biến thể chưa thấy; không dùng kết quả này như bảo đảm an toàn sản phẩm.

