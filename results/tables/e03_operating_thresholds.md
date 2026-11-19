# Lựa chọn ngưỡng vận hành M5

Tự sinh từ E03; không sửa số đo bằng tay. Chọn riêng cho từng mô hình bằng F1 lớn nhất trên validation; hòa thì ưu tiên recall cao hơn rồi ngưỡng nhỏ hơn. F1 cân bằng precision/recall khi chưa có chi phí vận hành được xác định. Đây là lựa chọn nghiên cứu, không phải cam kết giới hạn FPR trong sản phẩm.

- **logreg**: ngưỡng `0.4066400053`; F1 validation `0.962963`; recall validation `1.000000`. FPR corpus lành `0.146667` (22/150); recall test `0.849057`.
- **svm**: ngưỡng `0.5723147655`; F1 validation `0.980392`; recall validation `0.961538`. FPR corpus lành `0.013333` (2/150); recall test `0.773585`.
- **rf**: ngưỡng `0.37`; F1 validation `0.962963`; recall validation `1.000000`. FPR corpus lành `0.060000` (9/150); recall test `0.886792`.
- **distilbert**: ngưỡng `0.9670881629`; F1 validation `0.961538`; recall validation `0.961538`. FPR corpus lành `0.380000` (57/150); recall test `0.962264`.

Xem `e03_thresholds.csv` để kiểm tra toàn bộ ứng viên, `e03_fpr.csv` để so sánh với ngưỡng mặc định và xem CI. `m5_predictions.csv` giữ score/ID để tái tính. Test và corpus lành chỉ dùng đánh giá sau khi khóa ngưỡng.

CI percentile có thể suy biến khi không quan sát lỗi; không chứng minh tỷ lệ lỗi thật bằng không. Corpus tự biên soạn chưa được duyệt độc lập, không đại diện lưu lượng thực tế. Bypass chỉ là bỏ lọt qua bộ phân loại, không đo việc chatbot làm theo chỉ dẫn. Dịch bị bỏ qua khi thiếu backend; multi-turn được chặn nếu bất kỳ lượt nào có score đạt ngưỡng. F1 cao vẫn có thể bỏ lọt biến thể chưa thấy; không dùng kết quả này như bảo đảm an toàn sản phẩm.

