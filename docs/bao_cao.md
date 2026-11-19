# Báo cáo nghiên cứu PI-DEFENSE

Khung báo cáo theo Phần 12 của SPEC.md. Baseline classical và transformer đã được tạo;
các phần khác chờ mốc tương ứng. Chỉ đọc số đo từ tệp kết quả thí nghiệm.

## Giới thiệu và động lực

Chờ hoàn thiện ở mốc báo cáo.

## Dữ liệu và phương pháp

Dùng revision deepset ghim trong metadata E01; pool split gốc, loại trùng
và chia lại phân tầng theo seed. Đây không phải benchmark giữ test upstream.
Corpus lành tự biên soạn được giữ riêng. Chỉ fit đặc trưng và mô hình trên train.

## Mô hình phát hiện và kết quả

TF-IDF kết hợp word/char n-gram; so sánh Logistic Regression, Linear SVM và
Random Forest trên cùng held-out test. Precision, recall, F1 và ROC-AUC lấy
trực tiếp từ [CSV E01](../results/tables/e01_classical.csv); cấu hình, split,
seed, commit và phiên bản nằm trong
[metadata](../results/tables/e01_classical.metadata.json).

Đọc kết quả hiện tại bằng PowerShell, không chép số đo vào văn bản:

```powershell
Import-Csv results/tables/e01_classical.csv |
    Select-Object model, precision, recall, f1, roc_auc, f1_sanity_floor, sanity_pass
```

Linear SVM dùng sigmoid margin làm score, chưa hiệu chỉnh xác suất. Không
chọn tham số hoặc threshold bằng test. Các số đo tổng hợp này không chứng
minh khả năng chống obfuscation hay an toàn triển khai.

DistilBERT tùy chọn được fine-tune trên cùng train và đánh giá cùng test với
M2, giữ validation riêng. Các chỉ số đo thật nằm trong
[CSV transformer](../results/tables/e01_transformer.csv), cùng
[metadata tái lập](../results/tables/e01_transformer.metadata.json).
Không chọn checkpoint hoặc threshold theo test. Mô hình cắt văn bản theo
max_length nên có thể bỏ qua phần chỉ dẫn nguy hiểm ở cuối; softmax chưa
được hiệu chỉnh xác suất. Chỉ số cao không chứng minh an toàn triển khai.

```powershell
Import-Csv results/tables/e01_transformer.csv |
    Select-Object model, precision, recall, f1, roc_auc
```

## Tỷ lệ bị vượt qua và báo sai

E02 đánh giá mỗi kỹ thuật trên nguồn injection held-out, cùng biến thể cho
mọi mô hình. Bypass là mọi lượt đều có score dưới ngưỡng; đây chỉ là bỏ lọt
bộ phân loại, chưa đo hành vi chatbot. Dịch thiếu backend được ghi skipped,
không nằm trong mẫu số. Xem [bảng E02](../results/tables/e02_bypass_rate.csv).

E03 đo FP/(FP+TN) trên corpus lành riêng; ngưỡng được khóa bằng F1 validation
lớn nhất, ưu tiên recall khi hòa. Không tuning theo test hoặc corpus lành.
Lựa chọn và lý do cụ thể được sinh từ mã trong
[giải thích ngưỡng](../results/tables/e03_operating_thresholds.md);
[bảng E03](../results/tables/e03_fpr.csv) so sánh ngưỡng mặc định và ngưỡng chọn.
CI bootstrap giữ cố định mô hình và ngưỡng, có thể suy biến khi chưa thấy lỗi;
không chứng minh rủi ro thực bằng không. Corpus nhỏ tự biên soạn chưa đại diện
cho lưu lượng thực tế. F1 cao vẫn có thể bỏ lọt biến thể, không bảo đảm an toàn.

![Precision–recall test và điểm vận hành](../results/figures/e03_precision_recall.png)

[PDF vector](../results/figures/e03_precision_recall.pdf), PNG xuất 300 dpi;
đường cong và ứng viên ngưỡng có CSV riêng để tái tính.
Chi tiết giao thức và giới hạn nằm trong [quyết định M5](decisions.md).

## Kiểm thử chatbot Ollama

Chưa chạy kiểm thử trên demo cục bộ.

## Hiệu năng và đánh đổi

M3 đã đo latency transformer trên CPU, batch một, bao gồm tokenize và loại
trừ load/train/warmup. Mean/median/p95 đọc từ CSV transformer; từng mẫu và
điều kiện đo nằm trong metadata cạnh nó. Chưa chạy E5 so sánh với classical
hoặc bootstrap CI; chưa kết luận phù hợp làm guard đồng bộ.

```powershell
Import-Csv results/tables/e01_transformer.csv |
    Select-Object model, latency_mean_ms, latency_median_ms, latency_p95_ms
```

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
