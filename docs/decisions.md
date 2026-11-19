# Quyết định thiết kế M0

- Phạm vi theo mục M0 ở Phần 5 của SPEC.md. Các tham chiếu tới Phần 13
  trong phần mở đầu là nhầm số mục; Phần 13 chứa lời nhắc khởi động.
  Chỉ triển khai cấu hình và RNG, tạo vị trí dành sẵn cho các mô-đun còn lại
  đúng Phần 3. Chưa tải dữ liệu, huấn luyện, tạo payload hay chạy thí nghiệm.
- Tài liệu cho người dùng viết bằng tiếng Việt. Không đưa số đo, bảng hay hình
  đánh giá vào báo cáo khi chưa có thí nghiệm tạo dữ liệu nguồn.
- Dùng dataclass frozen với tên trường theo Phần 6 và mặc định theo Phần 7.
  YAML dùng safe loader; từ chối tài liệu rỗng, sai kiểu, trường lạ, giá trị
  không hữu hạn và các lựa chọn ngoài danh sách. Mapping rỗng dùng mặc định;
  các cấu hình mô hình là cấu hình từng phần, không có cơ chế kế thừa tệp.
- Hai tỷ lệ chia được tính trên toàn bộ dữ liệu: train = 1 - test - val.
  Test phải dương, validation có thể rỗng, train phải dương. Seed là số nguyên
  không âm, không chấp nhận boolean. Không kiểm tra sự tồn tại của dữ liệu
  ở M0. Đường dẫn giữ nguyên tính tương đối với thư mục làm việc.
- `AttackConfig.techniques` dùng tuple thay vì list để bảo toàn cấu hình
  bất biến; YAML vẫn dùng danh sách, loader chuyển sang tuple. Danh sách
  rỗng được phép để tắt các kỹ thuật, tên trùng bị từ chối.
- Máy chủ Ollama chỉ chấp nhận URL HTTP(S) loopback (`localhost`,
  `127.0.0.1` hoặc `::1`); không có
  thông tin đăng nhập, query, fragment hay đường dẫn API. M0 không kết nối
  máy chủ. Điều này hỗ trợ phạm vi chỉ kiểm thử demo của dự án.
- `SimRNG` tạo đúng một `numpy.random.Generator(PCG64(seed))` cho mỗi lần
  chạy; truyền cùng thuộc tính `generator` tới mọi bước ngẫu nhiên. Không
  dùng RNG toàn cục, không tự tạo stream con và không reseed khi truy cập.
  Tái lập đòi hỏi giữ seed, phiên bản NumPy và thứ tự gọi giống nhau. Việc
  lưu metadata thí nghiệm (cấu hình, seed, commit, phiên bản) sẽ đi cùng
  triển khai thí nghiệm ở mốc tương ứng, chưa tạo kết quả giả ở M0.
- Ghim phiên bản trực tiếp cho runtime, dev và extra trong `pyproject.toml`;
  dùng `uv.lock` cho phụ thuộc bắc cầu của `uv run`. Các extra nặng không được
  import hay cài để kiểm tra M0. Backend dịch là `deep-translator`, chỉ khai
  báo phụ thuộc tùy chọn, chưa dùng API hoặc gọi mạng.
- Makefile dùng cùng lệnh uv với `tasks.ps1`. Setup dùng
  `uv sync --locked --extra dev` để cài editable theo lockfile, tránh việc
  `uv pip install` tự resolve khác phần phụ thuộc bắc cầu do các extra được
  resolve chung trong lockfile. Lệnh `uv pip install -e ".[dev]"` vẫn được
  hỗ trợ và đã dùng để cài ban đầu. Máy hiện tại không có GNU Make;
  kiểm tra setup và chất lượng bằng lệnh uv trực tiếp và helper PowerShell.
  Các đích của mốc sau thất bại với thông báo rõ ràng, không giả vờ thành công.
- Dùng cache uv trong thư mục dự án để tránh lỗi quyền truy cập cache mặc định
  của Windows sandbox. Không đưa cache hay môi trường ảo vào Git.
- Kiểm thử M0 tập trung vào cấu hình, RNG và phạm vi gói attacks. Áp dụng
  ngưỡng coverage trong cấu hình pytest theo yêu cầu AGENTS.md, đồng thời đăng
  ký marker statistical và slow; slow bị loại khỏi lượt chạy mặc định.
- Mỗi tệp dưới gói attacks đều có chú thích phạm vi authorized-testing-only
  ở đầu. Các mô-đun này chỉ là placeholder và không tạo payload. Khi đóng gói
  sản phẩm ở mốc sau phải tách tập đánh giá khỏi middleware và mô hình.

## Quyết định M1

- Giữ chữ ký `load_deepset_prompt_injections(cfg, rng) -> DatasetSplits` theo
  Phần 6. Dùng API `datasets.load_dataset` đã kiểm tra từ phiên bản cài đặt;
  ghim revision upstream trong mã. Pool các split gốc rồi chia lại theo cấu
  hình dự án; đây không phải giao thức benchmark giữ nguyên test upstream.
- `offline=True` đọc CSV trực tiếp, không gọi Hub. Lỗi kết nối hoặc I/O cache
  tự chuyển CSV với cảnh báo không chứa payload hay nội dung exception.
  Lỗi schema/nhãn không tự chuyển dữ liệu. Trường `origin` phân biệt tập đầy
  đủ và fixture; tuyệt đối không dùng fixture để báo cáo hiệu quả mô hình.
  Đường dẫn tương đối tính từ thư mục gốc khi chạy, giống quyết định M0.
- CSV ngoại tuyến lấy nguyên văn từ revision ghim, giữ split/chỉ số hàng
  trong `source_id`; script `experiments/prepare_m1_sample.py` và metadata
  cho phép tái tạo, kiểm tra checksum và truy nguồn. Không dùng ngẫu nhiên
  khi tạo fixture; seed metadata là null. Cache tải về không đưa vào Git.
- Schema bất biến gồm text, label, source, source_id, technique tùy chọn.
  Nhãn chỉ nhận số nguyên 0/1, không nhận boolean. Không suy đoán kỹ thuật
  tấn công vì upstream không gắn nhãn này.
- Loại trùng trước khi chia bằng NFKC + casefold + gộp whitespace; giữ nguyên
  text và nguồn của lần xuất hiện đầu tiên. Nhãn mâu thuẫn trên cùng văn bản
  chuẩn hóa gây lỗi rõ ràng. Chính sách bảo thủ này có thể gộp các câu phân
  biệt chữ hoa/thường, nhằm tránh rò rỉ giữa các split.
- Với từng nhãn, phân bổ theo phần nguyên của n_nhãn * tỷ_lệ, sau đó cấp phần
  dư cho phần thập phân lớn nhất; hòa thì ưu tiên train, val, test. Mỗi lớp
  sai lệch dưới một hàng so với tỷ lệ lý tưởng. Từ chối dữ liệu quá nhỏ nếu
  bất kỳ split có tỷ lệ dương thiếu lớp; validation bằng không được phép.
  Mọi shuffle dùng đúng Generator của lần chạy, không tạo RNG khác.
- `label_balance()` trả số lượng và tỷ lệ từng nhãn trên mỗi split, kể cả
  validation rỗng. Không ghi raw text hoặc tự sinh bảng kết quả thí nghiệm.
- Corpus lành viết mới cho dự án với nguồn `student-curated, not scraped`;
  README corpus nêu rõ trợ lý hỗ trợ biên soạn, chưa có duyệt độc lập. Giữ
  riêng để đánh giá báo sai ở mốc sau, không đưa vào huấn luyện. Corpus này
  không đại diện mọi ngôn ngữ hoặc tình huống thực tế.
- Kiểm thử mặc định hoàn toàn ngoại tuyến: fixture có checksum/provenance,
  DatasetDict thật trong bộ nhớ để kiểm tra adapter online, giả lập lỗi tải,
  tỷ lệ chia và cân bằng nhãn, tái lập seed, tiến triển stream, làm tròn,
  loại trùng/rò rỉ, dữ liệu quá nhỏ và CSV không hợp lệ. Không triển khai M2.

## Quyết định M2

- Chỉ triển khai baseline classical và các chỉ số cần cho E01; chưa làm
  transformer, bootstrap, attack, middleware hoặc threshold tuning của mốc sau.
  Dùng entry point thí nghiệm chạy bằng Python/PowerShell; CLI tổng quát và
  lưu/nạp model triển khai để mốc sau, không cần cho acceptance M2.
- Theo yêu cầu M2, `Detector.predict(text)` trả tuple `(label, score)`, thay
  vì `GuardDecision` minh họa ở Phần 6. Label là benign/injection; score luôn
  hướng về injection. Middleware có thể chuyển tuple sang GuardDecision ở M6.
- Word n-gram và char n-gram đúng Phần 5; giữ mặc định TF-IDF scikit-learn:
  smooth IDF, tf tuyến tính, L2 riêng mỗi nhánh, ghép sparse với trọng số bằng
  nhau, không giới hạn vocabulary. Chỉ fit train, không fit lại trên val/test.
- Dùng mặc định estimator của phiên bản ghim, tăng giới hạn vòng lặp cho
  Logistic Regression và Linear SVM, chạy một worker để dễ tái lập. API đã
  kiểm tra bằng inspect/docstring của thư viện cài đặt trước khi triển khai.
  sklearn không nhận Generator tại random_state: lấy seed nguyên từ duy nhất
  Generator của lần chạy và truyền vào estimator; ngẫu nhiên nội bộ sklearn
  dùng seed đó, không tự reseed theo seed cấu hình. Ghi seed estimator và
  toàn bộ tham số vào metadata, giữ thứ tự logreg, svm, rf sau khi chia tập.
- LR/RF dùng xác suất lớp injection; SVM dùng sigmoid margin để có cùng miền
  score. Sigmoid chưa hiệu chỉnh xác suất; chỉ giữ thứ hạng AUC và ranh giới
  quyết định. Dùng threshold cố định, quy ước >=; không tuning trên test,
  validation để riêng. Metric coi injection là positive; precision không xác
  định trả về zero; AUC yêu cầu cả hai lớp, xử lý tie bằng nửa điểm.
- Sàn sanity là kiểm tra F1 của từng mô hình, khai báo trong mã thí nghiệm
  trước khi xem kết quả. Giá trị và trạng thái đọc từ cột f1_sanity_floor và
  sanity_pass của CSV; không chép số đo vào báo cáo. Các chỉ số khác vẫn báo
  đầy đủ, không áp cùng sàn vì chúng đo các khía cạnh khác nhau.
- E01 giữ giao thức M1 pool/deduplicate/resplit và từ chối fallback nhỏ khi
  chạy khoa học. Chế độ --fixture chỉ phục vụ hồi quy ngoại tuyến, ghi tệp
  riêng và không được ghi đè tên bảng khoa học. Golden do cùng mã thí nghiệm
  tạo ra; pytest tái tính và so sánh với dung sai số thực.
- Metadata cạnh mỗi CSV gồm cấu hình, seed, commit, working-tree dirty,
  phiên bản, hash mã nguồn/CSV và ID/hash văn bản theo split; không cần chép
  payload vào metadata. Hash mã nguồn xác định nội dung khi chưa commit.
  Tất cả kết quả giữ phạm vi nghiên cứu; F1 cao vẫn có thể bỏ lọt tấn công,
  chưa kiểm chứng trước biến thể đối kháng và không bảo đảm an toàn sản phẩm.
