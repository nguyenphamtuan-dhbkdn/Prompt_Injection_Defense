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

## Quyết định M3

- Chỉ thêm baseline transformer tùy chọn và timing dùng lại cho E5; chưa làm
  E5 so sánh, bootstrap CI hoặc mốc M4 trở đi. Dùng entry point riêng
  `experiments/e01_transformer.py` để giữ nguyên kết quả và luồng chạy M2.
- Đã đọc source của gói Transformers ghim trước khi viết adapter:
  `PreTrainedTokenizerBase.__call__`, `PreTrainedModel.from_pretrained`,
  `save_pretrained` và `DistilBertForSequenceClassification.forward`.
  Dùng AutoTokenizer/AutoModelForSequenceClassification, logits và loss trả
  bởi `return_dict=True`; không dùng Trainer nên không cần accelerate.
- Lazy import qua `import_module` trong `try/except ImportError`; không import
  torch/transformers ở module scope hoặc trong type hints. Lỗi thiếu extra có
  hướng dẫn cài rõ ràng. Kiểm thử subprocess mô phỏng cả hai gói không tồn tại
  và fit classical thật; kiểm thử slow được bỏ qua nếu thiếu backend.
- Fine-tune toàn bộ trọng số bằng CPU, eager attention, AdamW, learning rate
  cố định, weight decay, gradient clipping. Không scheduler, early stopping
  hoặc chọn checkpoint theo test. Giữ budget mặc định của cấu hình SPEC.
  Giới hạn CPU hai thread; kiểm tra max_length không vượt position capacity.
- Thư viện torch không nhận NumPy Generator cho dropout/khởi tạo: rút seed
  nguyên từ Generator duy nhất sau chia dữ liệu, seed CPU torch trong
  `fork_rng(devices=[])`, phục hồi RNG và thiết lập toàn cục khi thoát.
  Mỗi epoch shuffle trực tiếp bằng Generator đó. Bật deterministic algorithms;
  không hứa tái lập bitwise giữa phiên bản/hardware khác nhau.
- Revision pretrained được xác minh từ API Hugging Face và ghim trong mã.
  Metadata giữ revision yêu cầu/đã resolve, config kiến trúc, optimizer, seed,
  phiên bản, commit/dirty, hash mã/checkpoint/CSV và ID/hash split như M2.
  Cache mô hình nằm dưới data/raw; checkpoint không chứa payload huấn luyện.
- Dùng cùng metric M2, injection positive, softmax lớp một, threshold cố định
  và train-only fit. Không áp sàn classical lên transformer để rồi tuning trên
  test; báo cáo đủ chỉ số đo thật. Tên bảng khoa học từ chối --fixture.
- Timing bằng perf_counter_ns, batch một, bao gồm tokenization và host score,
  loại load/train/warmup; lưu từng mẫu và mean/median/p95. Đây là phép đo sơ bộ
  cho RQ5, chưa có CI hoặc kết luận so sánh classical. Giữ thông tin CPU/thread
  trong metadata để diễn giải; timing không phải đại lượng tái lập chính xác.
- Slow test dùng tokenizer cục bộ và DistilBERT nhỏ khởi tạo có seed để kiểm
  tra tái lập trọng số/score, gradient cập nhật, contract chung và save/load
  ngoại tuyến. Không dùng mạng ngẫu nhiên đó để tuyên bố hiệu quả pretrained.
  Coverage mặc định vẫn áp toàn src; không loại code transformer khỏi mẫu số.
- Mọi kết quả chỉ phục vụ nghiên cứu: truncation có thể bỏ phần chỉ dẫn nguy
  hiểm; softmax chưa hiệu chỉnh; F1 tổng hợp cao vẫn có thể bỏ lọt tấn công.

## Quyết định M4

- Chỉ triển khai generator trong `src/pidefense/attacks/` và kiểm thử; chưa
  triển khai E2, bypass rate, middleware, CLI hoặc gửi yêu cầu tới chatbot.
  Mỗi tệp nguồn trong gói bắt đầu bằng chú thích authorized-testing-only.
- Giữ hàm biến đổi chuỗi theo Phần 6. `generate_variant(sample, technique,
  rng, ...)` nhận hàng nhãn injection và trả `AttackVariant` bất biến, giữ
  `source`, `source_id`, nhãn, technique, text và tuple turns. Không sửa hàng
  nguồn; direct lấy nguyên văn hàng có nhãn. Trường chứa raw text không xuất
  hiện trong repr của variant. Generator không ghi log, tệp hoặc gọi mạng.
- Giới hạn payload đầu vào theo code point bằng `MAX_PAYLOAD_LENGTH` trong
  `attacks/common.py`; rỗng, sai kiểu hoặc quá dài gây ValueError không chứa
  raw text. Whitespace, control character và Unicode đều hợp lệ ở cấp hàm
  chuỗi, dù schema dữ liệu chỉ chấp nhận hàng có nội dung. Carrier và các
  bước dịch cũng được giới hạn để tránh đầu ra backend không hợp lệ.
- Base64 gồm lời dẫn cố định `BASE64_PREFIX` và phần Base64 chuẩn của UTF-8;
  giải mã phần sau prefix để lấy payload. `surrogatepass` cho phép round-trip
  cả lone surrogate trong chuỗi Python; không khẳng định surrogate là Unicode
  hợp lệ để gửi chatbot. Leetspeak thay từng ký tự theo bảng cố định, lấy mẫu
  bằng Generator được truyền vào; nếu không trúng ký tự nào thì ép một phép
  thay. Nếu không có ký tự phù hợp thì thêm khoảng trắng để thỏa acceptance
  luôn khác đầu vào; fallback này không đại diện obfuscation có ý nghĩa.
- Zero-width chèn một U+200B tại biên lấy mẫu bằng cùng Generator, kể cả biên
  đầu/cuối để hỗ trợ chuỗi một ký tự. Không xóa hoặc chuẩn hóa nội dung nguồn;
  xóa toàn bộ U+200B không phải phép đảo chính xác nếu nguồn đã có U+200B.
- Dịch round-trip dùng `Translator` callable được inject rõ ràng, với hợp
  đồng `(text, source_lang, target_lang) -> str`, thứ tự en→vi→en mặc định.
  Người dùng cung cấp backend dịch cục bộ; M4 không đóng gói mô hình dịch,
  không tự tải dependency hoặc dùng dịch vụ mạng. Thiếu backend, lỗi, đầu
  ra rỗng/quá dài/sai kiểu hoặc round-trip không đổi: trả chuỗi gốc;
  wrapper đánh dấu skipped với lý do cố định, không tính là biến thể thành
  công. Backend phải tránh log nội dung và dùng RNG của lần chạy nếu cần.
  Kiểm thử dùng backend giả để xác minh routing/fallback, không dùng nó để
  tuyên bố chất lượng dịch thật. Backend chặn lâu cần tự có timeout.
- Indirect thay đúng một marker `{payload}` theo nghĩa literal; giữ nguyên
  các dấu ngoặc khác, không format hoặc diễn giải payload. Carrier mặc định
  gồm yêu cầu tóm tắt và tài liệu chứa nguyên văn payload. Không cho carrier
  chỉ có marker vì sẽ tạo phép biến đổi đồng nhất.
- Multi-turn chỉ hỗ trợ hai hoặc ba lượt, chia code point liên tiếp cân
  bằng, không strip whitespace; ghép lại đúng chuỗi gốc. Từ chối nguồn ngắn
  hơn số lượt thay vì tạo lượt rỗng. `variant.turns` là dữ liệu hội thoại có
  thứ tự; `variant.text` chỉ là phép ghép để truy nguồn, không thay thế việc
  gửi từng lượt trong harness ở mốc sau. Không khẳng định từng lượt là benign.
- Spot-check phương pháp với yêu cầu tóm tắt lành: bảng leetspeak chỉ đổi
  ký tự cho phép, không đảo thứ tự hoặc bỏ từ; Base64 và splitting khôi phục
  chính xác; indirect giữ nguyên payload, zero-width chỉ chèn ký tự. Kiểm
  thử kiểm tra thêm các bất biến này trên hàng injection từ fixture có nguồn.
  Đây là kiểm tra cấu trúc và xem xét định tính phương pháp, không chứng minh
  bảo toàn ngữ nghĩa trước LLM. Dịch thật cần duyệt thủ công riêng khi cung
  cấp backend; leetspeak, ký tự ẩn và ngữ cảnh tài liệu có thể đổi cách LLM
  hiểu chỉ dẫn. Chưa đo bypass hoặc tuyên bố hiệu quả an toàn ở M4.
- Hypothesis kiểm tra ASCII/Unicode kể cả surrogate, tính khác/không rỗng,
  round-trip Base64, độ dài zero-width và tổng tính của nhánh dịch offline.
  Tắt example database để không lưu raw payload ngoài data/results; dùng
  derandomize cho dữ liệu kiểm thử tái lập. Ngẫu nhiên của generator sản phẩm
  luôn đi qua `SimRNG.generator`, không tạo stream riêng trong transform.

## Quyết định M5

- Chỉ triển khai E02/E03, không làm middleware, gọi Ollama hoặc triển khai
  các mốc tiếp theo. Hai entry point chạy chung `experiments/m5_evaluation.py`
  để dùng cùng mô hình, ngưỡng và biến thể. Classical được fit lại bằng đúng
  thứ tự/seed E01 trước các bước ngẫu nhiên khác. Transformer là tùy chọn
  qua `--checkpoint`; khi chọn, bắt buộc kiểm tra hash checkpoint, cấu hình
  và ID/hash cả ba split với metadata M3, tránh đánh giá nhầm mô hình/rò rỉ.
  Thiếu extra vẫn chạy classical; không tạo số đo transformer giả.
- Chọn ngưỡng riêng mỗi mô hình bằng F1 validation lớn nhất; ứng viên gồm
  score validation và ngưỡng mặc định. Hòa F1 thì ưu tiên recall rồi ngưỡng
  nhỏ hơn. F1 cân bằng precision/recall khi chưa có chi phí FP/FN thực tế;
  đây không phải tối ưu chi phí vận hành hay bảo đảm FPR. Test và corpus lành
  được giữ ngoài chọn ngưỡng. Bảng lưu cả ngưỡng mặc định và ngưỡng chọn để
  thấy đánh đổi. Lựa chọn cụ thể và số đo được mã thí nghiệm sinh vào
  [giải thích ngưỡng](../results/tables/e03_operating_thresholds.md);
  toàn bộ ứng viên nằm trong `e03_thresholds.csv`, không chép số đo thủ công.
- Bypass = số nguồn injection được cho qua / số nguồn được đánh giá, không
  điều kiện hóa trên việc nguồn gốc đã bị chặn. Chỉ tạo biến thể từ test
  injection, mỗi technique một biến thể mỗi source, chia sẻ giữa các mô hình.
  Multi-turn dùng max score từng lượt: chỉ bypass khi mọi lượt được cho qua;
  không dùng chuỗi ghép như một tin nhắn. Đây là giả định guard từng lượt,
  chưa mô phỏng bộ nhớ hội thoại và không đo chatbot tuân theo tấn công.
- Dịch thiếu backend cục bộ được ghi attempted/skipped với mẫu số evaluated
  bằng không và rate/CI để trống, không gán bypass bằng zero. Không thay backend
  dịch thật bằng stub. Các kỹ thuật còn lại chạy đầy đủ; score/ID/trạng thái
  lưu trong `m5_predictions.csv` để tái tính, không lưu raw text vào log.
- FPR = FP/(FP+TN) trên corpus lành riêng; từ chối nếu corpus giao với bất kỳ
  split nào theo chuẩn hóa M1. PR cần cả hai nhãn nên dùng validation và test
  deepset, không ghép corpus lành vào dữ liệu chọn ngưỡng. Hình PR vẽ test,
  đánh dấu ngưỡng đã chọn từ validation; CSV giữ cả hai đường cong cùng điểm
  cuối không có threshold. PNG xuất 300 dpi, PDF giữ vector và dpi tương tự
  cho thành phần raster. Hình bypass hiển thị CI và bỏ nhóm không đánh giá.
- CI percentile bootstrap theo source trong từng technique cho bypass và
  theo hàng corpus cho FPR; precision/recall/F1 test resample phân tầng lớp.
  Dùng cùng Generator và budget ghi trong metadata. CI giữ ngưỡng đã chọn,
  chưa bao gồm bất định do fit model, chọn ngưỡng, dịch hoặc biến thể khác.
  CI suy biến khi không quan sát lỗi không chứng minh xác suất lỗi bằng không.
- Full run từ chối fixture; `--fixture` luôn thêm hậu tố tên tệp, không ghi
  đè bảng khoa học. Golden nhỏ được sinh từ cùng runner với budget CI nhỏ;
  kiểm thử tái tính rate từ score, kiểm tra provenance và dpi. Không chạy
  full dataset/transformer trong CI mặc định. Metadata cạnh hai bảng chứa
  config, seed, commit/dirty, phiên bản, split/benign ID+hash, hash mã và output.
- Cache Arrow đã chuẩn bị có thể dùng qua `--prepared-cache` khi Hub không
  kết nối được. Chỉ chấp nhận thư mục revision ghim, kiểm tra số hàng rồi
  so sánh toàn bộ split ID/nhãn/hash với metadata E01 đầy đủ trước khi fit;
  không đổi loader M1 hay nhận fixture làm dữ liệu khoa học. Metadata E02/E03
  giữ hash các tệp cache để tái lập, không cần dịch vụ ngoài khi chạy M5.
- Corpus lành tự biên soạn, thiên về phân phối giới hạn và chưa duyệt độc lập;
  FPR không ước lượng đại diện cho lưu lượng sản phẩm. Score SVM/softmax chưa
  hiệu chỉnh. Ngưỡng chọn trên dữ liệu gốc có thể bỏ lọt obfuscation; dự án chỉ
  phục vụ nghiên cứu/đánh giá, không phải hệ thống an toàn cấp sản phẩm.

## Quyết định M6

- Chỉ triển khai middleware và HTTP client cục bộ cần cho middleware; chưa
  triển khai harness M7, demo CLI/FastAPI hoặc phép đo M8. `GuardedChat` là
  stateless: mỗi `send` tạo một system message và một user message, không có
  bộ nhớ hội thoại. Không import corpus `attacks` trong middleware/demo.
- `ClassifierGuard` nhận `Detector` đã fit, giữ hợp đồng score hướng injection
  từ M2/M3. Dùng `score >= threshold`, không dựa vào label mặc định của model;
  threshold mặc định theo cấu hình SPEC, người gọi truyền ngưỡng validation M5
  tương ứng model khi cần. Không tự tải model hoặc chọn ngưỡng trên dữ liệu mới.
  `technique_guess=None` vì detector nhị phân chưa được huấn luyện đoán kỹ thuật.
  Chuỗi rỗng/blank, sai kiểu và quá giới hạn `MAX_TEXT_LENGTH` bị từ chối;
  score không hữu hạn hoặc ngoài miền xác suất gây lỗi, không cho qua.
- Chỉ system message chứa chỉ dẫn có thẩm quyền. User request đặt trong
  `<untrusted_user>`, tài liệu tham chiếu đặt trong `<untrusted_document>`.
  Mọi nội dung nguồn được HTML-escape cả dấu ngoặc, ampersand và quote trước
  khi bọc; không unescape lại trước khi gửi. Vì vậy closing tag do nguồn cung
  cấp không thể kết thúc block trong cấu trúc chuỗi. Template giải thích rõ
  cả nội dung đã giải mã cũng không có thẩm quyền đổi vai trò/quy tắc.
  Đây là quy ước cho LLM, không phải ranh giới bảo mật được chứng minh.
- Screen user và document riêng, chặn ngay nếu một nguồn bị gắn injection;
  cả hai phải qua guard trước khi gọi Ollama. `ChatTurnResult.status` là
  allowed/blocked/error, chỉ allowed có response; repr không chứa response.
  Decision là nguồn đầu tiên bị chặn hoặc score lớn nhất khi các nguồn qua
  guard; rule cho biết classifier/output_filter/passed/layer_error. Mọi lỗi
  layer trả error cố định, không trả exception/backend body và không gọi
  chatbot tiếp sau lỗi input. Invalid input gọi guard trực tiếp gây ValueError.
- Output filter trả True khi cần chặn. Dùng private marker cố định, span từ
  liên tiếp trùng system, regex xác nhận tuân theo override và độ phủ tập từ
  system trong response. Độ phủ từ là semantic proxy nhẹ, không phải mô hình
  embedding hoặc bộ xác minh: không hiểu negation đầy đủ, đồng nghĩa hay ý đồ.
  Tham số heuristic được ghi ở docstring, chưa tuning bằng kết quả chatbot.
  Log rule output_filter dùng label injection/score một như cờ quy tắc nhị
  phân, không diễn giải thành xác suất model hay confidence đã hiệu chỉnh.
- Đã đối chiếu [Ollama Chat API](https://docs.ollama.com/api/chat): POST
  `/api/chat`, messages role/content, `stream=false`, đọc message.content
  khi done=true. Dùng urllib chuẩn Python để classical path không cần extra
  demo. Kiểm tra host bằng MiddlewareConfig; tắt proxy môi trường và redirect
  để không chuyển payload khỏi loopback. Timeout hữu hạn, giới hạn byte phản
  hồi và từ chối response sai cấu trúc. Không cần cài Ollama Python SDK.
- Log duy nhất là audit metadata: độ dài, SHA-256 tám hex đầu, label, score,
  rule ID cố định. Không log exception/stack trace, request, document, system
  hay response. Integration capture cả allow/block/error; kiểm thử grep-style
  quét sink trong middleware/demo, bổ sung AST allowlist cho biểu thức audit.
  Không cho phép sink log mới nếu chưa cập nhật chính sách và kiểm thử.
- Integration dùng detector TF-IDF thực fit fixture ngoại tuyến và mock riêng
  HTTP opener để kiểm tra contract request/response, không cần Ollama sống
  hay torch. Các prompt cố định chỉ kiểm tra nối các layer, không dùng làm
  ước lượng chất lượng held-out. ID pytest của payload là tên trường hợp,
  tránh đưa toàn nội dung vào tên/log kiểm thử. Chưa chạy red-team live M7.
- Đây vẫn là middleware nghiên cứu/đánh giá. Classifier có thể bỏ lọt biến
  thể, delimiter chỉ hướng dẫn model, output heuristic có thể báo sai/bỏ lọt
  paraphrase hoặc disclosure ngắn. Không tuyên bố an toàn cấp sản phẩm và
  không tạo bảng số đo mới khi chưa có thí nghiệm nguồn.
