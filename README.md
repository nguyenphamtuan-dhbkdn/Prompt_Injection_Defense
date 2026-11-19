# PI-DEFENSE

Dự án nghiên cứu và đánh giá phòng thủ trước prompt injection. Hiện chỉ có
M0–M3: cấu hình YAML, RNG có seed, tải/chia dữ liệu, TF-IDF và các bộ phân loại
Logistic Regression, Linear SVM, Random Forest, cùng DistilBERT tùy chọn.
Các mô-đun tấn công và middleware vẫn là vị trí dành sẵn.

## Chạy trên Windows PowerShell

Yêu cầu Python 3.11 trở lên và `uv`. Dùng môi trường `.venv` đã có, chạy tại
thư mục gốc dự án:

```powershell
uv pip install -e ".[dev]"
uv run ruff check .
uv run mypy --strict src
uv run pytest
```

Nếu `uv` không nằm trong PATH, thay `uv` bằng
`& "$env:USERPROFILE\.local\bin\uv.exe"`. Có thể chạy các tác vụ tương đương:

```powershell
.\tasks.ps1 setup
.\tasks.ps1 check
```

Trên máy có GNU Make: `make setup`, `make lint`, `make typecheck`, `make test`.
Đích setup của Makefile và PowerShell dùng `uv sync --locked --extra dev`
để cài editable theo đúng lockfile, kể cả phụ thuộc bắc cầu. Lệnh
`uv pip install -e ".[dev]"` phía trên cũng được hỗ trợ; `uv run` sẽ đồng bộ
phần phụ thuộc runtime với lockfile khi cần.
Đích `experiments` chạy E01 classical; `figures`, `report`, `demo` chưa triển khai.
Kiểm thử `slow` mặc định bị bỏ qua; chọn bằng `uv run pytest -m slow` khi
các mốc tiếp theo bổ sung chúng. Chọn kiểm thử thống kê bằng
`uv run pytest -m statistical` khi đã có kiểm thử tương ứng.

## Cấu hình và tái lập

`configs/default.yaml` giữ nguyên mặc định trong SPEC.md. Các tệp cấu hình mô
hình chỉ thay những trường ghi rõ; trường thiếu dùng mặc định của dataclass.
Đường dẫn dữ liệu tương đối được tính từ thư mục làm việc; chạy lệnh tại gốc
dự án. Trường không hợp lệ, trường lạ và máy chủ Ollama ngoài loopback bị từ chối.

```python
from pidefense.config import load_config
from pidefense.rng import SimRNG

config = load_config("configs/default.yaml")
rng = SimRNG.from_config(config)
# Truyền cùng rng.generator cho mọi bước ngẫu nhiên trong một lần chạy.
```

## Dữ liệu M1

```python
from pidefense.data.loader import load_benign_corpus, load_deepset_prompt_injections

# Tiếp tục dùng config và rng của ví dụ trên. Offline không gọi mạng.
splits = load_deepset_prompt_injections(config.data, rng.generator, offline=True)
balance = splits.label_balance()  # Chỉ số lượng và tỷ lệ nhãn, không có raw text.
benign = load_benign_corpus(config.data.benign_corpus_path)
```

Bỏ `offline=True` để tải/cache `deepset/prompt-injections` tại revision được
ghim; lỗi kết nối/I/O tự dùng CSV nhỏ với cảnh báo. `splits.origin` cho biết
nguồn đang dùng. Tập CSV nhỏ chỉ kiểm tra pipeline trong CI. Tham khảo
[nguồn và cách tái tạo](data/raw/README.md).

Train/val/test phân tầng theo tỷ lệ cấu hình sau khi loại trùng văn bản;
nhãn mâu thuẫn hoặc dữ liệu quá nhỏ gây lỗi. Cùng seed, phiên bản và thứ tự
gọi sẽ có cùng split. Corpus lành [tự biên soạn](data/benign_corpus/README.md)
giữ riêng để đánh giá báo sai, có nguồn `student-curated, not scraped`.

Phiên bản thư viện được ghim trong `pyproject.toml`; `uv.lock` cố định phần
phụ thuộc bắc cầu cho `uv run`. Không cài các extra `transformer`, `demo`,
`translate` để chạy M0–M2. Cache của uv nằm trong `.uv-cache` của dự án.

## Baseline M2

```powershell
uv run python experiments/e01_baseline_classical_vs_transformer.py
Import-Csv results/tables/e01_classical.csv | Format-Table
# Tương đương: .\tasks.ps1 experiments hoặc make experiments
```

E01 dùng toàn bộ revision ghim, loại trùng và chia lại theo M1. Chỉ fit TF-IDF
và mô hình trên train; validation để riêng, không chỉnh tham số theo test.
TF-IDF ghép word n-gram và char n-gram; cấu hình và tham số chính xác nằm
trong metadata cạnh CSV. Mỗi mô hình được kiểm tra ngưỡng F1 khai báo trong
cột `f1_sanity_floor`; trạng thái nằm ở `sanity_pass`. Không dùng mẫu CI để
báo cáo hiệu quả. Kết quả đọc trực tiếp từ
[CSV E01](results/tables/e01_classical.csv), không sao chép số đo vào văn bản.

```python
from pidefense.models.classical import train_classical

# Dùng splits và rng của cùng một lần chạy; chỉ đưa train vào hàm fit.
detector = train_classical(splits.train, config.classical_model, rng.generator)
label, score = detector.predict("What is a binary tree?")
```

`Detector.predict` trả `(label, score)`; label là `benign`/`injection`, score
luôn hướng về lớp injection. Logistic Regression/Random Forest dùng
`predict_proba`; Linear SVM dùng sigmoid của margin, **chưa hiệu chỉnh xác
suất**. E01 dùng cùng quy tắc threshold cho dự đoán và tính các chỉ số.
Metadata ghi seed, seed estimator, cấu hình, commit và trạng thái working tree,
phiên bản, hash mã nguồn, nguồn dữ liệu và thành viên split dưới dạng ID/hash.

Chạy hồi quy ngoại tuyến (chỉ fixture, tên tệp riêng):

```powershell
uv run python experiments/e01_baseline_classical_vs_transformer.py --fixture
```

Fixture golden được tạo bằng lệnh trên và kiểm tra trong pytest, không đại
diện chất lượng mô hình. CLI tổng quát và lưu/nạp model triển khai để mốc sau.

## Baseline M3 (tùy chọn)

```powershell
uv sync --locked --extra dev --extra transformer
uv run --extra transformer python experiments/e01_transformer.py
Import-Csv results/tables/e01_transformer.csv | Format-Table
uv run --extra transformer pytest -m slow
```

M3 fine-tune toàn bộ DistilBERT trên CPU, có giới hạn epoch trong cấu hình.
Revision trọng số mặc định được ghim trong mã; `--revision` cho phép tái chạy
revision ghi trong metadata. Lần đầu cần mạng để tải mô hình và dữ liệu;
`--local-files-only` yêu cầu mô hình có sẵn trong cache. Máy cần đủ RAM cho
trọng số, gradient và trạng thái AdamW. Không cần Trainer hay accelerate.

E01 transformer dùng cùng split và công thức precision/recall/F1/ROC-AUC với
M2; validation để riêng và không điều chỉnh theo test. CSV và metadata cạnh
nó lưu kết quả, seed torch rút từ Generator, phiên bản, revision, hash nguồn,
thành viên split, checkpoint và timing từng lần. Checkpoint native Hugging
Face nằm cạnh bảng trong thư mục `e01_transformer_model` (không đưa vào Git).
Nạp lại bằng `pidefense.models.transformer.load_transformer(Path(...))`.

Độ trễ dùng `predict` từng câu, bao gồm tokenize, loại trừ tải/huấn luyện và
warmup, cố định CPU hai thread. CSV chứa mean/median/p95; metadata giữ mẫu
timing để E5 dùng sau. Chưa có CI thống kê hoặc so sánh latency với classical.
Timing thay đổi giữa các lần chạy; chỉ tái lập trọng số/score trong cùng môi
trường, seed, revision và thứ tự gọi, không cam kết bitwise giữa phần cứng.

`--fixture` chạy mẫu CI, bắt buộc tên bảng riêng và không đại diện hiệu quả.
Kiểm thử slow tạo mạng DistilBERT rất nhỏ trên máy, không tải checkpoint;
nó kiểm tra huấn luyện, tái lập và save/load, không chứng minh chất lượng
baseline pretrained. Kiểm thử mặc định mô phỏng không có heavy dependencies
và vẫn chạy classical; M0–M2 không cần cài extra transformer.

## Đánh giá M5: bypass và báo sai

```powershell
# Classical standalone, không cần torch:
uv run python experiments/e02_obfuscation_bypass_rate.py
# Thêm checkpoint M3 đã huấn luyện và metadata gốc để so sánh hai họ model:
uv run --extra transformer python experiments/e03_benign_false_positive_rate.py --checkpoint results/tables/e01_transformer_model
# Hồi quy ngoại tuyến; không ghi đè tên bảng khoa học:
uv run python experiments/e03_benign_false_positive_rate.py --fixture --n-bootstrap 30
```

Hai entry point E02/E03 cùng chạy cả hai phép đo để giữ nguyên model và ngưỡng.
Khi dữ liệu đầy đủ đã có trong cache, có thể đọc Arrow hoàn toàn ngoại tuyến;
runner xác minh toàn bộ ID/nhãn/hash văn bản với metadata E01:

```powershell
$revision = '4f61ecb038e9c3fb77e21034b22511b523772cdd'
$cachePath = "data/raw/hf_cache/deepset___prompt-injections/default/0.0.0/$revision"
uv run python experiments/e02_obfuscation_bypass_rate.py --checkpoint results/tables/e01_transformer_model --prepared-cache $cachePath
```

Ngưỡng được chọn bằng F1 validation; xem
[giải thích tự sinh](results/tables/e03_operating_thresholds.md),
[bypass theo technique](results/tables/e02_bypass_rate.csv) và
[FPR corpus lành](results/tables/e03_fpr.csv). Điểm PR, score theo ID và toàn bộ
ứng viên ngưỡng có CSV; PNG/PDF ở `results/figures`, xuất 300 dpi.
Dịch thiếu backend được bỏ qua rõ ràng, không giả lập kết quả dịch.
Multi-turn chỉ đo giả định guard chặn từng lượt, chưa gọi chatbot.

## Middleware M6

`ClassifierGuard` nhận detector đã huấn luyện (classical hoặc transformer),
không cần corpus tấn công. Ví dụ ghép với detector đang có trong Python:

```python
from pidefense.config import MiddlewareConfig
from pidefense.demo.ollama_client import OllamaClient
from pidefense.middleware.classifier_guard import ClassifierGuard
from pidefense.middleware.pipeline import GuardedChat

cfg = MiddlewareConfig()
# detector là đối tượng Detector đã fit; có thể truyền ngưỡng validation M5.
guard = ClassifierGuard(detector, threshold=cfg.guard_threshold)
chat = GuardedChat(guard, OllamaClient(cfg.ollama_host, cfg.ollama_model))
turn = chat.send("Tóm tắt tài liệu này.", context_document="Nội dung tham chiếu.")
# turn.status: allowed / blocked / error; chỉ allowed có turn.response.
# Không ghi turn.response hoặc nội dung nguồn vào log.
```

Ollama phải chạy cục bộ và đã có model cấu hình. Client chỉ chấp nhận loopback,
tắt proxy/redirect; host thiếu hoặc response lỗi trả trạng thái error qua
pipeline. Không cần extra demo cho HTTP client này. Middleware kiểm tra riêng
user/document, HTML-escape dữ liệu trong tag cố định và áp dụng output filter
trước khi trả response. Quy ước delimiter và giới hạn heuristic nằm trong
[quyết định M6](docs/decisions.md#quyết-định-m6).

Kiểm thử M6 mặc định dùng TF-IDF thật với HTTP giả lập, không gọi model sống:

```powershell
uv run pytest tests/integration/test_middleware.py tests/unit/test_log_hygiene.py --no-cov
uv run ruff check .
uv run mypy --strict src
uv run pytest
```

## Red-team M7: Ollama cục bộ với middleware OFF/ON

Khởi động Ollama riêng với model trong cấu hình (mặc định `phi3:mini`) đã có
trên máy, rồi chạy từ thư mục gốc bằng PowerShell:

```powershell
uv run python experiments/e04_indirect_injection_vs_output_filter.py
# Hoặc dùng Arrow cache đầy đủ đã được E01 xác minh, không cần gọi Hub:
$revision = '4f61ecb038e9c3fb77e21034b22511b523772cdd'
$cachePath = "data/raw/hf_cache/deepset___prompt-injections/default/0.0.0/$revision"
uv run python experiments/e04_indirect_injection_vs_output_filter.py --prepared-cache $cachePath
```

Runner dùng toàn bộ nguồn injection held-out và đủ kỹ thuật, không subsample.
Hai chế độ chia sẻ biến thể/seed; ON áp dụng toàn bộ GuardedChat, OFF dùng
system cơ bản cùng điều cấm lộ system/override. Multi-turn giữ lịch sử riêng
cho từng trial; indirect có user request/document tách biệt khi ON.

`results/e04_trials.json` chứa transcript gốc, chỉ để đánh giá được cho phép.
`results/tables/e04_indirect_injection_vs_output_filter.csv` giữ rate/count
theo technique và chế độ. Metadata cạnh CSV giữ config, seed, commit/dirty,
phiên bản, hash mã/đầu ra và nguồn split. `automated_raw_violation_rate` gồm
đầu ra bị giữ lại; `automated_final_violation_rate`/`automated_success_rate`
chỉ xét phản hồi trả cho người dùng, kể cả lượt trung gian. Chúng là heuristic,
không chứng minh tấn công thành công. Error/skipped không vào mẫu số completed;
rate không xác định để trống. CLI trả mã lỗi khi có trial lỗi backend/layer.

Dịch mặc định skipped nếu chưa có backend thật. Cung cấp `--translation-cache
data/local_translation_steps.json` để dùng bản dịch cục bộ đã chuẩn bị: JSON
là danh sách object có `text`, `source_lang`, `target_lang`, `translation`,
chứa cả bước en→vi và vi→en, key text khớp chính xác. Thiếu entry/không đổi
được ghi skipped, không thay thế bằng bản dịch giả hoặc tính thành an toàn.
Tệp chứa raw payload phải ở `data/` hoặc `results/`. Cần tự duyệt chất lượng
dịch; cache chỉ xác minh định dạng, không chứng minh bảo toàn ngữ nghĩa.

Đánh giá thủ công: đọc rubric trong metadata và transcript, điền bản sao
`results/e04_manual_review.csv` với `manual_violation=yes/no/uncertain`,
`reviewer`, `reason`, `sanitized_prompt`, `sanitized_response`. Giữ nguyên
trial ID/hash. Chỉ điền verdict cho completed trial có đầu ra user-visible.
Các trial bị chặn mà chưa trả gì là nonviolating theo cấu trúc, không tính
là người đã duyệt. Nhãn pending/uncertain và coverage được báo riêng.
Không ghi đè review đã có; nếu chạy lại với transcript khác, tạo template
mới trong output directory khác. Tái tính mà không gọi LLM:

```powershell
uv run python experiments/e04_indirect_injection_vs_output_filter.py --report-from results/e04_trials.json --reviews results/e04_manual_review.csv --appendix docs/bao_cao.md
```

Phụ lục tự sinh che toàn bộ raw text, giữ dạng biến đổi/trạng thái/tín hiệu;
ví dụ do reviewer làm sạch và lý do bất đồng được bổ sung từ CSV. Người
duyệt phải che dữ liệu riêng, đích và system trước khi điền. Không diễn giải
phần chưa duyệt như đồng thuận. Đây là nghiên cứu, không phải bảo đảm an toàn.

Kiểm thử offline và fixture tổng hợp (không cần model hay dịch thật):

```powershell
uv run python experiments/e04_offline_fixture.py
uv run pytest tests/integration/test_redteam.py --no-cov
```

Fixture luôn có hậu tố `_fixture`, metadata `fixture_only` và transport
`injected_mock`; canned replies/phép dịch giả chỉ kiểm tra luồng, tuyệt đối
không dùng để kết luận hiệu quả trên Ollama. Script fixture cập nhật phụ lục
với nhãn mô phỏng rõ ràng. `--fixture` trên runner live chỉ giảm corpus;
nó vẫn gọi Ollama, khác script offline.

Kiểm thử live được đánh dấu slow và opt-in rõ ràng; không chạy trong CI mặc định:

```powershell
$env:PIDEFENSE_RUN_LIVE_OLLAMA = '1'
$env:PIDEFENSE_PREPARED_CACHE = $cachePath
# Tùy chọn: $env:PIDEFENSE_TRANSLATION_CACHE = 'data/local_translation_steps.json'
uv run pytest tests/integration/test_redteam.py -m slow --no-cov
```

## M8: robustness, latency và báo cáo tái lập

Chạy nhanh hoàn toàn ngoại tuyến bằng PowerShell (không cần torch/Ollama):

```powershell
uv run python experiments/run_all.py --quick
# Tương đương helper Windows:
.\tasks.ps1 quick
```

Quick chạy E01, E02/E03, E04 mô phỏng và E05/E06 trên fixture thật đã commit.
Kết quả nhỏ giữ hậu tố `_fixture` hoặc `_quick`, không thay thế kết quả tập
đầy đủ. E04 dùng canned replies và dịch tổng hợp, không đánh giá LLM/dịch thật.
`results/tables/run_all_quick.json` ghi trạng thái từng stage và provenance;
`docs/bao_cao.md` tự lắp bảng/hình và phần thiếu, gồm checklist hạn chế và ghi
chú Copilot. Số đo không được gõ bằng tay trong báo cáo.

Chạy tập đầy đủ từ nguồn/cache ghim, rồi thêm checkpoint/live nếu đã có:

```powershell
uv run python experiments/run_all.py --prepared-cache $cachePath
# Tùy chọn, cần extra transformer đã cài và checkpoint M3 hợp lệ:
uv run --extra transformer python experiments/run_all.py --prepared-cache $cachePath --checkpoint results/tables/e01_transformer_model
# Tùy chọn gọi demo Ollama cục bộ, sau đó review thủ công theo hướng dẫn M7:
uv run python experiments/run_all.py --prepared-cache $cachePath --live-ollama
# Có thể thêm --translation-cache data/local_translation_steps.json khi có bản dịch thật.
```

Full runner không tự tải/huấn luyện transformer hoặc bật chatbot sống.
M3 vẫn dùng entry point riêng ở trên; `--checkpoint` xác minh split/config/hash
trước khi đánh giá. Khi không chọn checkpoint, CSV có hàng optional với metric
để trống. Không ghép timing M3 cũ và timing classical mới làm đối chứng.

Hai entry point E05/E06 dùng chung fit/threshold/RNG và sinh cả hai nhóm output:

```powershell
uv run python experiments/e05_latency_classical_vs_transformer.py --quick
uv run python experiments/e06_adversarial_robustness.py --quick
# Chỉ vẽ/lắp báo cáo từ dữ liệu đã lưu, không gọi LLM hoặc đo lại:
uv run python experiments/plotting.py
uv run python experiments/plotting.py --quick
uv run python experiments/report.py
```

E05 lưu từng timing/source/repeat trong `e05_latency_samples*.csv`; bảng
`e05_latency*.csv` chứa mean/median/p95 và CI mean bootstrap theo cụm input,
giữ toàn bộ lần lặp. Dùng cùng input và cấu hình CPU/thread giữa các model;
load/train/warmup nằm ngoài thời gian đo. CI không đại diện nhiễu hệ điều
hành, hardware khác hoặc train; timing thay đổi giữa các lần chạy. Chưa có
SLO/workload vận hành nên chưa kết luận phù hợp guard đồng bộ.

E06 lưu từng search trong `e06_search*.csv`, tổng hợp trong
`e06_robustness*.csv`; candidate thành công (nếu có) ở JSON dưới results/.
Budget = từ thay thế + từ thêm bởi preamble. Tìm theo chi phí tăng dần trong
family/query hữu hạn; minimum chỉ là nhỏ nhất đã quan sát. Nguồn ban đầu đã
lọt ghi riêng, không tính flip mới. Không thấy flip để trống minimum, không
coi là chứng minh robust. Đồng nghĩa có thể đổi ngữ nghĩa; classifier flip
chưa phải tấn công chatbot thành công và cần duyệt độc lập.

Manifest, metadata cạnh CSV và `report_data.json` cho phép truy nguồn số đo,
split, commit, phiên bản và hash. Các hình dùng chung style, có PNG/PDF;
renderer giữ provenance mới khi vẽ lại hình cũ, không thay số đo. Report
luôn phân biệt số đo tập đầy đủ, quick và mô phỏng. E04 live/nhận định người
duyệt vẫn cần hoàn thành riêng khi có Ollama, không được thay bằng fixture.

```powershell
uv run ruff check .
uv run mypy --strict src
uv run pytest
```

## Phạm vi và giới hạn

Đây là dự án nghiên cứu, chưa phải hệ thống an toàn dùng trong sản phẩm.
Ngay cả mô hình có chỉ số tổng hợp cao vẫn có thể bỏ lọt tấn công; không có
cam kết an toàn tuyệt đối. Chỉ kiểm thử demo cục bộ của dự án, không tấn công
dịch vụ chatbot bên thứ ba. `attacks/` chỉ phục vụ đánh giá được cho phép;
không thuộc thành phần guardrail triển khai. Không ghi payload hoặc phản hồi
chatbot thô vào log ngoài `data/` và `results/`.

Các quyết định thiết kế nằm trong [docs/decisions.md](docs/decisions.md).
