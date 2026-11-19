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

## Phạm vi và giới hạn

Đây là dự án nghiên cứu, chưa phải hệ thống an toàn dùng trong sản phẩm.
Ngay cả mô hình có chỉ số tổng hợp cao vẫn có thể bỏ lọt tấn công; không có
cam kết an toàn tuyệt đối. Chỉ kiểm thử demo cục bộ của dự án, không tấn công
dịch vụ chatbot bên thứ ba. `attacks/` chỉ phục vụ đánh giá được cho phép;
không thuộc thành phần guardrail triển khai. Không ghi payload hoặc phản hồi
chatbot thô vào log ngoài `data/` và `results/`.

Các quyết định thiết kế nằm trong [docs/decisions.md](docs/decisions.md).
