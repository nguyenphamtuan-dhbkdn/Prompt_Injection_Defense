# PI-DEFENSE

Dự án nghiên cứu và đánh giá phòng thủ trước prompt injection. Hiện chỉ có
M0–M1: cấu hình YAML, bộ sinh số ngẫu nhiên có seed, tải dữ liệu và chia tập
phân tầng. Các mô-đun mô hình, tấn công, middleware và thí nghiệm đánh giá
vẫn là vị trí dành sẵn; chưa có huấn luyện hay kết quả đánh giá.

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
Các đích `experiments`, `figures`, `report`, `demo` báo chưa triển khai ở M0.
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
`translate` để chạy M0. Cache của uv nằm trong `.uv-cache` của dự án.

## Phạm vi và giới hạn

Đây là dự án nghiên cứu, chưa phải hệ thống an toàn dùng trong sản phẩm.
Ngay cả mô hình có chỉ số tổng hợp cao vẫn có thể bỏ lọt tấn công; không có
cam kết an toàn tuyệt đối. Chỉ kiểm thử demo cục bộ của dự án, không tấn công
dịch vụ chatbot bên thứ ba. `attacks/` chỉ phục vụ đánh giá được cho phép;
không thuộc thành phần guardrail triển khai. Không ghi payload hoặc phản hồi
chatbot thô vào log ngoài `data/` và `results/`.

Các quyết định thiết kế nằm trong [docs/decisions.md](docs/decisions.md).
