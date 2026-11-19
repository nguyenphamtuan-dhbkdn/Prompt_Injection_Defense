# Mẫu dữ liệu ngoại tuyến

`deepset_sample.csv` là tập con nguyên văn của `deepset/prompt-injections`;
không tự viết hoặc thay đổi nhãn. Chỉ dùng cho kiểm thử được cho phép và CI,
không dùng để kết luận hiệu quả phát hiện hoặc độ an toàn.

Nguồn: https://huggingface.co/datasets/deepset/prompt-injections
Revision được ghim trong `deepset_sample.metadata.json` và loader. `source_id`
giữ tên split và chỉ số hàng gốc (đếm từ không). Bộ tạo duyệt tên split theo
thứ tự từ điển, giữ các văn bản duy nhất đầu tiên của từng nhãn đến giới hạn
trong mã; không lấy mẫu ngẫu nhiên. Payload chỉ nằm trong thư mục dữ liệu.

Tái tạo từ PowerShell tại gốc dự án (cần mạng cho lần tải đầu):

```powershell
$env:HF_HOME = Join-Path $PWD 'data/raw/hf_home'
uv run python experiments/prepare_m1_sample.py
```

Metadata ghi phương pháp chọn, revision, phiên bản thư viện, commit hiện hành
và checksum CSV. Seed là `null` vì việc chọn fixture không dùng ngẫu nhiên.
Git giữ nguyên byte của CSV để checksum không đổi theo hệ điều hành.

Dataset card tại revision này khai báo `license: apache-2.0` ở cấp ngoài,
nhưng có `license: cc-by-4.0` trong `dataset_info`; giữ nguyên ghi nhận sự
không nhất quán này, không diễn giải thành giấy phép thống nhất. Xem
[dataset card gốc](https://huggingface.co/datasets/deepset/prompt-injections/blob/4f61ecb038e9c3fb77e21034b22511b523772cdd/README.md).
Các cache/download khác trong thư mục này không đưa vào Git.
