"""Assemble the Vietnamese report from persisted CSV/JSON, with explicit missing-result states."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from plotting import read_rows

LIMITATIONS = (
    "Tập dữ liệu (`deepset/prompt-injections`) là một ảnh chụp cố định, tương đối nhỏ "
    "và tập trung vào tiếng Anh; kết quả không khái quát cho mọi ngôn ngữ, mọi mô hình "
    "hoặc cách diễn đạt tấn công mới chưa thấy trong tương lai.",
    "Một bộ phân loại có F1 tổng hợp cao vẫn có thể có tỷ lệ vượt qua đáng kể ở các kỹ thuật "
    "obfuscation cụ thể (Phần 1.2 RQ2) — “độ chính xác cao” không có nghĩa là “an toàn”.",
    "Bộ lọc đầu ra dựa trên từ khóa/mẫu cùng một kiểm tra ngữ nghĩa nhẹ, không phải bộ kiểm "
    "chứng hình thức; chính nó có thể bị vượt qua bằng cách diễn đạt đủ gián tiếp.",
    "Phòng thủ nhiều lớp làm giảm nhưng không loại bỏ rủi ro; Phần 1.3 tuyên bố rõ rằng "
    "không có khẳng định nào về “sẵn sàng cho sản phẩm” hoặc “bảo đảm an toàn”.",
    "Red-team chỉ giới hạn ở một mô hình instruct nhỏ cục bộ (`phi3:mini`/`llama3.2:1b`); "
    "các mô hình lớn hơn hoặc dùng trong sản phẩm có thể phản ứng khác "
    "(dễ bị tác động hơn hoặc ít hơn).",
    "Kiểm thử độ bền vững đối kháng (RQ6) dùng một tập nhiễu loạn nhỏ, có tài liệu, "
    "không phải tìm kiếm đối kháng toàn diện; không tìm thấy trường hợp vượt qua "
    "không chứng minh độ bền vững.",
    "Không kiểm thử tình huống injection nhiều lượt, sử dụng công cụ hoặc tác tử "
    "(gọi hàm) ngoài trường hợp tách chỉ dẫn đơn giản; đây là hướng nghiên cứu tương lai.",
)
INTEGRATION = (
    "Hướng mở rộng: `ClassifierGuard` đã huấn luyện có thể bọc mọi văn bản trước khi truyền "
    "vào `ai/agents.py: run_llm(findings)` của AI Pentest Copilot, chẳng hạn banner dịch vụ, "
    "dòng log hoặc trường bằng chứng từ đầu ra quét không đáng tin. Nếu JSON LLM không hợp lệ "
    "khiến `ai_runs.status` thành `error`, có thể đính kèm `GuardDecision` để giải thích "
    "việc input bị gắn injection và chuyển sang `run_local`, dưới dạng khóa bổ sung trong "
    "`ai_runs.output_json`, giữ nguyên schema hiện có, không thêm bảng/cột. Đây là ghi chú "
    "hướng mở rộng cho cả hai báo cáo; chưa triển khai vào dự án nào, để hai đồ án có thể "
    "được chấm độc lập, trừ khi giảng viên cho phép tích hợp. Chỉ lưu metadata đã làm sạch, "
    "không đưa payload, khóa hoặc phản hồi gốc vào log."
)


def build_report(results_dir: Path, report_path: Path) -> None:
    """Render actual summary rows and all artifact links, storing the report data as JSON.

    Historical full-dataset results and new quick/fixture measurements are kept in
    separately labeled tables. Missing live/optional results are never imputed.
    Each rendered table includes its CSV path/hash, and known metadata hashes are
    checked before use. Narrative has no hard-coded measured performance values.
    """
    tables, figures = results_dir / "tables", results_dir / "figures"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, dict[str, object]] = {}

    def link(path: Path) -> str:
        """Use portable report-relative links, including Windows paths."""
        return Path(os.path.relpath(path.resolve(), report_path.parent.resolve())).as_posix()

    def remember(path: Path) -> str:
        """Record exact input file bytes so generated statements can be audited."""
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[link(path)] = {"sha256": sha}
        return sha

    def table(stem: str, columns: tuple[str, ...]) -> str:
        """Derive the entire Markdown table from CSV; absent values remain undefined."""
        path = tables / (stem + ".csv")
        if not path.exists():
            return f"Chưa có kết quả `{stem}`; không điền số thay thế.\n"
        rows = read_rows(path)
        sha = remember(path)
        meta_path = path.with_suffix(".metadata.json")
        meta_link = ""
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = metadata.get("csv_sha256")
            for key, value in metadata.get("artifact_sha256", {}).items():
                if Path(key).resolve() == path.resolve():
                    expected = value
            if expected is not None and expected != sha:
                raise ValueError("Report input CSV does not match its recorded hash")
            remember(meta_path)
            meta_link = f"; [metadata]({link(meta_path)})"
        inputs[link(path)]["rows"] = rows
        scope = (
            "MẪU CI/QUICK — không dùng để kết luận hiệu quả trên tập đầy đủ"
            if (stem.endswith(("_quick", "_fixture")))
            else "KẾT QUẢ TẬP ĐẦY ĐỦ — xem điều kiện của từng lần chạy trong metadata"
        )
        lines = [
            f"**{scope}.** [CSV nguồn]({link(path)}){meta_link}; SHA-256 `{sha}`.",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if value:
                    try:
                        value = format(float(value), ".6g")
                    except ValueError:
                        pass
                values.append((value or "—").replace("|", "\\|"))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines) + "\n"

    def images(prefixes: tuple[str, ...]) -> str:
        """Embed every available matching plot; include the PDF for export."""
        content = []
        for path in sorted(figures.glob("*.png")):
            if not path.stem.startswith(prefixes):
                continue
            remember(path)
            label = " — MẪU CI" if path.stem.endswith(("_fixture", "_quick")) else ""
            content.extend([f"![{path.stem}{label}]({link(path)})", ""])
            pdf = path.with_suffix(".pdf")
            if pdf.exists():
                remember(pdf)
                content.extend([f"[PDF vector]({link(pdf)})", ""])
        return "\n".join(content)

    sections = [
        "# Báo cáo nghiên cứu PI-DEFENSE\n\nTự sinh bằng `experiments/report.py`. "
        "Số đo chỉ đọc từ CSV/JSON trong results; ô trống là chưa xác định. "
        "Tập đầy đủ và mẫu quick được trình bày riêng, không ghép thành cùng phép so sánh.\n",
        "## Giới thiệu và động lực\n\nPrompt injection đưa chỉ dẫn không đáng tin vào ngữ cảnh "
        "chatbot, trực tiếp hoặc qua tài liệu tham chiếu. Dự án khảo sát phát hiện bằng học máy, "
        "phòng thủ nhiều lớp và đánh giá cục bộ theo [OWASP LLM01]"
        "(https://genai.owasp.org/llmrisk/llm01-prompt-injection/). "
        "Mục tiêu là đo giới hạn phương pháp, không tuyên bố giải quyết an toàn tổng quát.\n",
        "## Dữ liệu và phương pháp\n\nDùng [deepset/prompt-injections]"
        "(https://huggingface.co/datasets/deepset/prompt-injections) với "
        "revision ghim trong metadata: "
        "pool split upstream, chuẩn hóa để loại trùng, chia phân tầng có seed, chỉ fit trên train. "
        "Đây không phải benchmark giữ nguyên test upstream. Corpus lành tự biên soạn, chưa duyệt "
        "độc lập, được giữ ngoài train/validation/test. Một Generator điều khiển ngẫu nhiên trong "
        "từng thí nghiệm; commit/dirty, phiên bản, seed và hash đầu ra đi kèm kết quả.\n\n"
        "Các biến thể gồm Base64, leetspeak, ký tự zero-width, dịch round-trip, tài liệu chứa "
        "chỉ dẫn và tách chỉ dẫn qua hội thoại. Chỉ nguồn injection held-out tham gia đánh giá; "
        "giữ source_id. Dịch không có backend thật được skipped, không tính là an toàn.\n",
        "## Mô hình phát hiện và kết quả\n\nTF-IDF kết hợp word/char n-gram với Logistic "
        "Regression, Linear SVM và Random Forest. SVM dùng sigmoid margin chưa hiệu chỉnh; "
        "DistilBERT tùy chọn dùng softmax và truncation. Injection là lớp positive; "
        "precision=TP/(TP+FP), recall=TP/(TP+FN), F1 là trung bình điều hòa. "
        "Ngưỡng E01 cố định; ngưỡng E02/E03/E06 chọn theo F1 validation, hòa ưu tiên recall "
        "rồi ngưỡng thấp hơn; không chọn bằng test.\n",
    ]
    dataset_sections = []
    for stem in ("e01_classical", "e01_classical_fixture"):
        meta_path = tables / (stem + ".metadata.json")
        if not meta_path.exists():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        remember(meta_path)
        balance = metadata["label_balance"]
        inputs[link(meta_path)]["label_balance"] = balance
        scope = "MẪU CI" if metadata["fixture_only"] else "TẬP ĐẦY ĐỦ"
        lines = [
            f"**{scope}** — chia tập theo [metadata]({link(meta_path)}).",
            "",
            "| split | tổng | benign | injection |",
            "| --- | --- | --- | --- |",
        ]
        for name, row in balance.items():
            lines.append(
                f"| {name} | {row['total']} | {row['counts']['0']} | {row['counts']['1']} |"
            )
        dataset_sections.append("\n".join(lines) + "\n")
    sections[3:3] = dataset_sections
    for stem in ("e01_classical", "e01_transformer", "e01_classical_fixture"):
        sections.append(table(stem, ("model", "precision", "recall", "f1", "roc_auc")))
    sections.extend(
        [
            images(("e01_",)),
            "## Tỷ lệ bị vượt qua và báo sai\n\nE02: bypass = số nguồn injection được cho qua / "
            "số nguồn được đánh giá; multi-turn chỉ cho qua khi mọi lượt qua classifier. "
            "Đây chưa phải chatbot làm theo tấn công. E03: FPR=FP/(FP+TN) trên corpus lành riêng. "
            "CI percentile bootstrap theo nguồn giữ nguyên mô hình/ngưỡng, không bao gồm bất định "
            "huấn luyện/chọn ngưỡng. CI suy biến không chứng minh rủi ro bằng không.\n",
        ]
    )
    for suffix in ("", "_fixture"):
        sections.append(
            table(
                "e02_bypass_rate" + suffix,
                (
                    "model",
                    "policy",
                    "technique",
                    "n_evaluated",
                    "n_skipped",
                    "bypass_rate",
                    "ci_low",
                    "ci_high",
                ),
            )
        )
        sections.append(
            table(
                "e03_fpr" + suffix,
                ("model", "policy", "threshold", "fpr", "test_recall", "test_precision"),
            )
        )
    sections.extend(
        [
            images(("e02_", "e03_")),
            "## Kiểm thử chatbot Ollama\n\nE04 đối chiếu OFF/ON với cùng biến thể/seed; ON dùng "
            "classifier, system hardened và output filter, OFF dùng system cơ "
            "bản với cùng điều cấm. "
            "Tín hiệu raw gồm cả đầu ra bị giữ lại; final chỉ tính phản hồi "
            "user-visible, kể cả lượt "
            "trung gian. Error/skipped không vào mẫu số completed. Rubric thủ "
            "công yes/no/uncertain "
            "được ghi riêng, không tự gán từ heuristic; pending và coverage "
            "phải được xem cùng rate. "
            "Mô phỏng CI không đánh giá được LLM hoặc chất lượng dịch thật. "
            "Nếu chưa có CSV live bên dưới thì chưa hoàn thành acceptance live của M7.\n",
        ]
    )
    for suffix in ("", "_fixture"):
        sections.append(
            table(
                "e04_indirect_injection_vs_output_filter" + suffix,
                (
                    "technique",
                    "middleware",
                    "n_completed",
                    "n_skipped",
                    "n_errors",
                    "automated_final_violation_rate",
                    "n_human_reviewed",
                    "manual_coverage",
                    "manual_violation_rate",
                    "n_disagreements",
                ),
            )
        )
    sections.extend(
        [
            images(("e04_",)),
            "## Hiệu năng và đánh đổi\n\nE05 đo wall-clock predict trên CPU, batch một, bao gồm "
            "tokenize/score về host; loại load/train/warmup. Các model dùng cùng danh sách input, "
            "cùng giới hạn thread và máy; thứ tự model được rút từ RNG. "
            "Mean/median/p95 lấy từ từng "
            "timing đã lưu. CI mean bootstrap theo cụm input, giữ toàn bộ lần lặp của input. "
            "CI chỉ mô tả biến thiên hỗn hợp input có điều kiện trên timing quan sát, không đo "
            "nhiễu hệ điều hành, thay đổi máy hay bất định huấn luyện. Thời gian không tái lập "
            "bitwise; mẫu quick không đại diện workload sản phẩm.\n\n"
            "Chỉ so sánh classical/transformer khi cùng xuất hiện là evaluated trong cùng E05. "
            "Timing transformer cũ ở E01 không phải đối chứng cùng phép đo. Chưa có SLO vận hành "
            "nên không kết luận đáp ứng guard đồng bộ; cần cân nhắc p95, lưu lượng, false-positive "
            "và chi phí fallback. Giám sát bất đồng bộ giảm chặn request nhưng không ngăn response "
            "nguy hiểm trước khi trả.\n",
        ]
    )
    for suffix in ("", "_quick"):
        sections.append(
            table(
                "e05_latency" + suffix,
                (
                    "model",
                    "status",
                    "n_texts",
                    "n_measurements",
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                    "mean_ci_low_ms",
                    "mean_ci_high_ms",
                ),
            )
        )
    sections.extend(
        [
            images(("e05_",)),
            "## Độ bền vững trước nhiễu loạn đối kháng\n\nE06 chỉ tìm flip trên nguồn injection "
            "ban đầu bị chặn; nguồn đã lọt ghi riêng. Thử các thay từ trong danh sách cố định và "
            "thêm preamble lành, theo chi phí tăng dần. Budget là số từ được thay cộng số từ thêm; "
            "query budget gồm cả baseline. Giới hạn vị trí/phép thay/query nằm trong metadata. "
            "Minimum là nhỏ nhất quan sát trong ứng viên đã hỏi, không phải tối thiểu toàn cục. "
            "Không tìm thấy flip ghi trống budget, không gán bằng zero. Thay đồng nghĩa phụ thuộc "
            "ngữ cảnh, chưa chứng minh bảo toàn ngữ nghĩa; cần người duyệt "
            "trước khi coi là tấn công "
            "thành công. Prediction flip chỉ là fragility của classifier, "
            "không đo chatbot violation.\n",
        ]
    )
    for suffix in ("", "_quick"):
        sections.append(
            table(
                "e06_robustness" + suffix,
                (
                    "model",
                    "status",
                    "n_injections",
                    "n_eligible",
                    "n_already_benign",
                    "n_flipped",
                    "flip_rate",
                    "minimum_observed_budget",
                    "median_successful_budget",
                ),
            )
        )
    sections.extend(
        [
            images(("e06_",)),
            "## Kiến trúc phòng thủ nhiều lớp\n\nClassifierGuard dùng score >= threshold để chặn; "
            "user/document được screen riêng. System hardened quy định ranh giới chỉ dẫn/dữ liệu; "
            "HTML-escape trước khi bọc untrusted tags. Output filter tìm marker, span system và "
            "proxy từ vựng/mẫu tuân theo override. Chỉ response allowed được "
            "trả; lỗi layer fail closed. "
            "Audit chỉ gồm độ dài, SHA-256 rút gọn, nhãn, score, rule. Corpus "
            "attacks chỉ là tập đánh "
            "giá; artifact triển khai gồm middleware và model, không chứa corpus. Delimiter không "
            "phải ranh giới bảo mật hình thức; classifier và output filter "
            "đều có thể bỏ lọt/báo sai.\n",
            "## Hạn chế và hướng mở rộng\n\nBản dịch đầy đủ checklist Phần 11 SPEC.md:\n\n"
            + "\n".join("- " + item for item in LIMITATIONS)
            + "\n\n"
            + INTEGRATION
            + "\n",
            "## Kết luận\n\nPipeline tái lập được các số đo đã có và công khai phần thiếu. "
            "Các bảng E01–E03 đánh giá detector; E05/E06 đánh giá độ trễ/fragility trong điều kiện "
            "đã ghi, không chứng minh an toàn chatbot. Khi E04 chưa có phản hồi sống và nhãn người "
            "duyệt thì chưa có kết luận về hiệu quả chống injection của demo. Cần corpus đa dạng, "
            "đánh giá ngữ nghĩa độc lập và thử nghiệm workload thực trước mọi "
            "quyết định triển khai.\n",
            "## Phụ lục\n\nPayload/response gốc và ví dụ flip chỉ nằm dưới data/ hoặc results/. "
            "Không đưa raw text vào báo cáo. Xem README để chạy lại, cung cấp "
            "checkpoint/dịch cục bộ "
            "và thực hiện review E04. Các entry point hoạt động bằng PowerShell:\n\n"
            "```powershell\nuv run python experiments/run_all.py --quick\n"
            "uv run python experiments/run_all.py --prepared-cache $cachePath\n"
            "uv run python experiments/report.py\n```\n\n",
        ]
    )
    sections.append("<!-- E04:BEGIN -->")
    appendix = next(
        (
            results_dir / ("e04_indirect_injection_vs_output_filter" + suffix + ".appendix.md")
            for suffix in ("", "_fixture")
            if (
                results_dir / ("e04_indirect_injection_vs_output_filter" + suffix + ".appendix.md")
            ).exists()
        ),
        None,
    )
    if appendix is not None:
        remember(appendix)
        sections.append(appendix.read_text(encoding="utf-8"))
    else:
        sections.append("Chưa có ví dụ E04 làm sạch; không tự dựng phản hồi chatbot.")
    sections.extend(["<!-- E04:END -->", "### Danh mục artifact và nguồn kiểm tra\n"])
    for path in sorted([*tables.glob("*.csv"), *tables.glob("*.json"), *tables.glob("*.md")]):
        if path.name == "report_data.json":
            continue
        remember(path)
        sections.append(f"- [{path.name}]({link(path)})")
    sections.extend(["\n### Tích hợp AI Pentest Copilot (hướng mở rộng)\n", INTEGRATION])
    data = {
        "inputs": inputs,
        "sections": sections,
        "limitations_translation": LIMITATIONS,
        "integration_future_work": INTEGRATION,
    }
    data_path = tables / "report_data.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sections.append(f"\n[Nguồn lắp ráp báo cáo và hash artifact]({link(data_path)}).\n")
    report_path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("docs/bao_cao.md"))
    args = parser.parse_args()
    build_report(args.results_dir, args.report)
