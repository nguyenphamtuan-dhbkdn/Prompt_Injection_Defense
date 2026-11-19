# Rules and Testing Strategy for Coding Agent

## Environment Note (Windows)
- The host OS is **Windows (PowerShell)**.
- In addition to Makefile, always run verification commands directly via `uv run` (e.g., `uv run pytest`, `uv run ruff check .`, `uv run mypy --strict src`) or provide a `tasks.ps1` helper script so commands work natively on Windows PowerShell without requiring GNU make.

## 0. Rules for the coding agent (read first)

1. **Work milestone by milestone** (Section 5 in SPEC.md). Do not start milestone N+1 until all acceptance criteria of N pass (`pytest`, `ruff`, `mypy`).
2. **Never fabricate results.** Every number, table, and figure in `docs/` and `results/` must be produced by code in `experiments/` and stored as CSV/JSON in `results/`. No hard-coded "expected" metrics in reports.
3. **Reproducibility.** All randomness (train/test split, bootstrap, obfuscation sampling) goes through a single `numpy.random.Generator` created from a seed in the config. Every experiment writes its config + seed + git commit hash + library versions next to its output.
4. **Never log or persist raw attack payloads or raw chatbot outputs in plaintext logs outside `data/` and `results/`.** Demo/middleware logs store only: payload length, a SHA-256 fingerprint (first 8 hex chars), predicted label, confidence score, and the rule/model that fired.
5. **Separate "test payloads for evaluation" from "operational guardrail."** The attack payload corpus (`attacks/`) exists to *measure* the defense, not to be shipped as part of the product. The deployable artifact is only `middleware/` + the trained model file. Every file under `attacks/` must carry an authorized-testing-only scope comment at the top.
6. **Type hints everywhere, docstrings with the method/formula used (e.g., TF-IDF weighting, confusion-matrix definitions, bootstrap CI), `ruff` + `mypy --strict` clean.**
7. **Do not invent API details** of `scikit-learn`, `transformers`, `datasets`, `ollama`, etc. Check the installed version's docs (`help()`, source) before use. Do not use removed/renamed APIs. Pin versions in `pyproject.toml`.
8. If a requirement is ambiguous, choose the documented default, write the decision in `docs/decisions.md`, and continue. Do not stop to ask.
9. **Be honest in docs about limits:** this is a **research/evaluation project**, not a production-grade safety system. A classifier with high aggregate accuracy still lets some attacks through; document the residual risk explicitly (Section 11 in SPEC.md).
10. **No real external chatbot/API is attacked.** All red-teaming targets only the project's own local demo (Ollama model running on `localhost`), never a third-party production service.
11. **Optional heavy dependencies (`torch`, `transformers`) must be guarded by `try/except ImportError`** everywhere they are used, so the classical-ML pipeline (Milestones M0–M2, M4–M6) works standalone on a machine without GPU/torch installed.

## 8. Test strategy

- **Unit:** data loaders (split proportions, no leakage between train/val/test), each obfuscation transform (output differs from input, non-empty, traceable `source_id`), metrics math (precision/recall/F1/ROC-AUC) against hand-computed small examples.
- **Statistical tests:** bootstrap confidence intervals for precision/recall/F1 and for bypass rate per technique; fixed seeds; mark `@pytest.mark.statistical` and allow running them separately.
- **Property-based (hypothesis):** every obfuscation function (`base64_wrap`, `leetspeak`, `zero_width_insert`, `translate_roundtrip`) is total (never raises) on arbitrary non-empty ASCII/Unicode strings within a bounded length; `zero_width_insert` output length ≥ input length; `base64_wrap` output decodes back to exactly the original payload.
- **Negative tests:** empty string input to each transform, extremely long input (length guard triggers), malformed config (validation error with a clear message), `ClassifierGuard.check()` called with `None`/empty text, missing Ollama host (graceful error, not a crash).
- **Cross-model equivalence (where applicable):** classical and transformer detectors expose the same `Detector` interface; a shared test suite runs against both (transformer path marked `@pytest.mark.slow`, skipped by default if `torch` is not installed).
- **Integration tests:** a fixed list of known-injection and known-benign prompts run through `GuardedChat`, asserting the expected allow/block decision for each; grep-based security-hygiene test asserting no module logs raw payload text (mirrors "never log full keys" from Rule 4, adapted to "never log raw payload text").
- **Regression of experiment outputs:** store small golden CSVs for tiny configs (e.g., a 50-row subsample) so CI stays fast; full-dataset runs are a separate `make experiments` target, not part of default CI.
- **Coverage target:** ≥ 80% on `src/pidefense`.
- **Slow/optional tests:** anything requiring `torch`/`transformers` or a live Ollama instance is marked `@pytest.mark.slow` and skipped by default in CI, runnable explicitly via `make test-slow`.
- **Security-hygiene tests:** assert no module logs raw payload text or raw chatbot responses (grep-based test over `middleware/` and `demo/`); assert every file under `attacks/` contains the authorized-testing-only scope comment.
