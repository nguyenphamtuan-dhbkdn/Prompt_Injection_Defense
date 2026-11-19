# PI-DEFENSE: Detection and Defense-in-Depth against Prompt Injection in LLM Chatbots (ML baseline + Transformer + Red-Team Harness + Middleware)

> **How to use this file.** Put it in the repo root as `SPEC.md` (and copy Section 0 + Section 15 into `AGENTS.md`). Give Codex one milestone at a time (Section 13). Every milestone has acceptance criteria that must pass before moving on.
> The spec is written in English for precision. All end-user documentation and the final report are written in **Vietnamese** (see Section 12).

---

## 0. Rules for the coding agent (read first)

1. **Work milestone by milestone** (Section 13). Do not start milestone N+1 until all acceptance criteria of N pass (`pytest`, `ruff`, `mypy`).
2. **Never fabricate results.** Every number, table, and figure in `docs/` and `results/` must be produced by code in `experiments/` and stored as CSV/JSON in `results/`. No hard-coded "expected" metrics in reports.
3. **Reproducibility.** All randomness (train/test split, bootstrap, obfuscation sampling) goes through a single `numpy.random.Generator` created from a seed in the config. Every experiment writes its config + seed + git commit hash + library versions next to its output.
4. **Never log or persist raw attack payloads or raw chatbot outputs in plaintext logs outside `data/` and `results/`.** Demo/middleware logs store only: payload length, a SHA-256 fingerprint (first 8 hex chars), predicted label, confidence score, and the rule/model that fired. This mirrors a real deployment where full user input should not sit unredacted in application logs.
5. **Separate "test payloads for evaluation" from "operational guardrail."** The attack payload corpus (Section 4) exists to *measure* the defense, not to be shipped as part of the product. Clearly mark `attacks/` as a research/test module; the deployable artifact is only `middleware/` + the trained model file.
6. **Type hints everywhere, docstrings with the method/formula used (e.g., TF-IDF weighting, confusion-matrix definitions, bootstrap CI), `ruff` + `mypy --strict` clean.**
7. **Do not invent API details** of `scikit-learn`, `transformers`, `datasets`, `ollama`, etc. Check the installed version's docs (`help()`, source) before use. Pin versions in `pyproject.toml`.
8. If a requirement is ambiguous, choose the documented default, write the decision in `docs/decisions.md`, and continue. Do not stop to ask.
9. **Be honest in docs about limits:** this is a **research/evaluation project**, not a production-grade safety system. A classifier with 95% accuracy still lets attacks through; document the residual risk explicitly (see Section 11).
10. **No real external chatbot/API is attacked.** All red-teaming targets only the project's own local demo (Ollama model running on `localhost`), never a third-party production service.

---

## 1. Project goals

### 1.1 Primary goal

Build a modular, tested, reproducible Python project that:

1. Loads and explores the **deepset/prompt-injections** dataset (benign vs. injection-labeled prompts) plus an optional secondary corpus assembled from OWASP LLM01 examples for qualitative coverage of attack *types* (direct injection, indirect injection, jailbreak/role-play, obfuscation).
2. Trains and compares **two families of detectors**:
   - Classical ML baseline: TF-IDF (word + char n-gram) features + Logistic Regression / Linear SVM / Random Forest.
   - Transformer baseline (optional extra, guarded by `try/except ImportError`): fine-tuned `distilbert-base-uncased` (or a smaller distilled model) for binary classification.
3. Builds an **attack/obfuscation generator** (`attacks/`) that takes a labeled injection string and produces variants: base64-wrapping, leetspeak, whitespace/zero-width-character insertion, translation round-trip, instruction-splitting across turns, and indirect injection (payload embedded inside a "retrieved document" that the chatbot is told to summarize).
4. Runs a **red-team evaluation harness** against a local demo chatbot (Ollama) with and without the defense middleware enabled, measuring how many obfuscated/indirect payloads still cause the chatbot to violate its system prompt.
5. Implements a **defense-in-depth middleware** (`middleware/`): (a) input classifier guard (the trained model), (b) a hardened system-prompt template with explicit instruction-data separation, (c) an output filter that checks the chatbot's response for leaked system-prompt content or compliance with disallowed instructions.
6. Evaluates detection quality (precision/recall/F1/ROC-AUC), **bypass rate** per obfuscation technique, **latency** (classical vs transformer), and **adversarial robustness** of the classifier itself (can a paraphrase/obfuscation flip its prediction?).
7. Documents an **integration path** into the existing `AI Pentest Copilot` project (Section 11.3): the trained classifier as a reusable guard on any text that will be passed to `run_llm(...)`.
8. Produces all experiments, figures, and tables needed for a university report in Vietnamese.

### 1.2 Research questions the experiments must answer

- **RQ1:** What is the baseline detection performance (precision/recall/F1/ROC-AUC) of TF-IDF+classical-ML vs. fine-tuned DistilBERT on the held-out test split of `deepset/prompt-injections`?
- **RQ2:** How does detection accuracy degrade per obfuscation technique (base64, leetspeak, zero-width chars, translation round-trip, instruction-splitting)? Which technique yields the highest **bypass rate**?
- **RQ3:** What is the **false-positive rate** on a benign-query corpus (ordinary chatbot questions), and how does it trade off against recall (precision-recall curve, chosen operating threshold)?
- **RQ4:** Does **indirect injection** (payload hidden inside a document the chatbot is asked to summarize/translate) bypass the input-classifier guard at a higher rate than direct injection, and does the output filter catch what the input guard misses?
- **RQ5:** What is the **inference latency** of classical ML vs. transformer detectors, and what does that imply for using either as a synchronous request-blocking guard vs. an asynchronous monitoring signal?
- **RQ6:** Is the trained classifier itself **adversarially fragile** — can a small, semantically-preserving perturbation (synonym substitution, added benign preamble) flip a correctly-classified injection to "benign"? Quantify the minimum perturbation budget observed.

### 1.3 Non-goals

- No training or evaluation against a real third-party production chatbot/API without explicit written permission; all red-teaming is against the project's own local Ollama instance.
- No attempt to build a general-purpose jailbreak toolkit for distribution; obfuscation generators exist only to test the defense and must stay inside `attacks/` with a clear research/testing label and a top-of-file comment stating authorized-testing-only scope.
- No claim that the final classifier is "production-ready" or "guarantees safety" — Section 11 must state residual risk explicitly.
- No multi-turn conversational memory attack simulation beyond a simple 2–3 turn instruction-splitting test (full multi-turn red-teaming is listed as stretch/future work).

---

## 2. Tech stack and environment

- Python **3.11+**, `src/` layout, `pyproject.toml` (PEP 621), dependency management with `uv` or `pip`.
- Runtime deps: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `datasets` (Hugging Face, for loading `deepset/prompt-injections`), `pyyaml`, `typer` (CLI), `rich` (console), `tqdm`.
- Dev deps: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pre-commit`.
- Optional extras (`[project.optional-dependencies]`):
  - `transformer`: `torch`, `transformers` — guarded by `try/except ImportError` everywhere it is used, so the classical-ML pipeline works standalone without GPU/torch installed.
  - `demo`: `fastapi`, `uvicorn`, `httpx` (for the middleware demo and calling the local Ollama HTTP API).
  - `translate`: optional translation backend for the translation-round-trip obfuscation (must fail gracefully / be skippable offline).
- `Makefile` targets: `setup`, `test`, `lint`, `typecheck`, `experiments`, `figures`, `report`, `demo`.
- GitHub Actions CI (optional): lint + typecheck + tests (classical-ML path only, transformer path marked `@pytest.mark.slow` / skipped in CI by default).
- Local LLM for the demo chatbot: **Ollama** (same tool already used in the AI Pentest Copilot project), small instruct model (e.g. `llama3.2:1b` or `phi3:mini`) to keep RAM usage compatible with an 8 GB machine.

---

## 3. Repository layout

```
pi-defense/
├── pyproject.toml
├── Makefile
├── README.md                     # Vietnamese quick start
├── SPEC.md                       # this file
├── AGENTS.md                     # Section 0 + 10 (test strategy)
├── configs/
│   ├── default.yaml
│   ├── classical_tfidf.yaml
│   ├── transformer_distilbert.yaml
│   └── experiments/              # one yaml per experiment E1..E10
├── data/
│   ├── raw/                      # downloaded deepset/prompt-injections cache
│   ├── benign_corpus/            # curated benign queries for FPR testing (small, versioned)
│   └── owasp_examples/           # OWASP LLM01 qualitative examples, source-cited, labeled
├── src/pidefense/
│   ├── __init__.py
│   ├── config.py                 # dataclasses + YAML loading + validation
│   ├── rng.py                    # SimRNG (seeded) wrapper for all stochastic steps
│   ├── data/
│   │   ├── loader.py             # load_deepset_prompt_injections(), load_benign_corpus()
│   │   ├── splits.py             # stratified train/val/test split, seeded
│   │   └── schema.py             # PromptSample dataclass: text, label, source, technique
│   ├── features/
│   │   ├── tfidf.py              # word+char n-gram TF-IDF vectorizer builder
│   │   └── embeddings.py         # OPTIONAL: sentence-embedding features for classical models
│   ├── models/
│   │   ├── classical.py          # train_logreg(), train_svm(), train_random_forest()
│   │   ├── transformer.py        # OPTIONAL extra: fine-tune/load DistilBERT classifier
│   │   └── registry.py           # uniform predict(text) -> (label, score) interface for both families
│   ├── attacks/
│   │   ├── __init__.py           # top-of-file scope comment: authorized-testing-only
│   │   ├── direct.py             # pass-through of labeled injection strings (OWASP LLM01 direct)
│   │   ├── obfuscate.py          # base64_wrap(), leetspeak(), zero_width_insert(), translate_roundtrip()
│   │   ├── indirect.py           # embed_in_document(payload, carrier_template) -> "poisoned" document
│   │   └── multiturn.py          # instruction_splitting(payload, n_turns)
│   ├── middleware/
│   │   ├── classifier_guard.py   # ClassifierGuard.check(text) -> GuardDecision
│   │   ├── system_prompt.py      # hardened system-prompt template, instruction-data delimiter scheme
│   │   ├── output_filter.py      # check_response_for_leak_or_compliance(response, system_prompt) -> bool
│   │   └── pipeline.py           # GuardedChat: wraps Ollama call with input guard + output filter
│   ├── eval/
│   │   ├── metrics.py            # precision/recall/F1/ROC-AUC, bootstrap CI
│   │   ├── bypass_rate.py        # per-technique bypass-rate computation
│   │   ├── robustness.py         # adversarial perturbation search against the trained classifier
│   │   └── latency.py            # timing harness, classical vs transformer
│   ├── demo/
│   │   ├── ollama_client.py      # thin HTTP client for local Ollama
│   │   └── chatbot_demo.py       # CLI/FastAPI demo: chat with/without GuardedChat enabled
│   └── cli.py                    # typer CLI
├── experiments/
│   ├── e01_baseline_classical_vs_transformer.py
│   ├── e02_obfuscation_bypass_rate.py
│   ├── e03_benign_false_positive_rate.py
│   ├── e04_indirect_injection_vs_output_filter.py
│   ├── e05_latency_classical_vs_transformer.py
│   ├── e06_adversarial_robustness.py
│   ├── run_all.py
│   └── plotting.py               # consistent matplotlib/seaborn style, saves PNG+PDF (300 dpi)
├── tests/
│   ├── unit/                     # data loading, feature extraction, metrics math
│   ├── statistical/               # bootstrap CI coverage, seeded reproducibility
│   └── integration/              # full guard pipeline on a small fixed payload set
├── docs/
│   ├── decisions.md
│   ├── bao_cao.md                # Vietnamese final report skeleton
│   └── references.bib
└── results/
    ├── figures/
    └── tables/
```

---

## 4. Attack / obfuscation taxonomy (maps to OWASP LLM01 and to Section 1.2 RQs)

| Category | Technique | Module | Example transform |
|---|---|---|---|
| Direct injection | Plain override instruction | `attacks/direct.py` | `"Ignore previous instructions and ..."` (taken verbatim from labeled dataset rows, never authored from scratch for realism) |
| Obfuscated direct | Base64 wrapping | `attacks/obfuscate.py: base64_wrap` | Payload base64-encoded, chatbot asked to "decode and follow" |
| Obfuscated direct | Leetspeak / homoglyph substitution | `attacks/obfuscate.py: leetspeak` | `"1gn0r3 pr3v10us 1nstruct10ns"` |
| Obfuscated direct | Zero-width / invisible character insertion | `attacks/obfuscate.py: zero_width_insert` | Injects `\u200b` between characters to break tokenization/keyword matching |
| Obfuscated direct | Translation round-trip | `attacks/obfuscate.py: translate_roundtrip` | English → Vietnamese → English (or similar) to alter surface form while preserving intent |
| Indirect injection | Payload embedded in a "document to summarize" | `attacks/indirect.py: embed_in_document` | Simulates RAG/tool-use scenario: chatbot told to summarize a web page; page contains hidden instruction |
| Multi-turn | Instruction splitting | `attacks/multiturn.py: instruction_splitting` | Payload split across 2–3 turns so no single message looks malicious alone |

Every generated variant keeps a traceable link back to its source row in the dataset (`source_id`) and its `technique` label, so bypass rate can be reported **per technique**, not just in aggregate.

---

## 5. Milestones (give one to Codex at a time)

### M0 — Project skeleton
- Repo layout from Section 3, `pyproject.toml`, `Makefile`, config loader (`config.py`), seeded RNG wrapper.
- **Acceptance:** `make setup` succeeds; `pytest` runs (even if trivial); `ruff`/`mypy --strict` clean on the skeleton.

### M1 — Data loading and splits
- `load_deepset_prompt_injections()` (download/cache via `datasets`, fallback to a committed small local CSV sample for offline/CI use), `load_benign_corpus()` (curated by the student, min. 150 rows, diverse topics), stratified seeded train/val/test split.
- **Acceptance:** unit tests assert split proportions, label balance reporting, no leakage (same text not in two splits), deterministic given the same seed.

### M2 — Classical ML baseline
- TF-IDF (word 1–2 gram + char 3–5 gram) vectorizer, Logistic Regression / Linear SVM / Random Forest training, `models/registry.py` uniform `predict(text) -> (label, score)`.
- **Acceptance:** RQ1 baseline numbers produced into `results/tables/e01_classical.csv`; precision/recall/F1/ROC-AUC on held-out test split meet a documented sanity floor (e.g., F1 ≥ 0.80 — adjust once real numbers are seen, never hard-code the number itself into the report prose, always read from CSV).

### M3 — Transformer baseline (optional extra)
- Fine-tune DistilBERT (small epoch budget, CPU-feasible or short GPU run), guarded entirely by `try/except ImportError` so the rest of the project works without `torch`.
- **Acceptance:** same metrics as M2, reproducible given seed + logged library versions; latency measured and stored for RQ5.

### M4 — Attack/obfuscation generators
- Implement all techniques in Section 4 with unit tests verifying each transform is reversible/traceable (`source_id` preserved) and does not silently corrupt the payload's semantic intent (spot-checked, documented, not claimed to be perfectly validated).
- **Acceptance:** property-based tests (hypothesis) confirm `base64_wrap`/`leetspeak`/`zero_width_insert` always produce a string different from the input and a non-empty output for any non-empty input.

### M5 — Evaluation: bypass rate and false-positive rate
- `eval/bypass_rate.py`: run both trained detectors (M2, M3) against every obfuscated/indirect variant from M4; compute bypass rate per technique (RQ2).
- `eval/metrics.py` + `e03_benign_false_positive_rate.py`: compute FPR on the benign corpus, plot precision-recall curve, choose and document an operating threshold (RQ3).
- **Acceptance:** `results/tables/e02_bypass_rate.csv` and `e03_fpr.csv` populated; plots saved at 300 dpi.

### M6 — Middleware (defense-in-depth)
- `middleware/classifier_guard.py` wraps the trained model behind a stable `GuardDecision(label, score, technique_guess)` interface.
- `middleware/system_prompt.py`: hardened template with an explicit, documented instruction/data delimiter convention (e.g., untrusted content always wrapped in a fixed tagged block, with the system prompt explicitly instructing the model to never treat tagged-block content as instructions).
- `middleware/output_filter.py`: regex/keyword + lightweight semantic check for system-prompt leakage or compliance with a disallowed instruction pattern.
- `middleware/pipeline.py`: `GuardedChat` composing all three layers around a call to the local Ollama demo model.
- **Acceptance:** integration tests run a fixed set of known injection + known benign prompts through `GuardedChat` and assert expected block/allow decisions; no raw payload text appears in any log line (grep-based security-hygiene test, mirroring the QKD project's "never log full keys" rule, adapted to "never log raw payload text").

### M7 — Red-team harness against the local chatbot
- `experiments/e04_indirect_injection_vs_output_filter.py`: run the full obfuscation/indirect corpus against the Ollama demo **with middleware OFF** and **with middleware ON**; record whether the chatbot's final response violates the system prompt (manual rubric + automated output-filter signal, both reported, discrepancies discussed).
- **Acceptance:** CSV with per-technique success rate before/after middleware; at least one qualitative example of each technique's prompt and the chatbot's response (sanitized per Rule 4) included in `docs/bao_cao.md` appendix.

### M8 — Adversarial robustness of the classifier + latency + report
- `eval/robustness.py` (RQ6): a small, documented perturbation search (synonym substitution list, benign preamble prepending) against the trained classifier; report the smallest perturbation observed that flips a prediction.
- `eval/latency.py` (RQ5): timing harness comparing classical vs transformer inference latency under identical hardware, averaged over N runs with CI.
- Assemble all figures/tables into `docs/bao_cao.md` (Vietnamese), write `docs/decisions.md` for every ambiguous choice made along the way, write Section 11 honest-limitations checklist verbatim into the report.
- **Acceptance:** `make experiments report` reproduces every figure/table from a clean clone; every number in `docs/bao_cao.md` is traceable to a file in `results/`.

---

## 6. Key interfaces (guidance; keep names stable)

```python
# config.py
@dataclass(frozen=True)
class DataConfig: dataset_name: str; benign_corpus_path: Path; test_fraction: float; val_fraction: float
@dataclass(frozen=True)
class ClassicalModelConfig: vectorizer: Literal["tfidf_word", "tfidf_char", "tfidf_combo"]; classifier: Literal["logreg", "svm", "rf"]
@dataclass(frozen=True)
class TransformerModelConfig: base_model: str; max_length: int; epochs: int; batch_size: int
@dataclass(frozen=True)
class AttackConfig: techniques: list[Literal["base64", "leetspeak", "zero_width", "translate", "indirect", "multiturn"]]
@dataclass(frozen=True)
class MiddlewareConfig: guard_threshold: float; ollama_model: str; ollama_host: str

# data
def load_deepset_prompt_injections(cfg: DataConfig, rng: Generator) -> DatasetSplits
def load_benign_corpus(path: Path) -> list[PromptSample]

# models (uniform interface across classical and transformer)
class Detector(Protocol):
    def predict(self, text: str) -> GuardDecision: ...

# attacks
def base64_wrap(payload: str) -> str
def leetspeak(payload: str, rng: Generator) -> str
def zero_width_insert(payload: str, rng: Generator) -> str
def translate_roundtrip(payload: str, via_lang: str) -> str
def embed_in_document(payload: str, carrier_template: str) -> str
def instruction_splitting(payload: str, n_turns: int) -> list[str]

# middleware
@dataclass(frozen=True)
class GuardDecision: label: Literal["benign", "injection"]; score: float; technique_guess: str | None
class ClassifierGuard:
    def check(self, text: str) -> GuardDecision: ...
class GuardedChat:
    def send(self, user_message: str, context_document: str | None = None) -> ChatTurnResult: ...

# eval
def compute_classification_metrics(y_true, y_score, threshold: float) -> MetricsReport
def bypass_rate_by_technique(detector: Detector, variants: list[PromptSample]) -> pd.DataFrame
def adversarial_min_perturbation(detector: Detector, payload: str, rng: Generator) -> RobustnessReport
```

CLI examples (must exist):
```
pidefense train --model configs/classical_tfidf.yaml
pidefense train --model configs/transformer_distilbert.yaml
pidefense evaluate --model results/models/classical.joblib --report results/tables/e01_classical.csv
pidefense attack generate --technique base64 --input data/raw/test.csv --out data/generated/base64_variants.csv
pidefense redteam run --middleware on --model phi3:mini
pidefense redteam run --middleware off --model phi3:mini
pidefense experiment E2 --quick
pidefense experiments --all
```

---

## 7. Default parameters (put in `configs/default.yaml`)

```yaml
seed: 20260119
data:
  dataset_name: "deepset/prompt-injections"
  benign_corpus_path: "data/benign_corpus/benign_queries.csv"
  test_fraction: 0.2
  val_fraction: 0.1
classical_model:
  vectorizer: tfidf_combo
  classifier: logreg
transformer_model:
  base_model: "distilbert-base-uncased"
  max_length: 128
  epochs: 3
  batch_size: 16
attack:
  techniques: [base64, leetspeak, zero_width, translate, indirect, multiturn]
middleware:
  guard_threshold: 0.5
  ollama_model: "phi3:mini"
  ollama_host: "http://localhost:11434"
```

---

## 8. Test strategy (copy into `AGENTS.md` Section 10)

- **Unit:** data loaders (split proportions, no leakage), each obfuscation transform (output differs from input, non-empty, traceable `source_id`), metrics math against hand-computed small examples.
- **Statistical tests:** bootstrap confidence intervals for precision/recall/F1 and for bypass rate per technique; fixed seeds; mark `@pytest.mark.statistical`.
- **Property-based (hypothesis):** every obfuscation function is total (never raises) on arbitrary non-empty ASCII/Unicode strings within a bounded length; `zero_width_insert` output length ≥ input length; `base64_wrap` output is valid base64 of the original payload.
- **Negative tests:** empty string input to each transform, extremely long input (length guard), malformed config (validation error with a clear message), classifier guard called with `None`/empty text.
- **Integration tests:** fixed known-injection and known-benign prompt lists through `GuardedChat`, asserting the expected allow/block decision; grep-based security-hygiene test asserting no module logs raw payload text (mirrors the QKD project's "never log full keys" rule).
- **Regression of experiment outputs:** store small golden CSVs for tiny configs (e.g., 50-row subset) so CI stays fast; full-dataset runs are a separate `make experiments` target, not part of default CI.
- **Coverage target:** ≥ 80% on `src/pidefense` (slightly lower than a pure-crypto project is acceptable given ML training code is harder to exhaustively branch-test; document the rationale in `docs/decisions.md`).
- **Slow/optional tests:** anything requiring `torch`/`transformers` or a live Ollama instance is marked `@pytest.mark.slow` and skipped by default in CI, runnable explicitly via `make test-slow`.

---

## 9. Experiments (E1–E6, map 1:1 to Section 1.2 RQs)

| # | Experiment | Research question | Key output |
|---|---|---|---|
| E1 | `e01_baseline_classical_vs_transformer.py` | RQ1 | Metrics table, ROC curves |
| E2 | `e02_obfuscation_bypass_rate.py` | RQ2 | Bypass-rate table per technique, bar chart |
| E3 | `e03_benign_false_positive_rate.py` | RQ3 | Precision-recall curve, chosen threshold justification |
| E4 | `e04_indirect_injection_vs_output_filter.py` | RQ4 | Before/after middleware success-rate table, qualitative examples |
| E5 | `e05_latency_classical_vs_transformer.py` | RQ5 | Latency table with CI, discussion of sync-guard feasibility |
| E6 | `e06_adversarial_robustness.py` | RQ6 | Minimum-perturbation report, discussion of classifier fragility |

`experiments/run_all.py` runs all of the above with a `--quick` mode (small subsample, finishes in a few minutes) and a full mode.

---

## 10. Integration with the existing AI Pentest Copilot project (optional but recommended)

Document this as a short appendix, not a required deliverable, so the two projects stay independently gradable (per the earlier decision to keep each course's deliverable self-contained):

- The trained `ClassifierGuard` (M6) can wrap any text that is about to be passed into Copilot's `ai/agents.py: run_llm(findings)` — e.g., service banners, log lines, or evidence fields that originate from untrusted scan output, before they reach the LLM prompt.
- If `ai_runs.status` would be `error` due to invalid LLM JSON output, the guard's `GuardDecision` can be attached as extra context explaining *why* (e.g., "input flagged as injection, fell back to `run_local`"), without modifying Copilot's existing schema (10 tables, no new columns required — this can be logged purely in `ai_runs.output_json` as an additional key).
- This integration is explicitly listed as a one-paragraph "hướng mở rộng" (future work) note in both reports, not implemented inside either course deliverable unless the instructor explicitly allows a cross-course combined submission.

---

## 11. Honest-limitations checklist (must appear in the report)

- The dataset (`deepset/prompt-injections`) is a fixed, moderately small, English-centric snapshot; results do not generalize to all languages, all models, or future unseen attack phrasing.
- A classifier with high aggregate F1 can still have a non-trivial bypass rate on specific obfuscation techniques (Section 1.2 RQ2) — "high accuracy" is not "safe."
- The output filter is keyword/pattern-based plus a lightweight semantic check, not a formal verifier; it can itself be evaded by sufficiently indirect phrasing.
- Defense-in-depth reduces but does not eliminate risk; Section 1.3 explicitly disclaims any "production-ready" or "guaranteed safety" claim.
- Red-teaming is limited to a small local instruct model (`phi3:mini`/`llama3.2:1b`); larger/production models may behave differently (more or less susceptible).
- Adversarial robustness testing (RQ6) uses a small, documented perturbation set, not an exhaustive adversarial search; absence of a found bypass does not prove robustness.
- No multi-turn, tool-use, or agentic (function-calling) injection scenarios are tested beyond the simple instruction-splitting case; this is listed as future work.

---

## 12. Report language and structure (Vietnamese, `docs/bao_cao.md`)

1. Giới thiệu & động lực (OWASP LLM01, bối cảnh triển khai chatbot thực tế)
2. Dữ liệu và phương pháp (dataset, split, các kỹ thuật obfuscation/indirect injection)
3. Mô hình phát hiện (classical ML vs transformer) và kết quả (E1)
4. Tỷ lệ bị vượt qua (bypass rate) theo từng kỹ thuật (E2) và tỷ lệ báo sai trên dữ liệu lành (E3)
5. Kiểm thử trên chatbot thật (Ollama) có/không có middleware (E4), ví dụ minh họa
6. Hiệu năng (độ trễ) và đánh đổi (E5)
7. Độ bền vững trước nhiễu loạn đối kháng của chính mô hình phát hiện (E6)
8. Kiến trúc phòng thủ nhiều lớp (middleware) và lý do thiết kế
9. Hạn chế (Section 11, dịch sang tiếng Việt) và hướng mở rộng (bao gồm tích hợp với AI Pentest Copilot, Section 10)
10. Kết luận
11. Phụ lục: toàn bộ payload, cấu hình, lệnh chạy

---

## 13. Kickoff prompts for Codex (copy one at a time)

**Prompt M0**
> Read SPEC.md fully. Implement Milestone M0 only. Follow Section 0 rules. Create the repo layout (Section 3), `pyproject.toml`, `Makefile`, config loader, seeded RNG wrapper. Run lint, mypy, pytest and fix everything. Summarize what you implemented, any decisions (also write them to docs/decisions.md), and the exact commands to verify.

**Prompt M1**
> Implement Milestone M1: data loading (`deepset/prompt-injections` via `datasets`, with an offline CSV fallback committed under `data/raw/`), a curated benign-query corpus (at least 150 diverse rows you write yourself, clearly sourced as "student-curated, not scraped"), and a seeded stratified train/val/test split. Write the unit tests described in the M1 acceptance criteria.

**Prompt M2**
> Implement Milestone M2: TF-IDF feature extraction (word + char n-gram) and classical classifiers (Logistic Regression, Linear SVM, Random Forest) behind the uniform `Detector` interface. Produce `results/tables/e01_classical.csv` with precision/recall/F1/ROC-AUC on the held-out test split. Do not hard-code any metric value in documentation prose — always read it from the CSV.

**Prompt M3**
> Implement Milestone M3: an optional DistilBERT fine-tuning path, fully guarded by `try/except ImportError` so the rest of the project works without `torch` installed. Check the installed `transformers` version's API before coding; do not invent method names. Add the same metrics reporting as M2, plus latency measurement for later use in E5.

**Prompt M4**
> Implement Milestone M4: the attack/obfuscation generators in `attacks/` (Section 4) with the authorized-testing-only scope comment at the top of the package. Add property-based tests (hypothesis) per the M4 acceptance criteria.

**Prompt M5**
> Implement Milestone M5: bypass-rate evaluation (per technique) and false-positive-rate evaluation on the benign corpus, including a precision-recall curve and a documented, justified operating threshold choice. Populate `results/tables/e02_bypass_rate.csv` and `e03_fpr.csv`.

**Prompt M6**
> Implement Milestone M6: the defense-in-depth middleware — `ClassifierGuard`, the hardened system-prompt template with an explicit instruction/data delimiter convention, the output filter, and `GuardedChat` wrapping a local Ollama call. Add the integration tests and the grep-based "no raw payload in logs" security-hygiene test from Section 8.

**Prompt M7**
> Implement Milestone M7: the red-team harness running the full obfuscation/indirect corpus against the local Ollama demo model with middleware ON and OFF, recording per-technique success/violation rates and at least one sanitized qualitative example per technique.

**Prompt M8**
> Implement Milestone M8: adversarial robustness search (RQ6), latency benchmarking (RQ5), and assemble all figures/tables into `docs/bao_cao.md` following Section 12's structure, in Vietnamese, including the honest-limitations checklist from Section 11 verbatim (translated). Do not fabricate any result; everything must trace back to a file in `results/`.

---

## 14. Definition of done (whole project)

- All M0–M8 acceptance criteria pass in CI (fast path) and locally (slow path including transformer + live Ollama red-team).
- `make setup experiments report` reproduces every figure/table from a clean clone (transformer/live-Ollama steps may be marked optional/slow but must still be runnable).
- Report assets in Vietnamese, with limitations (Section 11) included verbatim.
- Every claim in docs is traceable to a CSV, a test, or a cited source (dataset citation, OWASP LLM01 citation).
- The `attacks/` package carries a clear authorized-testing-only scope statement; no payload generator is documented or packaged as a standalone offensive tool.
- Stretch items (multi-turn deep red-teaming, production-model testing, formal adversarial robustness certification) are either implemented with tests or clearly listed as future work in Section 11/12.
