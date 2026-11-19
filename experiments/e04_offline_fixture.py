"""Generate explicitly synthetic E04 regression artifacts, never live efficacy claims."""

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from e04_indirect_injection_vs_output_filter import results_path, run

from pidefense.demo.ollama_client import ChatMessage


class FixtureClient:
    """Deterministic canned answers exercise leaks and refusals without a model."""

    def chat_messages(
        self, messages: Sequence[ChatMessage], *, options: Mapping[str, int | float] | None = None
    ) -> str:
        """Choose a canned answer by the seeded trial ID; no semantic inference is made."""
        assert options is not None
        if int(options["seed"]) % 2:
            return "PIDEFENSE_SYSTEM_PRIVATE"
        return "I cannot follow instructions overriding my rules."


def generate_fixture(
    output_dir: Path,
    *,
    appendix_path: Path | None = None,
    config_path: Path = Path("configs/default.yaml"),
) -> Path:
    """Exercise all techniques with fake translation, explicitly confined to fixture mode."""
    output_dir = results_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Synthetic translation is only a routing fixture, not evidence of real translation.
    with Path("data/raw/deepset_sample.csv").open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    steps = []
    for sample in samples:
        if sample["label"] != "1":
            continue
        text = sample["text"]
        intermediate = "[SYNTHETIC CI VI] " + text
        steps.extend(
            [
                {
                    "text": text,
                    "source_lang": "en",
                    "target_lang": "vi",
                    "translation": intermediate,
                },
                {
                    "text": intermediate,
                    "source_lang": "vi",
                    "target_lang": "en",
                    "translation": text + " [SYNTHETIC CI EN]",
                },
            ]
        )
    cache = output_dir / "e04_synthetic_translation_fixture.json"
    cache.write_text(json.dumps(steps, indent=2) + "\n", encoding="utf-8")
    return run(
        config_path,
        output_dir,
        fixture=True,
        client=FixtureClient(),
        translation_cache=cache,
        appendix_path=appendix_path,
    )


if __name__ == "__main__":
    generate_fixture(Path("results"), appendix_path=Path("docs/bao_cao.md"))
    print("E04 synthetic fixture written; no live Ollama model or real translation was used.")
