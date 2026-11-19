"""Grep-style log-sink policy plus AST checks for the sole metadata audit call."""

import ast
import re
from pathlib import Path


def test_no_raw_payload_logging_in_middleware_or_demo() -> None:
    """Ban log sinks except the reviewed fixed metadata-only audit expression.

    Text scanning detects common logging, printing and persistence sinks;
    AST verifies the allowed call and fingerprint expression so a variable
    rename or multiline logging call cannot silently add raw content.
    Runtime integration capture separately checks the emitted log records.
    """
    roots = (Path("src/pidefense/middleware"), Path("src/pidefense/demo"))
    sink = re.compile(
        r"\b(?:debug|info|warning|warn|error|exception|critical|log|print|pprint|"
        r"write|writelines|write_text|write_bytes|open|basicConfig|FileHandler)\s*\(",
    )
    allowed = (
        '_LOGGER.info("length=%d fingerprint=%s label=%s score=%.6f rule=%s", '
        "len(text), fingerprint, decision.label, decision.score, rule)"
    )
    expected = ast.dump(ast.parse(allowed).body[0], include_attributes=False)
    count = 0
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call = ast.get_source_segment(source, node) or ""
                func = ast.get_source_segment(source, node.func) or ""
                if not sink.search(func + "("):
                    continue
                if func == "self._opener.open":
                    # The HTTP read sink is local-only and is not a log sink.
                    assert path.name == "ollama_client.py"
                    continue
                assert path.name == "pipeline.py", path
                assert ast.dump(ast.parse(call).body[0], include_attributes=False) == expected, path
                count += 1
            if path.name == "pipeline.py":
                fingerprint = 'sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()[:8]'
                assert fingerprint in source
            assert not re.search(r"\b(?:exc_info|stack_info)\s*=\s*True", source), path
            assert not re.search(r"(?:from|import)\s+pidefense\.attacks", source), path
    assert count == 1
