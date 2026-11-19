"""Enforce authorized-testing-only comments throughout the evaluation package."""

from pathlib import Path


def test_every_attack_file_has_scope_comment() -> None:
    """Each authored attack file must begin with the authorized testing scope comment."""
    files = [
        path
        for path in Path("src/pidefense/attacks").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    assert files
    for path in files:
        first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert first_line.startswith("# Authorized-testing-only:"), path
