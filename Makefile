UV ?= uv

.PHONY: setup test test-slow lint typecheck experiments figures report demo

setup:
	$(UV) sync --locked --extra dev

test:
	$(UV) run pytest

test-slow:
	$(UV) run pytest -m slow

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy --strict src

# Later milestones must replace these explicit failures with real implementations.
experiments figures report demo:
	$(UV) run python -c "import sys; sys.exit('Target $@ is deferred beyond M0; see SPEC.md.')"
