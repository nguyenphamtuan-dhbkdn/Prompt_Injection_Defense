UV ?= uv

.PHONY: setup test test-slow lint typecheck experiments transformer redteam figures report demo

setup:
	$(UV) sync --locked --extra dev

test:
	$(UV) run pytest

test-slow:
	$(UV) run --extra transformer pytest -m slow

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy --strict src

experiments:
	$(UV) run python experiments/e01_baseline_classical_vs_transformer.py

transformer:
	$(UV) run --extra transformer python experiments/e01_transformer.py

redteam:
	$(UV) run python experiments/e04_indirect_injection_vs_output_filter.py

# Later milestones must replace these explicit failures with real implementations.
figures report demo:
	$(UV) run python -c "import sys; sys.exit('Target $@ is deferred beyond M0; see SPEC.md.')"
