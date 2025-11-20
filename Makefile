SHELL := /bin/bash

# Paths
SCRIPT := scripts/mdx_snippets_gen.py

# Helper to ensure we run within the local virtualenv
ACTIVATE_VENV := . .venv/bin/activate

.PHONY: py ts rs snippets

# Generate Python MDX snippets
py:
	@$(ACTIVATE_VENV); uv run $(SCRIPT) -s examples/py

# Generate TypeScript MDX snippets
ts:
	@$(ACTIVATE_VENV); uv run $(SCRIPT) -s examples/ts

# Generate Rust MDX snippets
rs:
	@$(ACTIVATE_VENV); uv run $(SCRIPT) -s examples/rs

# Convenience: generate all snippets
snippets: py ts rs

