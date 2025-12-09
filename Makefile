# Paths
SCRIPT := scripts/mdx_snippets_gen.py

# uv run automatically handles virtualenv, so no activation needed
.PHONY: py ts rs snippets

# Generate Python MDX snippets
py:
	@uv run $(SCRIPT) -s tests/py

# Generate TypeScript MDX snippets
ts:
	@uv run $(SCRIPT) -s tests/ts

# Generate Rust MDX snippets
rs:
	@uv run $(SCRIPT) -s tests/rs

# Convenience: generate all snippets
snippets: py ts rs

