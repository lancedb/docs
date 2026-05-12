# Paths
SCRIPT := scripts/mdx_snippets_gen.py
HF_SYNC_SCRIPT := scripts/sync_hf_datasets.py

# uv run automatically handles virtualenv, so no activation needed
.PHONY: py ts rs snippets hf-sync

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

# Sync Lance dataset cards from lance-format/lance-huggingface into docs/datasets/.
# Regenerates per-dataset MDX pages, the landing-page card grid, and the
# Datasets tab in docs.json based on scripts/hf_datasets.yaml.
hf-sync:
	@uv run $(HF_SYNC_SCRIPT)

