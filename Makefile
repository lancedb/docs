# Paths
SCRIPT := scripts/mdx_snippets_gen.py
HF_SYNC_SCRIPT := scripts/sync_hf_datasets.py
ASSEMBLE_SCRIPT := scripts/assemble.py
# The assembler needs only pyyaml; skipping the project env keeps CI from
# resolving lancedb, pyarrow, polars and geneva to run a file-copying script.
ASSEMBLE_RUN := uv run --no-project --with pyyaml

# uv run automatically handles virtualenv, so no activation needed
.PHONY: py ts rs snippets hf-sync assemble check-spec sync-spec

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

# Assemble the published tree from the roots declared in assemble.yaml into
# build/site. Today that is one root and the output matches docs/ byte for byte;
# later phases add the lancedb and sophon roots without changing the script.
assemble:
	@$(ASSEMBLE_RUN) $(ASSEMBLE_SCRIPT)

# Fail if the tracked OpenAPI spec has drifted from the release pinned in
# assemble.yaml. Run in CI so the pin cannot rot unnoticed.
check-spec:
	@$(ASSEMBLE_RUN) $(ASSEMBLE_SCRIPT) --check-spec

# Rewrite the tracked OpenAPI spec from its pinned release.
sync-spec:
	@$(ASSEMBLE_RUN) $(ASSEMBLE_SCRIPT) --sync-spec

