# Paths
HF_SYNC_SCRIPT := scripts/sync_hf_datasets.py
ASSEMBLE_SCRIPT := scripts/assemble.py
# The assembler needs only pyyaml; skipping the project env keeps CI from
# resolving lancedb, pyarrow, polars and geneva to run a file-copying script.
ASSEMBLE_RUN := uv run --no-project --with pyyaml

# uv run automatically handles virtualenv, so no activation needed
.PHONY: py ts rs snippets hf-sync assemble test-assemble check-spec sync-spec

# Snippets are generated in lancedb/lancedb, beside the tests they come from;
# these targets only say so. See "Code snippets" in README.md.
py ts rs snippets:
	@echo "Snippets are not generated in this repository. In a lancedb/lancedb checkout, run:" >&2
	@echo "  uv run docs/web-tests/mdx_snippets_gen.py -s docs/web-tests/py -s docs/web-tests/ts -s docs/web-tests/rs -o docs/web/snippets" >&2
	@echo "The Geneva snippets in docs/snippets/ are frozen copies and are not regenerated." >&2
	@exit 1

# Sync Lance dataset cards from lance-format/lance-huggingface into docs/datasets/.
# Regenerates per-dataset MDX pages, the landing-page card grid, and the
# Datasets tab in docs.json based on scripts/hf_datasets.yaml.
hf-sync:
	@uv run $(HF_SYNC_SCRIPT)

# Assemble the published tree from the roots declared in assemble.yaml into
# build/site. Set LANCEDB_DOCS_ROOT and SOPHON_DOCS_ROOT when the lancedb and
# sophon checkouts are not beside this one.
assemble:
	@$(ASSEMBLE_RUN) $(ASSEMBLE_SCRIPT)

# Test the assembler's overlay, navigation and private-root guards.
test-assemble:
	@$(ASSEMBLE_RUN) --with pytest pytest scripts/tests -q

# Fail if the tracked OpenAPI spec has drifted from the release pinned in
# assemble.yaml. Run in CI so the pin cannot rot unnoticed.
check-spec:
	@$(ASSEMBLE_RUN) $(ASSEMBLE_SCRIPT) --check-spec

# Rewrite the tracked OpenAPI spec from its pinned release.
sync-spec:
	@$(ASSEMBLE_RUN) $(ASSEMBLE_SCRIPT) --sync-spec

