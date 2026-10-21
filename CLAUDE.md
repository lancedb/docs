# LanceDB Mintlify Documentation

This is a documentation site for [LanceDB](https://docs.lancedb.com).

## Languages used

- Code examples are primarily in three language SDKs: Python, TypeScript and Rust.
- Best practices for linting, formatting and code complexity for each respective language apply.
- Write idiomatic code as far as possible

## Snippets

- Code snippets are generated in `lancedb/lancedb` from the tests under `docs/web-tests/`, not in this repository. See "Code snippets" in README.md for the command.
- The four Geneva snippets in `docs/snippets/` are frozen copies; do not regenerate or edit them.

## Saved editions

- `editions/` holds saved documentation editions and `releases/` their records. Create an edition only with `scripts/edition.py prepare`, and change one only as a reviewed correction listed in its record. See "Saved editions" in README.md.
- Write new documentation in `lancedb/lancedb` or the Enterprise documentation repository, not in a saved edition.
- `make compose` builds the versioned site from the saved editions, and `compose.yaml` names each product's preferred edition. See "Composing the site" in README.md.
- Shared branding, integrations and global assets belong in `site/settings.json` and `site/static/`, not in a saved edition.
- The Compose workflow keeps each run's versioned site as a candidate, and the Publish workflow puts candidates on `staging` and, after approval, the one staging shows on `assembled`. Never push to either branch yourself. See "Publishing" in README.md.
- `scripts/discover.py` and the Discover releases workflow prepare editions of eligible releases on `editions/<product>-<version>` branches and open their pull requests for review; `discover.yaml` says where each product starts. See "Discovering releases" in README.md.
