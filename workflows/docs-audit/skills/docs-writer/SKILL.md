---
name: docs-writer
description: Use when closing documentation gaps surfaced by a docs-audit run (e.g., workflows/docs-audit/artifacts/runs/<run-id>/report.md) by editing pages in this Mintlify site. The skill grounds every claim in source code from caller-named repos and prevents fabricated APIs, parameters, or behaviors.
---

# Docs writer (audit-driven)

You've been given a list of documentation gaps — typically from a `report.md` produced by a docs-audit run — and your job is to close them by editing pages under `docs/`. The work is mechanical, but two failure modes can ruin it:

1. **Fabrication**: writing about parameters, methods, or defaults that don't exist in the actual code.
2. **Drift**: leaving the prose updated but runnable snippets stale, so the page contradicts itself on the next regen.

Everything below exists to prevent those two failures.

## Workflow

### 1. Read the gap report

Start with the artifact the user pointed you at. For each gap, capture:

- **Page**: which docs page is affected (path under `docs/`).
- **Surface**: the specific behavior, parameter, or API the report says is missing or wrong.
- **Repos**: any source repos the report or user names. These are the source of truth — don't substitute or guess.

If a gap is vague, ask the user before writing. Vague gaps produce filler.

### 2. Inspect the source code

Locate the relevant repos before writing a single sentence. The user will name them; sibling checkouts are usually under `../` from this repo, but confirm with `ls ../` rather than assuming. If you can't find a named repo locally, ask the user for the path before proceeding — never substitute a different repo or fall back to memory.

Whatever repo the user names, that repo wins over your prior knowledge or training data.

For every claim you're about to write:

- **Grep for the symbol**: confirm the parameter, method, or flag exists by name.
- **Read the surrounding code**: understand the actual behavior, not just the signature. Defaults, error paths, and edge cases all matter.
- **Cite paths and line numbers** in the response so the user can audit your check.

If the source code disagrees with the gap report, trust the source and flag the discrepancy back to the user — it might be a real bug, or the report may be stale.

### 3. Draft the doc updates

Edit the affected MDX pages directly. Keep the change scoped to the gap; don't sweep in unrelated improvements unless the user asked for them.

For prose:

- **Placement**: put new sections where readers will encounter the concept naturally, not in the next empty slot at the bottom of the page.
- **Depth and tone**: match the heading depth and voice of surrounding sections.
- **Cross-links**: link to related pages with anchor links when it helps the reader, without spraying too many links and making the prose look cluttered.

For code examples:

- **Short illustrative one-liners**: inline code blocks are fine.
- **Anything canonical or multi-line**: follow the test → snippet → MDX pipeline described in `skills/docs-writer/SKILL.md`. Add a runnable test, mark it with `--8<-- [start:name]` / `[end:name]`, run `make snippets`, and import the generated `{Py|Ts|Rs}{TitleCase}` export.
- **Consistency**: if the page already uses snippet imports, don't introduce inline blocks (or vice versa).

### 4. Verify before reporting back

- Run the relevant test suite if you added tests (`pytest tests/py/...`, `npx jest ...`, `cargo run --example ...`).
- Run `make snippets` if test sources changed, and confirm `git status` shows the regenerated MDX so it lands in the same commit.
- Re-read each new sentence against the code one more time. If a claim isn't directly supported by something you grepped, cut it.

## Style rules

Apply these to every piece of prose you write or edit through this skill.

- **Bullet separators**: use colons, not em dashes. Write `Placement: put new sections where...`, not `Placement — put new sections where...`.
- **Contractions**: write `it's`, `don't`, `you'll`. Skip the formal register; the docs are technical, but they're not a legal document.
- **Subheader case**: sentence case only — capitalize just the first word and any proper nouns. Use `Tag-based versioning`, not `Tag-Based Versioning`.
- **Tone**: technical but approachable. Write primarily for engineers, but define jargon on first use and ground abstract ideas in concrete examples. Don't assume the reader has the same context you have from reading the source.

## Anti-fabrication checklist

Before finalizing any doc update, walk this list:

- Every parameter named in prose was found via grep in the source repo.
- Every method call shown in a code example compiles or runs (in a test, ideally).
- Every claimed default value was checked in the code, not assumed.
- Every cross-language parity claim (e.g., "Python and TypeScript both expose X") was checked in each binding's source.
- Every "this is exempt", "this is preserved", or "this is automatic" claim has a code or comment citation.

If an item fails, either fix the doc or surface the question to the user. Never paper over a gap with confident-sounding prose.

## Output contract

When using this skill, the final result should:

- Update the affected MDX pages so the identified gaps are genuinely closed.
- Cite source-code paths and line numbers for non-trivial claims, both in the PR description and in the response back to the user.
- Flag any gap that turned out to be inaccurate, ambiguous, or out of scope, with reasoning.
- Leave the snippet pipeline coherent if code examples were touched: tests pass, `docs/snippets/` regenerated, regenerated files staged alongside the test changes.

## Request template

A typical invocation looks like:

```text
Use workflows/docs-audit/skills/docs-writer/SKILL.md to close the gaps in
workflows/docs-audit/artifacts/runs/<run-id>/report.md. Cross-check each claim
against <repo-path-1> and <repo-path-2>, and update the affected pages under
docs/.
```

The user supplies the repo paths in the invocation. Don't infer them, confirm with the user if you are unsure which source repos to use as a grounding reference.
