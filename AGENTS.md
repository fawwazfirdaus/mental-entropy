# AGENTS.md - Mental Entropy Agent Entry Point

Primary instructions for coding agents in this repository.

This is intentionally pointer-first. Load the smallest relevant context, then continue.

## Quick Load Order

1. `AGENTS.md` (this file)
2. `CLAUDE.md` (high-level architecture + commands)
3. `docs/agent_docs_index.md` (task-based doc routing)
4. Only the docs needed for the current task

## Project Snapshot

Mental Entropy Score (MES) pipeline for journal text analysis.

- Implemented: embedding layer + CE/SE/NE/CLE feature extraction
- Planned: learned subscore models + final MES combiner
- Stack: Python 3.12, PyTorch, Transformers, spaCy, NumPy, pytest

## Essential Commands

```bash
# Install dependencies
uv sync                    # With uv (recommended)
uv sync --all-extras       # With dev deps (pytest)

# Or without uv:
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Run ALL tests (PYTHONPATH required for module resolution)
PYTHONPATH=src .venv/bin/pytest tests/ -v

# Run a SINGLE test file
PYTHONPATH=src .venv/bin/pytest tests/test_embedding_smoke.py -v
PYTHONPATH=src .venv/bin/pytest tests/features/test_ce_features.py -v

# Run a SINGLE test function
PYTHONPATH=src .venv/bin/pytest tests/features/test_ce_features.py::test_ce_features_empty_input -v

# Run tests matching a pattern
PYTHONPATH=src .venv/bin/pytest tests/ -v -k "empty"

# Quick sanity check (verify package loads)
PYTHONPATH=src .venv/bin/python -c "from mental_entropy import embed_journal_entry; print('OK')"
```

## Agent Workflow

1. Read only what is needed via `docs/agent_docs_index.md`.
2. Prefer minimal, isolated edits.
3. Run targeted tests first, then broader tests if risk is high.
4. Keep outputs deterministic and JSON-serializable (`int`/`float` scalars).

## Code And Test Conventions

Detailed conventions live in:

- `docs/agent_style_guide.md`

This includes imports, typing style, dataclass patterns, naming, error handling, and test patterns.

## Architecture References

- `CLAUDE.md`: high-level architecture and common gotchas
- `docs/codebase_mental_model.md`: implementation behavior and invariants
- `docs/mes_architecture.md`: target-vs-current MES architecture

## Commit Safety Helpers

- `scripts/committer`: safe commit flow that requires explicit scope
- `scripts/docs-list`: quick list of docs with summary/read-when hints
- `scripts/browser-tools`: Chrome DevTools helper CLI (global wrapper)
- `scripts/global-scripts-sync`: sync global helper scripts from `steipete/agent-scripts`
- Global script install/update reference: `~/.codex/scripts/README.md`

Examples:

```bash
scripts/docs-list
scripts/committer -m "Add CE edge-case test" tests/features/test_ce_features.py
scripts/committer --staged -m "Refactor feature validation"
```

## Maintaining CLAUDE.md

Keep `CLAUDE.md` updated whenever changes affect:

- Essential development commands
- Architecture or module responsibilities
- Cross-cutting design patterns
- Common gotchas discovered during implementation

Keep overlap consistent:

- `CLAUDE.md`: high-level architecture and command quickstart
- `AGENTS.md`: entry-point routing and operating contract
- `docs/agent_style_guide.md`: detailed implementation conventions

## Lesson Log

- MES v9 tuning: raise high-label weight before false-low weight; false-low weight breaks emotional controls fast.
- MES audit: compare feature-family correlations and combiner sign agreement before retraining.
- MES v10: sign-constrained final combiner alone can worsen locked MAE; inspect dimension models next.
- MES ground-up: embeddings carry stronger global OE signal than individual feature families.
- MES labels: consensus CSV must carry raw rater std; golden anchors drift after reaggregation.
