# Agent Docs Index

Use this file to choose the minimum docs needed for a task.

## Load Order

1. `AGENTS.md`
2. `CLAUDE.md`
3. `docs/agent_docs_index.md` (this file)
4. Task-specific docs from the table below

## Document Map

| Path | Summary | Read When |
| --- | --- | --- |
| `AGENTS.md` | Primary operating contract for coding agents in this repo. | Always first. |
| `CLAUDE.md` | High-level architecture and command reference. | Start of any implementation task; architecture questions. |
| `docs/agent_style_guide.md` | Detailed style and testing conventions for new/edited code. | Before editing Python modules or tests. |
| `docs/codebase_mental_model.md` | Implementation-focused behavior and invariants validated by tests. | Debugging behavior mismatches; onboarding to current state. |
| `docs/mes_architecture.md` | Target MES architecture and current-vs-planned boundaries. | Planning roadmap work; subscore/combiner design tasks. |
| `README.md` | Public usage and feature overview. | User-facing docs updates; API usage examples. |

- Research evaluation and reproduction: `docs/research_notes.md`, `autoresearch-macos/README.md`

## Fast Picks By Task

- Test failures in feature modules: `docs/agent_style_guide.md`, `docs/codebase_mental_model.md`
- Architecture or roadmap changes: `CLAUDE.md`, `docs/mes_architecture.md`
- Public API/documentation updates: `README.md`, `CLAUDE.md`
- Datagen workflow work: `CLAUDE.md`, `docs/codebase_mental_model.md`
