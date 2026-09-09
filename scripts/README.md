# Scripts

Repo-local wrappers for globally installed helper scripts in `~/.codex/scripts`.

## `scripts/committer`

Delegates to `~/.codex/scripts/committer` (safe commit helper).

Examples:

```bash
scripts/committer -m "Add NE regression test" tests/features/test_ne_features.py
scripts/committer --staged -m "Refactor embedding text normalization"
scripts/committer --dry-run -m "Update docs index" docs/agent_docs_index.md
```

## `scripts/docs-list`

Delegates to `~/.codex/scripts/docs-list` (prints the docs map from `docs/agent_docs_index.md`).

```bash
scripts/docs-list
```

## `scripts/browser-tools`

Delegates to `~/.codex/scripts/browser-tools` (Chrome DevTools helper CLI).

```bash
scripts/browser-tools --help
```

## `scripts/global-scripts-sync`

Delegates to `~/.codex/scripts/sync-agent-scripts` (one-command global tool sync from `steipete/agent-scripts`).

```bash
scripts/global-scripts-sync
```
