# Agent Style Guide

Detailed coding and testing conventions for this codebase.

## Imports

1. `from __future__ import annotations` first
2. Standard library imports
3. Third-party imports
4. Local imports

Use `TYPE_CHECKING` for heavy type-only imports.

## Type Annotations

- Annotate all function parameters and return types.
- Use Python 3.10+ syntax: `list[X]`, `dict[str, int | float]`, `X | None`.
- Use `dict[str, int | float]` for feature dictionaries.
- Prefer immutable tuple typing where practical (`tuple[str, ...]`).

## Dataclasses

Use frozen, slotted dataclasses for immutable data models:

```python
@dataclass(frozen=True, slots=True)
class SentenceEmbedding:
    id: int
    text: str
    embedding: list[float]
```

## Docstrings

- Module docstrings should explain module purpose.
- Public functions should use Google-style docstrings.

## Constants And Thresholds

- Constants use `UPPER_SNAKE_CASE`.
- Centralize thresholds in `src/mental_entropy/utils/thresholds.py`.
- Add short rationale docstrings for calibrated constants.

## Feature Keys

Each feature module defines an exact keyset with `frozenset[str]`:

```python
CE_FEATURE_KEYS: frozenset[str] = frozenset([
    "ce_n_sentences",
    "ce_adj_mean",
    ...
])
```

## Naming

- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Internal helpers: `_leading_underscore`
- Feature keys: `{module_prefix}_{feature_name}` (for example `ce_adj_mean`)

## Error Handling

- Validate input early and raise `ValueError` with explicit messages.
- Reject NaN/Inf numeric input before feature computations.

```python
if not np.isfinite(X).all():
    raise ValueError("Embeddings contain NaN or Inf values")
```

## JSON Serialization

Feature values must be Python `int`/`float`, not NumPy scalars:

```python
return {
    "ce_n_sentences": int(n),
    "ce_adj_mean": float(np.mean(s)),
}
```

## Edge Cases

Handle empty/small inputs with deterministic safe defaults:

```python
def ce_features(sentence_embeddings: list[list[float]]) -> dict[str, int | float]:
    if len(sentence_embeddings) == 0:
        return _empty_features(n_sentences=0)
    if len(sentence_embeddings) == 1:
        return _empty_features(n_sentences=1)
```

## Testing Patterns

- Test names: `test_{function}_{scenario}`.
- Use synthetic unit vectors for feature tests (no model inference dependency).
- Use approximate float assertions (`abs(actual - expected) < 1e-5`).
- Validate exact keysets and JSON-serializability explicitly.

Example helper:

```python
EMBEDDING_DIM = 1024

def make_unit_vector(direction: int = 0) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[direction % EMBEDDING_DIM] = 1.0
    return vec
```
