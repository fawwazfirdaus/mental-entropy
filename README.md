# Mental Entropy Score (MES)

Measures semantic coherence, topic fragmentation, narrative structure, and cognitive load in journal entries. Produces a 0-100 score where higher = more fragmented thinking.

**Stack**: Python 3.12, PyTorch, Transformers (mxbai-embed-large-v1), FastAPI, scikit-learn

**Live API**: `https://mental-entropy-production.up.railway.app`

## Quick Start

```bash
# Score a journal via the live API
curl -X POST https://mental-entropy-production.up.railway.app/score \
  -H "Content-Type: application/json" \
  -d '{"text": "Today I felt scattered, jumping between tasks without finishing any."}'
```

```bash
# Or run locally
uv sync --all-extras
PYTHONPATH=src .venv/bin/python -m mental_entropy.api
# API available at http://localhost:8000
```

## API Reference

Base URL: `https://mental-entropy-production.up.railway.app`

### POST /score

Score a single journal entry.

```bash
curl -X POST /score \
  -H "Content-Type: application/json" \
  -d '{"text": "I had a productive day. Finished my report and went for a walk."}'
```

**Request:**
```json
{ "text": "Journal entry text (10-50,000 characters)" }
```

**Response:**
```json
{
  "mes_score": 22.4,
  "interpretation": "Low entropy: Well-organized with clear flow and minimal fragmentation.",
  "features": { "ce_adj_mean": 0.82, "se_n_clusters": 1, ... },
  "subscores": null
}
```

### POST /score/batch

Score up to 50 journal entries in one request.

```bash
curl -X POST /score/batch \
  -H "Content-Type: application/json" \
  -d '{"entries": [
    {"id": "j1", "text": "Calm and organized day."},
    {"id": "j2", "text": "Everything is falling apart I cant focus on anything."}
  ]}'
```

**Request:**
```json
{
  "entries": [
    { "id": "unique-id", "text": "Journal text..." }
  ]
}
```

**Response:**
```json
{
  "results": [
    { "id": "j1", "mes_score": 15.2, "interpretation": "Very low entropy..." },
    { "id": "j2", "mes_score": 72.8, "interpretation": "High entropy..." }
  ]
}
```

### POST /user-state

Get a user's **current mental entropy state** from their journal history. Uses exponential decay weighting — recent entries count more than old ones.

```bash
curl -X POST /user-state \
  -H "Content-Type: application/json" \
  -d '{"entries": [
    {"timestamp": "2026-03-20T10:00:00Z", "mes_score": 55},
    {"timestamp": "2026-03-22T14:00:00Z", "mes_score": 48},
    {"timestamp": "2026-03-24T09:00:00Z", "mes_score": 42},
    {"timestamp": "2026-03-25T11:00:00Z", "mes_score": 38}
  ]}'
```

**Request:**
```json
{
  "entries": [
    { "timestamp": "ISO 8601", "mes_score": 0-100 }
  ],
  "half_life_days": 14.0
}
```

**Response:**
```json
{
  "current_mes": 45.15,
  "confidence": 0.83,
  "trend": "improving",
  "trend_slope": -23.5,
  "baseline_mes": 45.75,
  "deviation_from_baseline": -0.6,
  "n_entries": 4,
  "n_recent_entries": 4,
  "days_since_last_entry": 0.44,
  "interpretation": "Moderate entropy...",
  "status": "ok"
}
```

**Key fields:**
- `confidence` (0-1): How reliable the estimate is. Show the score if > 0.3, prompt "journal to update" if below.
- `status`: `"ok"` (enough data), `"stale"` (old data), `"insufficient_data"` (no entries)
- `half_life_days`: How fast old entries fade. Default 14 days. At 14 days ago, weight = 0.5.

### POST /temporal

Compute 21 temporal entropy features for longitudinal analysis.

```bash
curl -X POST /temporal \
  -H "Content-Type: application/json" \
  -d '{"entries": [
    {"timestamp": "2026-03-20T10:00:00Z", "mes_score": 55},
    {"timestamp": "2026-03-22T14:00:00Z", "mes_score": 48}
  ]}'
```

**Response:**
```json
{
  "temporal_features": {
    "te_n_entries": 2,
    "te_slope_per_week": -24.5,
    "te_rolling_7d_mean": 51.5,
    "te_baseline_mean": 51.5,
    "te_trend_direction": -1,
    "te_volatility": 0.10,
    ...
  },
  "trend": "improving",
  "summary": "Based on 2 entries over 2 days: trend is improving (-24.5 points/week)..."
}
```

### POST /classify

Classify a user's mental state from MES signals. Deterministic — no LLM, runs in microseconds.

```bash
curl -X POST /classify \
  -H "Content-Type: application/json" \
  -d '{
    "user_state": {
      "current_mes": 58.25,
      "confidence": 0.93,
      "trend": "worsening",
      "trend_slope": 9.4,
      "baseline_mes": 52.0,
      "n_entries": 12
    },
    "temporal_features": { "te_volatility": 0.30 },
    "current_subscores": { "compression_progress": 1.56 }
  }'
```

**Response:**
```json
{
  "state": "overwhelm",
  "state_display": "Overwhelmed",
  "description": "High entropy with significant volatility...",
  "confidence": 0.93,
  "primary_signals": ["current_mes=58.2 (above 55.0)", "volatility=0.30 (above 0.25)"],
  "weakest_dimensions": [{"name": "compression_progress", "value": 1.56, "display_name": "Compression Progress", "description": "..."}],
  "directive": "Ground, simplify, help find anchors...",
  "avoid": "Do not introduce new complexity...",
  "target_direction": "decrease"
}
```

**States:** `overwhelm`, `stuck`, `rigidity`, `active_integration`, `baseline`, `insufficient_data`

### POST /insight

Assemble a state-aware LLM prompt for insight generation. Returns a ready-to-send prompt payload + feedback context for the closed loop.

```bash
curl -X POST /insight \
  -H "Content-Type: application/json" \
  -d '{
    "user_state": {
      "current_mes": 58.25,
      "confidence": 0.93,
      "trend": "worsening",
      "trend_slope": 9.4,
      "baseline_mes": 52.0,
      "n_entries": 12
    },
    "journal_text": "I keep going back and forth about whether to leave my job...",
    "timestamp": "2026-04-01T10:00:00Z"
  }'
```

**Response:**
```json
{
  "classification": { "state": "overwhelm", "...": "..." },
  "prompt": {
    "system": "You are a therapeutic insight companion...",
    "user": "## Current State\n...\n## Journal Entry\n...",
    "model_recommendation": "claude-sonnet-4-20250514",
    "max_tokens": 300,
    "temperature": 0.5
  },
  "feedback_context": {
    "timestamp": "2026-04-01T10:00:00Z",
    "state": "overwhelm",
    "mes_at_insight": 58.25,
    "subscores_at_insight": {},
    "target_direction": "decrease"
  }
}
```

Send `prompt.system` + `prompt.user` to your LLM. Store `feedback_context` for the feedback loop.

### POST /insight/feedback

Measure whether a previous insight moved MES in the right direction. Send back the stored `feedback_context` + the new MES score.

```bash
curl -X POST /insight/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "feedback_context": {
      "timestamp": "2026-04-01T10:00:00Z",
      "state": "overwhelm",
      "mes_at_insight": 58.25,
      "subscores_at_insight": {},
      "target_direction": "decrease"
    },
    "new_mes": 51.3
  }'
```

**Response:**
```json
{
  "effective": true,
  "mes_delta": -6.95,
  "target_direction": "decrease",
  "moved_correctly": true,
  "subscore_deltas": {},
  "interpretation": "MES decreased by 7.0 points — insight moved entropy in the right direction."
}
```

### GET /health

Health check and model info.

```bash
curl /health
```

**Response:**
```json
{
  "status": "ok",
  "model": "hybrid_v6",
  "architecture": "hybrid_v6",
  "n_training_entries": 4240,
  "human_corr": 0.6362,
  "version": "1.0.0"
}
```

## How It Works

```
Journal text
    ↓
mxbai-embed-large-v1 → 1024-dim document embedding
    ↓
Ridge regression on [embedding + 10 hand-crafted features]
    ↓
MES score (0-100)
```

**Score interpretation:**
- 0-20: Very low entropy (highly organized)
- 21-35: Low entropy (well-organized)
- 36-50: Moderate entropy (some fragmentation)
- 51-65: Elevated entropy (noticeable fragmentation)
- 66-80: High entropy (significant disorganization)
- 81-100: Very high entropy (severely fragmented)

## Integration (Node.js / Empath)

```javascript
const MES_API = process.env.MES_API_URL || 'https://mental-entropy-production.up.railway.app';
const Anthropic = require('@anthropic-ai/sdk');
const anthropic = new Anthropic();

// Score a journal
async function scoreJournal(text) {
  const res = await fetch(`${MES_API}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return res.json(); // { mes_score, interpretation, features, subscores }
}

// Get user's current state
async function getUserState(entries) {
  const res = await fetch(`${MES_API}/user-state`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entries }), // [{ timestamp, mes_score }]
  });
  return res.json(); // { current_mes, confidence, trend, trend_slope, ... }
}

// Generate a state-aware insight (full closed loop)
async function generateInsight(journalText, userState, score) {
  // 1. Get prompt from MES service
  const res = await fetch(`${MES_API}/insight`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_state: {
        current_mes: userState.current_mes,
        confidence: userState.confidence,
        trend: userState.trend,
        trend_slope: userState.trend_slope,
        baseline_mes: userState.baseline_mes,
        n_entries: userState.n_entries,
      },
      journal_text: journalText,
      timestamp: new Date().toISOString(),
    }),
  });
  const insight = await res.json();

  // 2. Send to LLM
  const llm = await anthropic.messages.create({
    model: insight.prompt.model_recommendation,
    max_tokens: insight.prompt.max_tokens,
    temperature: insight.prompt.temperature,
    system: insight.prompt.system,
    messages: [{ role: 'user', content: insight.prompt.user }],
  });

  // 3. Store feedback_context for the feedback loop (send back on next entry)
  return {
    insightText: llm.content[0].text,
    state: insight.classification.state_display,
    feedbackContext: insight.feedback_context, // store in DB
  };
}
```

> **Full integration guide**: See [`docs/insight_engine_integration.md`](docs/insight_engine_integration.md) for database schema, service layer, error handling, and analytics queries.

## Model Details

**Current model**: Hybrid v6 — Ridge regression on [1024-dim mxbai embedding + top-10 features]

- Trained on 4240 entries (2995 human journals + 1245 synthetic)
- Multi-rater consensus labels (3 passes x Claude Sonnet 4.6, inter-rater r=0.83-0.92)
- human_corr = 0.636, cv_corr = 0.774
- Deterministic: same input always produces same output
- ~100-150ms per entry on CPU

## Development

```bash
# Install with all dependencies
uv sync --all-extras

# Run tests (378 tests)
PYTHONPATH=src .venv/bin/pytest tests/ -v

# Run locally
PYTHONPATH=src .venv/bin/python -m mental_entropy.api
```

## Architecture

```
src/mental_entropy/
├── api.py              # FastAPI REST service (8 endpoints)
├── score.py            # MES scoring (auto/embedding/subscore/linear methods)
├── embedding/          # Text → 1024-dim embeddings (mxbai)
├── features/           # 77 hand-crafted features (CE/SE/NE/CLE/BC)
├── models/             # Ridge + XGBoost model artifacts
├── temporal/           # 21 longitudinal features + user state
├── insight/            # Insight engine (classify → prompt → feedback)
│   ├── types.py        #   6 mental states, dataclasses
│   ├── classify.py     #   Deterministic state classification
│   ├── prompts.py      #   State-aware LLM prompt assembly
│   └── feedback.py     #   Closed-loop effectiveness measurement
└── utils/              # Thresholds, linguistic patterns
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MES_PORT` | 8000 | API server port |
| `MES_HOST` | 0.0.0.0 | Bind address |
| `MES_WORKERS` | 1 | Uvicorn workers (keep at 1 — model is shared) |
| `MES_CORS_ORIGINS` | * | Comma-separated allowed CORS origins |
| `MES_DEVICE` | cpu | PyTorch device (cpu or cuda) |
| `PORT` | — | Railway/Heroku port override (takes priority) |

See `CLAUDE.md` for full architecture documentation and `BACKEND_CONTEXT.md` for Empath integration details.
