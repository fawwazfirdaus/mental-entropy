# Insight Engine — Integration Guide for Empath Backend

> **Audience**: Empath backend developers integrating the MES Insight Engine into the Node.js/Express journaling app.
>
> **Prerequisites**: You're already calling `/score` and `/user-state`. This guide adds three new endpoints to your integration.

---

## What the Insight Engine Does

The MES service already tells you *how much* mental entropy a user has. The Insight Engine tells you **what to do about it**:

1. **Classify** the user's mental state from MES signals (deterministic, no LLM)
2. **Assemble** a state-aware LLM prompt tailored to what they need right now
3. **Measure** whether the insight actually helped (closed-loop feedback)

The MES service stays **stateless** — it returns prompts, never calls LLMs. Empath owns the user, the LLM client, and the state storage.

---

## Mental States

The classifier diagnoses one of 6 states:

| State | When | What it means | Target |
|-------|------|---------------|--------|
| `overwhelm` | MES > 55 + volatile | Too much unintegrated information | MES should **decrease** |
| `stuck` | MES > 55 + flat slope | Internal model not updating | MES should **decrease** |
| `rigidity` | MES < 35 + flat slope | Over-compressed, filtering out new info | MES should **increase** |
| `active_integration` | Improving trend, slope < -1.0/week | Mind actively compressing experience | MES should keep **decreasing** |
| `baseline` | Normal range, normal fluctuation | Healthy — nothing to fix | MES should stay **stable** |
| `insufficient_data` | < 3 entries or confidence < 0.1 | Not enough history to classify | MES should stay **stable** |

**Key insight**: MES is not a number to minimize. Zero entropy = rigidity/depression. The goal is dynamic equilibrium — the right amount of entropy for active processing.

---

## Architecture & Data Flow

```
User writes journal entry
        │
        ▼
┌─ Empath Backend ──────────────────────────────────────────┐
│                                                            │
│   1. POST /score  { text }                                 │
│      → mes_score, features, subscores                      │
│      → Store in DB                                         │
│                                                            │
│   2. POST /user-state  { entries: history }                │
│      → current_mes, confidence, trend, trend_slope         │
│                                                            │
│   3. POST /temporal  { entries: history }  (optional)      │
│      → te_volatility, te_rolling_7d_std                    │
│                                                            │
│   4. POST /insight  { user_state + subscores + text }      │
│      → classification (state, signals, weak dims)          │
│      → prompt (system + user message, model, tokens, temp) │
│      → feedback_context (snapshot — store as-is)           │
│                                                            │
│   5. Empath calls Claude API with the prompt               │
│      → insight_text                                        │
│                                                            │
│   6. Show insight to user, store feedback_context          │
│                                                            │
│   ~~~ Next journal entry ~~~                               │
│                                                            │
│   7. POST /score (new entry)                               │
│                                                            │
│   8. POST /insight/feedback  { feedback_context + new_mes }│
│      → effective (bool), mes_delta, interpretation         │
│      → Store effectiveness                                 │
│                                                            │
│   9. POST /insight (new entry, with previous_insight)      │
│      → ... loop continues                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

The MES service (steps 1-4, 7-9) handles **all computation**. Empath handles **state storage, LLM calls, and user-facing presentation**.

---

## API Reference

Base URL: `https://mental-entropy-production.up.railway.app`

### POST /classify

Fast deterministic state classification. No LLM, no embeddings — runs in microseconds. Use this to track state on every entry, even if you don't generate an insight.

**Request:**
```json
{
  "user_state": {
    "current_mes": 58.25,
    "confidence": 0.93,
    "trend": "improving",
    "trend_slope": -2.5,
    "baseline_mes": 52.0,
    "n_entries": 12
  },
  "temporal_features": {
    "te_volatility": 0.30,
    "te_rolling_7d_std": 13.0
  },
  "current_subscores": {
    "prediction_coherence": 2.6,
    "model_complexity": 2.55,
    "compression_progress": 1.56,
    "belief_integration": 2.43,
    "precision_weighting": 2.47
  }
}
```

| Field | Required | Source | Description |
|-------|----------|--------|-------------|
| `user_state.current_mes` | ✅ | `/user-state` response | Decay-weighted current MES (0-100) |
| `user_state.confidence` | ✅ | `/user-state` response | Confidence in estimate (0-1) |
| `user_state.trend` | ✅ | `/user-state` response | `"improving"`, `"worsening"`, or `"stable"` |
| `user_state.trend_slope` | ✅ | `/user-state` response | Points per week (negative = improving) |
| `user_state.baseline_mes` | ✅ | `/user-state` response | All-time baseline MES |
| `user_state.n_entries` | ✅ | `/user-state` response | Number of entries in history |
| `temporal_features.te_volatility` | ❌ | `/temporal` response | Coefficient of variation |
| `temporal_features.te_rolling_7d_std` | ❌ | `/temporal` response | Rolling 7-day standard deviation |
| `current_subscores.*` | ❌ | `/score` response `subscores` | FEP dimension scores (1-5 each) |

**Response:**
```json
{
  "state": "overwhelm",
  "state_display": "Overwhelmed",
  "description": "High entropy with significant volatility — too much unintegrated information.",
  "confidence": 0.93,
  "primary_signals": [
    "current_mes=58.2 (above 55.0)",
    "volatility=0.30 (above 0.25)"
  ],
  "weakest_dimensions": [
    {
      "name": "compression_progress",
      "value": 1.56,
      "display_name": "Compression Progress",
      "description": "Active learning and insight formation..."
    }
  ],
  "directive": "Ground, simplify, help find anchors...",
  "avoid": "Do not introduce new complexity...",
  "target_direction": "decrease"
}
```

| Response Field | Type | Description |
|----------------|------|-------------|
| `state` | string | Machine-readable state ID |
| `state_display` | string | Human-readable label (show to user/therapist) |
| `description` | string | One-sentence explanation |
| `confidence` | float | Confidence in classification (0-1) |
| `primary_signals` | string[] | What triggered this classification |
| `weakest_dimensions` | object[] | Up to 2 subscore dimensions below threshold |
| `directive` | string | What therapeutic approach to take |
| `avoid` | string | What NOT to do in this state |
| `target_direction` | string | `"decrease"`, `"increase"`, or `"stable"` |

---

### POST /insight

Full prompt assembly. Returns everything needed to call an LLM and generate a state-appropriate insight.

**Request:**
```json
{
  "user_state": {
    "current_mes": 58.25,
    "confidence": 0.93,
    "trend": "worsening",
    "trend_slope": 9.4,
    "baseline_mes": 52.0,
    "n_entries": 12
  },
  "temporal_features": {
    "te_volatility": 0.30,
    "te_rolling_7d_std": 13.0
  },
  "current_subscores": {
    "prediction_coherence": 2.6,
    "model_complexity": 2.55,
    "compression_progress": 1.56,
    "belief_integration": 2.43,
    "precision_weighting": 2.47
  },
  "journal_text": "I keep going back and forth about whether to leave my job...",
  "previous_insight": {
    "text": "Last time we noticed you were juggling many threads...",
    "effectiveness": "effective"
  },
  "timestamp": "2026-04-01T10:00:00Z"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `user_state` | ✅ | Same as `/classify` |
| `temporal_features` | ❌ | Same as `/classify` |
| `current_subscores` | ❌ | Same as `/classify` |
| `journal_text` | ✅ | The journal entry text (10-50,000 chars) |
| `previous_insight.text` | ❌ | Text of the last insight shown to user |
| `previous_insight.effectiveness` | ❌ | `"effective"` or `"ineffective"` from feedback |
| `timestamp` | ❌ | ISO 8601 timestamp for the feedback context |

**Response:**
```json
{
  "classification": {
    "state": "overwhelm",
    "state_display": "Overwhelmed",
    "description": "...",
    "confidence": 0.93,
    "primary_signals": ["..."],
    "weakest_dimensions": [{"name": "...", "value": 1.56, "...": "..."}],
    "directive": "...",
    "avoid": "...",
    "target_direction": "decrease"
  },
  "prompt": {
    "system": "You are a therapeutic insight companion grounded in...",
    "user": "## Current State\n**Diagnosis:** Overwhelmed\n...\n## Journal Entry\n...",
    "model_recommendation": "claude-sonnet-4-20250514",
    "max_tokens": 300,
    "temperature": 0.5
  },
  "feedback_context": {
    "timestamp": "2026-04-01T10:00:00Z",
    "state": "overwhelm",
    "mes_at_insight": 58.25,
    "subscores_at_insight": {
      "prediction_coherence": 2.6,
      "compression_progress": 1.56
    },
    "target_direction": "decrease"
  }
}
```

**How to use the prompt:**

```javascript
// Send to Claude (or any LLM)
const response = await anthropic.messages.create({
  model: insight.prompt.model_recommendation,
  max_tokens: insight.prompt.max_tokens,
  temperature: insight.prompt.temperature,
  system: insight.prompt.system,
  messages: [{ role: 'user', content: insight.prompt.user }],
});

const insightText = response.content[0].text;
```

**How to store the feedback context:**

Store `feedback_context` as a JSON blob alongside the insight record. You'll send it back verbatim when the next entry comes in.

---

### POST /insight/feedback

Measures whether a previous insight moved MES in the right direction.

**Request:**
```json
{
  "feedback_context": {
    "timestamp": "2026-04-01T10:00:00Z",
    "state": "overwhelm",
    "mes_at_insight": 58.25,
    "subscores_at_insight": {
      "prediction_coherence": 2.6,
      "compression_progress": 1.56
    },
    "target_direction": "decrease"
  },
  "new_mes": 51.3,
  "new_subscores": {
    "prediction_coherence": 2.9,
    "compression_progress": 2.1
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `feedback_context` | ✅ | The exact `feedback_context` from the `/insight` response |
| `new_mes` | ✅ | MES score from the follow-up journal entry (0-100) |
| `new_subscores` | ❌ | Updated subscore values from the follow-up `/score` |

**Response:**
```json
{
  "effective": true,
  "mes_delta": -6.95,
  "target_direction": "decrease",
  "moved_correctly": true,
  "subscore_deltas": {
    "prediction_coherence": { "delta": 0.3, "improved": true },
    "compression_progress": { "delta": 0.54, "improved": true }
  },
  "interpretation": "MES decreased by 7.0 points — insight moved entropy in the right direction (target: decrease). 2 dimension(s) improved."
}
```

| Response Field | Type | Description |
|----------------|------|-------------|
| `effective` | bool | Whether the insight produced the desired effect |
| `mes_delta` | float | Change in MES (negative = decreased) |
| `target_direction` | string | What direction was expected |
| `moved_correctly` | bool | Whether MES moved in target direction |
| `subscore_deltas` | object | Per-dimension change tracking |
| `interpretation` | string | Human-readable summary |

**Effectiveness rules:**
- Target `"decrease"`: effective if MES dropped by more than 1.0 points
- Target `"increase"`: effective if MES rose by more than 1.0 points
- Target `"stable"`: effective if MES stayed within ±3.0 points

---

## Implementation Guide

### Database Schema Addition

Add one table for the feedback loop:

```sql
CREATE TABLE insight_records (
  id UUID PRIMARY KEY DEFAULT (UUID()),
  client_id UUID NOT NULL,
  journal_id UUID NOT NULL,

  -- Classification
  state VARCHAR(30) NOT NULL,
  state_display VARCHAR(50) NOT NULL,
  target_direction VARCHAR(10) NOT NULL,

  -- Stored prompt context (for debugging / analytics)
  feedback_context JSON NOT NULL,

  -- LLM output
  insight_text TEXT,

  -- Feedback (null until next entry closes the loop)
  effective BOOLEAN DEFAULT NULL,
  mes_delta FLOAT DEFAULT NULL,
  interpretation TEXT DEFAULT NULL,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  closed_at TIMESTAMP DEFAULT NULL,

  INDEX idx_client_pending (client_id, effective),
  INDEX idx_state (state),
  FOREIGN KEY (client_id) REFERENCES clients(client_id),
  FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
);
```

### Service Layer (Node.js)

```javascript
// services/insightService.js

const MES_API = process.env.MES_API_URL;
const Anthropic = require('@anthropic-ai/sdk');
const anthropic = new Anthropic();

/**
 * Generate a state-aware insight for a journal entry.
 *
 * Call this after scoring the entry and computing user state.
 * Returns the insight text and stores the feedback context.
 */
async function generateInsight({ clientId, journalId, journalText, score, userState, temporal }) {

  // 1. Close previous feedback loop
  const pending = await db.query(
    'SELECT id, feedback_context FROM insight_records WHERE client_id = ? AND effective IS NULL ORDER BY created_at DESC LIMIT 1',
    [clientId]
  );

  if (pending.length > 0) {
    const feedbackRes = await fetch(`${MES_API}/insight/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        feedback_context: pending[0].feedback_context,
        new_mes: score.mes_score,
        new_subscores: extractSubscoreValues(score.subscores),
      }),
    });
    const feedback = await feedbackRes.json();

    await db.query(
      'UPDATE insight_records SET effective = ?, mes_delta = ?, interpretation = ?, closed_at = NOW() WHERE id = ?',
      [feedback.effective, feedback.mes_delta, feedback.interpretation, pending[0].id]
    );
  }

  // 2. Build previous insight context
  const lastInsight = await db.query(
    'SELECT insight_text, effective FROM insight_records WHERE client_id = ? ORDER BY created_at DESC LIMIT 1',
    [clientId]
  );
  const previousInsight = lastInsight.length > 0 ? {
    text: lastInsight[0].insight_text,
    effectiveness: lastInsight[0].effective === null ? null
      : lastInsight[0].effective ? 'effective' : 'ineffective',
  } : null;

  // 3. Request insight prompt from MES service
  const insightRes = await fetch(`${MES_API}/insight`, {
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
      temporal_features: temporal ? {
        te_volatility: temporal.temporal_features.te_volatility,
        te_rolling_7d_std: temporal.temporal_features.te_rolling_7d_std,
      } : undefined,
      current_subscores: extractSubscoreValues(score.subscores),
      journal_text: journalText,
      previous_insight: previousInsight,
      timestamp: new Date().toISOString(),
    }),
  });
  const insight = await insightRes.json();

  // 4. Generate insight text via Claude
  const llmResponse = await anthropic.messages.create({
    model: insight.prompt.model_recommendation,
    max_tokens: insight.prompt.max_tokens,
    temperature: insight.prompt.temperature,
    system: insight.prompt.system,
    messages: [{ role: 'user', content: insight.prompt.user }],
  });
  const insightText = llmResponse.content[0].text;

  // 5. Store for feedback loop
  await db.query(
    `INSERT INTO insight_records (id, client_id, journal_id, state, state_display, target_direction, feedback_context, insight_text)
     VALUES (UUID(), ?, ?, ?, ?, ?, ?, ?)`,
    [
      clientId,
      journalId,
      insight.classification.state,
      insight.classification.state_display,
      insight.classification.target_direction,
      JSON.stringify(insight.feedback_context),
      insightText,
    ]
  );

  return {
    state: insight.classification.state,
    stateDisplay: insight.classification.state_display,
    description: insight.classification.description,
    targetDirection: insight.classification.target_direction,
    weakestDimensions: insight.classification.weakest_dimensions,
    insightText,
  };
}

/**
 * Extract numeric values from the subscores response.
 * /score returns { dim: { value, display_name, description } }
 * /insight expects { dim: value }
 */
function extractSubscoreValues(subscores) {
  if (!subscores) return undefined;
  const result = {};
  for (const [key, detail] of Object.entries(subscores)) {
    result[key] = detail.value;
  }
  return result;
}

module.exports = { generateInsight };
```

### Wiring into Journal Creation

Add insight generation to the existing journal processing pipeline:

```javascript
// controllers/journals/journalsController.js — in the create handler

// ... existing code: sentiment, title extraction, emotion detection ...

// After scoring and storing the journal:
const insightService = require('../../services/insightService');

// Only generate insights when appropriate (see "When to Generate" below)
if (shouldGenerateInsight(userState, lastInsightDate)) {
  const insight = await insightService.generateInsight({
    clientId,
    journalId: newJournal.journal_id,
    journalText: decryptedJournal.text,
    score,
    userState,
    temporal,  // from POST /temporal, or null
  });

  // Include in response to frontend
  response.insight = {
    state: insight.stateDisplay,
    text: insight.insightText,
    weakDimensions: insight.weakestDimensions,
  };
}
```

### When to Generate Insights

Don't generate on every single entry. Recommended triggers:

```javascript
function shouldGenerateInsight(userState, lastInsightDate) {
  // Always generate if this is a first-time or returning user
  if (!lastInsightDate) return true;

  // Generate if state changed since last insight
  // (requires storing last state — add to insight_records query)

  // Generate if enough time has passed (e.g., every 2+ entries)
  const hoursSinceLastInsight = (Date.now() - lastInsightDate) / (1000 * 60 * 60);
  if (hoursSinceLastInsight > 12) return true;

  // Generate if MES crossed a threshold
  if (userState.current_mes > 55 || userState.current_mes < 35) return true;

  // Don't overwhelm the user with insights
  return false;
}
```

---

## Frontend Presentation

### What to Show the User

| State | Show to user | UI treatment |
|-------|-------------|--------------|
| `overwhelm` | `state_display` + short insight | Calming colors, minimal UI |
| `stuck` | `state_display` + insight with a question | Encouraging tone |
| `rigidity` | `state_display` + gentle insight | Warm, curious tone |
| `active_integration` | `state_display` + celebratory insight | Positive reinforcement |
| `baseline` | `state_display` + reflective insight | Neutral, balanced |
| `insufficient_data` | "Keep journaling" encouragement | Warm, no diagnostic claims |

### What to Show the Therapist

Therapists get the full picture:

```json
{
  "state": "overwhelm",
  "confidence": 0.93,
  "primary_signals": ["current_mes=58.2 (above 55.0)", "volatility=0.30 (above 0.25)"],
  "weakest_dimensions": [
    { "display_name": "Compression Progress", "value": 1.56 },
    { "display_name": "Belief Integration", "value": 2.43 }
  ],
  "target_direction": "decrease",
  "directive": "Ground, simplify, help find anchors...",
  "avoid": "Do not introduce new complexity..."
}
```

### Subscores — FEP Dimension Display Names

The subscores returned by `/score` have user-friendly display names:

| Internal Name | Display Name | What It Means |
|---------------|-------------|---------------|
| `prediction_coherence` | Thought Flow | How smoothly thoughts connect |
| `model_complexity` | Focus | How well organized around key themes |
| `compression_progress` | Insight | Whether the user moved toward understanding |
| `belief_integration` | Integration | How well contradictions are held with awareness |
| `precision_weighting` | Clarity | How decisively the user expresses themselves |

---

## Analytics & Learning

Track insight effectiveness over time to learn what works:

```sql
-- Effectiveness rate by state
SELECT state,
       COUNT(*) as total,
       SUM(effective) as effective_count,
       ROUND(AVG(effective) * 100, 1) as effectiveness_rate
FROM insight_records
WHERE effective IS NOT NULL
GROUP BY state;

-- Average MES movement by state
SELECT state,
       ROUND(AVG(mes_delta), 1) as avg_mes_delta,
       ROUND(AVG(CASE WHEN effective THEN mes_delta END), 1) as avg_effective_delta
FROM insight_records
WHERE effective IS NOT NULL
GROUP BY state;

-- Which dimensions improve most after insights
-- (requires storing subscore_deltas — extend the table if needed)
```

---

## Error Handling

All insight endpoints return standard HTTP errors:

| Status | When | Action |
|--------|------|--------|
| 200 | Success | Use the response |
| 422 | Validation error (e.g., text too short) | Show user-friendly message |
| 500 | Internal error | Fall back to no insight, log for debugging |

**Graceful degradation**: If any insight endpoint fails, the app should still work — just without the insight. Scoring and user-state are independent.

```javascript
try {
  const insight = await insightService.generateInsight({ ... });
  response.insight = insight;
} catch (err) {
  logger.error('Insight generation failed', err);
  // App continues without insight — score and journal are already saved
}
```

---

## Environment Variables

Add to Empath's `.env`:

```env
# MES Service
MES_API_URL=https://mental-entropy-production.up.railway.app

# Anthropic (for LLM insight generation)
ANTHROPIC_API_KEY=sk-ant-...
```

No changes needed to the MES service — it's already deployed.
