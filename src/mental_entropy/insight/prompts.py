"""Prompt assembly for state-aware LLM insight generation.

Composes system + user prompts from the classification result, journal text,
and optional previous insight context. The MES service does NOT call LLMs —
it returns ready-to-send prompt payloads for the consuming backend.
"""

from __future__ import annotations

from mental_entropy.insight.classify import (
    DIMENSION_DESCRIPTIONS,
    DIMENSION_DISPLAY_NAMES,
    classify_mental_state,
)
from mental_entropy.insight.types import (
    ClassificationResult,
    FeedbackContext,
    InsightPrompt,
    MentalState,
    STATE_TARGET_DIRECTION,
    WeakDimension,
)

# ---------------------------------------------------------------------------
# System prompt building blocks
# ---------------------------------------------------------------------------

_BASE_CONTEXT = """\
You are a therapeutic insight companion grounded in Karl Friston's Free Energy \
Principle. Your role is to help the user reduce unnecessary mental entropy — \
not to zero (that would be rigidity), but toward active compression where \
experience is being integrated into a coherent internal model.

Mental entropy is not inherently bad. The goal is dynamic equilibrium: \
the mind actively processing and compressing experience, not frozen and not \
overwhelmed. You measure success by whether entropy moves in the right \
direction for the user's current state.

Guidelines:
- Be concise and specific. Avoid generic platitudes.
- Ground insights in what the user actually wrote.
- Name patterns, don't just describe feelings.
- One clear insight is better than three vague ones.
- Never diagnose mental health conditions.
- Never prescribe medication or specific therapies.
- If the user appears in crisis, acknowledge it and suggest professional help.\
"""

_STATE_POSTURES: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: """\
The user has very few journal entries. You don't have enough data to identify \
patterns. Be warm and encouraging. Invite them to keep journaling. Do NOT \
make claims about their mental state based on insufficient evidence.\
""",
    MentalState.ACTIVE_INTEGRATION: """\
The user's mental entropy is declining — their mind is actively compressing \
experience into a more coherent model. This is a positive trajectory. \
Your role is to reflect and deepen: name the insight that appears to be \
forming, help consolidate gains, and celebrate the progress without \
disrupting the integration process.\
""",
    MentalState.OVERWHELM: """\
The user's mental entropy is high and volatile — too much unintegrated \
information. Their mind is struggling to process. Your role is to ground \
and simplify: offer one clear anchor point, reduce cognitive load, and \
help them find solid ground. Keep your response SHORT and CONCRETE.\
""",
    MentalState.STUCK: """\
The user's mental entropy is persistently high without movement — their \
internal model isn't updating despite elevated disorganization. Your role \
is to gently challenge: surface one specific pattern they might not be \
seeing, create a small opening for movement. Don't add more information; \
redirect attention.\
""",
    MentalState.RIGIDITY: """\
The user's mental entropy is very low and flat — their world model may be \
over-compressed, filtering out new information rather than processing it. \
Your role is to carefully introduce novelty: ask one gentle question that \
challenges a single assumption. Be curious, not confrontational.\
""",
    MentalState.BASELINE: """\
The user's mental entropy is in a healthy range with normal fluctuation. \
There is no pathology to address. Provide balanced reflective support — \
help them maintain self-awareness and recognize patterns, without \
over-pathologizing normal human experience.\
""",
}

# Model recommendations per state
_MODEL_RECOMMENDATIONS: dict[MentalState, str] = {
    MentalState.INSUFFICIENT_DATA: "claude-sonnet-4-20250514",
    MentalState.ACTIVE_INTEGRATION: "claude-sonnet-4-20250514",
    MentalState.OVERWHELM: "claude-sonnet-4-20250514",
    MentalState.STUCK: "claude-sonnet-4-20250514",
    MentalState.RIGIDITY: "claude-sonnet-4-20250514",
    MentalState.BASELINE: "claude-sonnet-4-20250514",
}

# Max tokens per state (shorter for overwhelm)
_MAX_TOKENS: dict[MentalState, int] = {
    MentalState.INSUFFICIENT_DATA: 300,
    MentalState.ACTIVE_INTEGRATION: 500,
    MentalState.OVERWHELM: 300,
    MentalState.STUCK: 500,
    MentalState.RIGIDITY: 500,
    MentalState.BASELINE: 400,
}

# Temperature per state (lower for overwhelm = more focused)
_TEMPERATURES: dict[MentalState, float] = {
    MentalState.INSUFFICIENT_DATA: 0.7,
    MentalState.ACTIVE_INTEGRATION: 0.7,
    MentalState.OVERWHELM: 0.5,
    MentalState.STUCK: 0.8,
    MentalState.RIGIDITY: 0.8,
    MentalState.BASELINE: 0.7,
}


# ---------------------------------------------------------------------------
# Dimension targeting
# ---------------------------------------------------------------------------


def _dimension_instructions(weak_dims: list[WeakDimension]) -> str:
    """Build targeting instructions for the weakest subscore dimensions."""
    if not weak_dims:
        return ""

    lines = ["\n## Dimension Targeting"]
    lines.append(
        "The following dimensions are below threshold. "
        "Focus your insight on improving these areas:"
    )
    for dim in weak_dims:
        display = dim.display_name
        desc = dim.description or DIMENSION_DESCRIPTIONS.get(dim.name, "")
        lines.append(f"- **{display}** (score: {dim.value}/5.0): {desc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User message construction
# ---------------------------------------------------------------------------


def _build_user_message(
    classification: ClassificationResult,
    journal_text: str,
    previous_insight: dict[str, str] | None = None,
) -> str:
    """Build the user message with state context and journal text."""
    sections: list[str] = []

    # Current state summary
    sections.append("## Current State")
    sections.append(f"**Diagnosis:** {classification.state_display}")
    sections.append(f"**Description:** {classification.description}")
    sections.append(f"**Signals:** {'; '.join(classification.primary_signals)}")
    sections.append(f"**Target direction:** MES should {classification.target_direction}")

    # Weak dimensions
    if classification.weakest_dimensions:
        dim_lines = []
        for d in classification.weakest_dimensions:
            dim_lines.append(f"  - {d.display_name}: {d.value}/5.0")
        sections.append("**Weakest dimensions:**\n" + "\n".join(dim_lines))

    # Previous insight context
    if previous_insight:
        sections.append("\n## Previous Insight")
        if "text" in previous_insight:
            sections.append(f"Last insight: {previous_insight['text']}")
        if "effectiveness" in previous_insight:
            sections.append(f"Effectiveness: {previous_insight['effectiveness']}")

    # Journal entry
    sections.append("\n## Journal Entry")
    sections.append(journal_text)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_insight_prompt(
    *,
    current_mes: float,
    confidence: float,
    trend: str,
    trend_slope: float,
    baseline_mes: float,
    n_entries: int,
    volatility: float | None = None,
    rolling_7d_std: float | None = None,
    current_subscores: dict[str, float] | None = None,
    journal_text: str,
    previous_insight: dict[str, str] | None = None,
    timestamp: str | None = None,
) -> tuple[ClassificationResult, InsightPrompt, FeedbackContext]:
    """Assemble a complete insight prompt payload.

    This is the main entry point for prompt generation. It:
    1. Classifies the user's mental state
    2. Assembles state-aware system + user prompts
    3. Prepares a feedback context snapshot for the feedback loop

    Args:
        current_mes: Current MES score (0-100).
        confidence: Confidence in the current MES (0-1).
        trend: Trend direction ("improving", "worsening", "stable").
        trend_slope: MES slope in points per week.
        baseline_mes: User's all-time baseline MES.
        n_entries: Number of journal entries in history.
        volatility: Coefficient of variation of MES scores.
        rolling_7d_std: Rolling 7-day standard deviation.
        current_subscores: Dict mapping FEP dimension names to scores (1-5).
        journal_text: The journal entry text to generate insight for.
        previous_insight: Optional dict with "text" and "effectiveness" keys.
        timestamp: ISO 8601 timestamp for the feedback context.

    Returns:
        Tuple of (ClassificationResult, InsightPrompt, FeedbackContext).
    """
    # Step 1: Classify
    classification = classify_mental_state(
        current_mes=current_mes,
        confidence=confidence,
        trend=trend,
        trend_slope=trend_slope,
        baseline_mes=baseline_mes,
        n_entries=n_entries,
        volatility=volatility,
        rolling_7d_std=rolling_7d_std,
        current_subscores=current_subscores,
    )

    # Step 2: Build system prompt
    system_parts = [
        _BASE_CONTEXT,
        "\n## State-Specific Guidance",
        _STATE_POSTURES[classification.state],
        _dimension_instructions(classification.weakest_dimensions),
        f"\n## Constraints",
        f"Directive: {classification.directive}",
        f"Avoid: {classification.avoid}",
    ]
    system_prompt = "\n".join(system_parts)

    # Step 3: Build user message
    user_prompt = _build_user_message(
        classification, journal_text, previous_insight,
    )

    # Step 4: Build prompt payload
    state = classification.state
    prompt = InsightPrompt(
        system=system_prompt,
        user=user_prompt,
        model_recommendation=_MODEL_RECOMMENDATIONS[state],
        max_tokens=_MAX_TOKENS[state],
        temperature=_TEMPERATURES[state],
    )

    # Step 5: Build feedback context
    subscores = current_subscores or {}
    feedback_ctx = FeedbackContext(
        timestamp=timestamp or "",
        state=classification.state.value,
        mes_at_insight=current_mes,
        subscores_at_insight=dict(subscores),
        target_direction=STATE_TARGET_DIRECTION[classification.state],
    )

    return classification, prompt, feedback_ctx
