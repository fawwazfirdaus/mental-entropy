"""MES Insight Engine — state classification, prompt assembly, and feedback loop.

The insight engine is a closed-loop system that:
1. Diagnoses the user's mental state from MES signals (classify)
2. Assembles state-aware LLM prompts for insight generation (prompts)
3. Tracks whether insights moved MES in the right direction (feedback)

The MES service does NOT call LLMs. It returns deterministic classifications
and ready-to-send prompt payloads. The consuming backend calls its own LLM.
"""

from __future__ import annotations

from mental_entropy.insight.types import (
    MentalState,
    ClassificationResult,
    WeakDimension,
    InsightPrompt,
    FeedbackContext,
    FeedbackResult,
    SubscoreDelta,
)
from mental_entropy.insight.classify import classify_mental_state
from mental_entropy.insight.prompts import assemble_insight_prompt
from mental_entropy.insight.feedback import compute_feedback

__all__ = [
    "MentalState",
    "ClassificationResult",
    "WeakDimension",
    "InsightPrompt",
    "FeedbackContext",
    "FeedbackResult",
    "SubscoreDelta",
    "classify_mental_state",
    "assemble_insight_prompt",
    "compute_feedback",
]
