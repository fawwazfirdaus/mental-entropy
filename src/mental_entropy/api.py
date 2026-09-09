"""MES (Mental Entropy Score) REST API.

FastAPI service exposing the MES scoring pipeline as HTTP endpoints.
Standalone local interface for research experiments and scoring demos.

Usage:
    # Install API dependencies
    uv sync --extra api

    # Run the server
    PYTHONPATH=src .venv/bin/python -m mental_entropy.api

    # Or with environment variables
    MES_PORT=8000 MES_HOST=0.0.0.0 mes-api

Endpoints:
    POST /score         - Score a single journal entry
    POST /score/batch   - Score multiple entries
    POST /temporal      - Compute temporal entropy features
    GET  /health        - Health check + model info
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("mes-api")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50_000)


class SubscoreDetail(BaseModel):
    value: float
    display_name: str
    description: str


class ScoreResponse(BaseModel):
    mes_score: float
    interpretation: str
    features: dict[str, float]
    subscores: dict[str, SubscoreDetail] | None = None


class BatchEntry(BaseModel):
    id: str
    text: str = Field(..., min_length=10, max_length=50_000)


class BatchRequest(BaseModel):
    entries: list[BatchEntry] = Field(..., min_length=1, max_length=50)


class BatchResultItem(BaseModel):
    id: str
    mes_score: float
    interpretation: str


class BatchResponse(BaseModel):
    results: list[BatchResultItem]


class TemporalEntry(BaseModel):
    timestamp: str  # ISO 8601
    mes_score: float


class TemporalRequest(BaseModel):
    entries: list[TemporalEntry] = Field(..., min_length=2)


class TemporalResponse(BaseModel):
    temporal_features: dict[str, int | float]
    trend: str
    summary: str


class UserStateEntry(BaseModel):
    timestamp: str  # ISO 8601
    mes_score: float = Field(..., ge=0.0, le=100.0)


class UserStateRequest(BaseModel):
    entries: list[UserStateEntry] = Field(..., min_length=1)
    half_life_days: float = Field(default=14.0, ge=1.0, le=90.0)


class UserStateResponse(BaseModel):
    current_mes: float
    confidence: float
    trend: str
    trend_slope: float
    baseline_mes: float
    deviation_from_baseline: float
    n_entries: int
    n_recent_entries: int
    days_since_last_entry: float
    interpretation: str
    status: str


class HealthResponse(BaseModel):
    status: str
    model: str
    architecture: str
    n_training_entries: int
    human_corr: float
    version: str


# -- Insight engine request/response models --------------------------------


class ClassifyUserState(BaseModel):
    current_mes: float = Field(..., ge=0.0, le=100.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    trend: str = Field(..., pattern="^(improving|worsening|stable)$")
    trend_slope: float
    baseline_mes: float = Field(..., ge=0.0, le=100.0)
    n_entries: int = Field(..., ge=0)


class ClassifyTemporalFeatures(BaseModel):
    te_volatility: float | None = None
    te_rolling_7d_std: float | None = None


class ClassifySubscores(BaseModel):
    prediction_coherence: float | None = None
    model_complexity: float | None = None
    compression_progress: float | None = None
    belief_integration: float | None = None
    precision_weighting: float | None = None


class ClassifyRequest(BaseModel):
    user_state: ClassifyUserState
    temporal_features: ClassifyTemporalFeatures | None = None
    current_subscores: ClassifySubscores | None = None


class WeakDimensionResponse(BaseModel):
    name: str
    value: float
    display_name: str
    description: str


class ClassifyResponse(BaseModel):
    state: str
    state_display: str
    description: str
    confidence: float
    primary_signals: list[str]
    weakest_dimensions: list[WeakDimensionResponse]
    directive: str
    avoid: str
    target_direction: str


class InsightPreviousInsight(BaseModel):
    text: str | None = None
    effectiveness: str | None = None


class InsightRequest(BaseModel):
    user_state: ClassifyUserState
    temporal_features: ClassifyTemporalFeatures | None = None
    current_subscores: ClassifySubscores | None = None
    journal_text: str = Field(..., min_length=10, max_length=50_000)
    previous_insight: InsightPreviousInsight | None = None
    timestamp: str | None = None


class InsightPromptResponse(BaseModel):
    system: str
    user: str
    model_recommendation: str
    max_tokens: int
    temperature: float


class InsightFeedbackContextResponse(BaseModel):
    timestamp: str
    state: str
    mes_at_insight: float
    subscores_at_insight: dict[str, float]
    target_direction: str


class InsightResponse(BaseModel):
    classification: ClassifyResponse
    prompt: InsightPromptResponse
    feedback_context: InsightFeedbackContextResponse


class FeedbackContextRequest(BaseModel):
    timestamp: str
    state: str
    mes_at_insight: float
    subscores_at_insight: dict[str, float]
    target_direction: str


class FeedbackRequest(BaseModel):
    feedback_context: FeedbackContextRequest
    new_mes: float = Field(..., ge=0.0, le=100.0)
    new_subscores: dict[str, float] | None = None


class SubscoreDeltaResponse(BaseModel):
    delta: float
    improved: bool


class FeedbackResponse(BaseModel):
    effective: bool
    mes_delta: float
    target_direction: str
    moved_correctly: bool
    subscore_deltas: dict[str, SubscoreDeltaResponse]
    interpretation: str


# ---------------------------------------------------------------------------
# Model preloading
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_initialized = False


def _preload_models() -> None:
    """Load embedding model and scoring artifacts on startup."""
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        logger.info("Preloading MES models...")
        t0 = time.time()

        # Force-load the embedding model by embedding a dummy sentence
        from mental_entropy.embedding.embed import embed_journal_entry
        embed_journal_entry("Preload warmup sentence for model initialization.")

        # Force-load the hybrid Ridge model
        from mental_entropy.models._registry import (
            hybrid_model_available,
            _load_hybrid_model,
        )
        if hybrid_model_available():
            _load_hybrid_model()
            logger.info("Hybrid v6 model loaded")

        elapsed = time.time() - t0
        logger.info(f"Models preloaded in {elapsed:.1f}s")
        _initialized = True


def _get_manifest() -> dict[str, Any]:
    """Load model manifest metadata."""
    manifest_path = (
        Path(__file__).parent / "models" / "_artifacts" / "manifest.json"
    )
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preload models on startup."""
    _preload_models()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MES API",
        description="Mental Entropy Score — journal text analysis",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # CORS origins are configurable for local clients.
    cors_origins = os.environ.get("MES_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # POST /score
    # ------------------------------------------------------------------
    @app.post("/score", response_model=ScoreResponse)
    async def score_entry(req: ScoreRequest) -> ScoreResponse:
        """Score a single journal entry."""
        from mental_entropy.score import compute_mes_from_text, interpret_mes_score

        try:
            result = compute_mes_from_text(req.text, method="auto")
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

        mes_score = result.pop("mes_score")
        interpretation = interpret_mes_score(mes_score)

        # Separate subscores from features
        raw_subscores: dict[str, float] = {}
        features: dict[str, float] = {}
        for k, v in result.items():
            if k.endswith("_subscore"):
                raw_subscores[k] = v
            else:
                features[k] = v

        # Enrich subscores with display names
        subscores: dict[str, SubscoreDetail] | None = None
        if raw_subscores:
            try:
                from mental_entropy.models._registry import get_dimension_display
                display = get_dimension_display()
            except Exception:
                display = {}

            subscores = {}
            for k, v in raw_subscores.items():
                dim_name = k.removesuffix("_subscore")
                info = display.get(dim_name, {})
                subscores[dim_name] = SubscoreDetail(
                    value=round(v, 2),
                    display_name=info.get("name", dim_name.replace("_", " ").title()),
                    description=info.get("description", ""),
                )

        return ScoreResponse(
            mes_score=round(mes_score, 2),
            interpretation=interpretation,
            features=features,
            subscores=subscores,
        )

    # ------------------------------------------------------------------
    # POST /score/batch
    # ------------------------------------------------------------------
    @app.post("/score/batch", response_model=BatchResponse)
    async def score_batch(req: BatchRequest) -> BatchResponse:
        """Score multiple journal entries. Max 50 per request."""
        from mental_entropy.score import compute_mes_from_text, interpret_mes_score

        results: list[BatchResultItem] = []
        for entry in req.entries:
            try:
                result = compute_mes_from_text(entry.text, method="auto")
                mes_score = result["mes_score"]
                results.append(BatchResultItem(
                    id=entry.id,
                    mes_score=round(mes_score, 2),
                    interpretation=interpret_mes_score(mes_score),
                ))
            except Exception as e:
                logger.error(f"Batch entry {entry.id} failed: {e}")
                results.append(BatchResultItem(
                    id=entry.id,
                    mes_score=-1.0,
                    interpretation=f"Error: {e}",
                ))

        return BatchResponse(results=results)

    # ------------------------------------------------------------------
    # POST /temporal
    # ------------------------------------------------------------------
    @app.post("/temporal", response_model=TemporalResponse)
    async def temporal_analysis(req: TemporalRequest) -> TemporalResponse:
        """Compute temporal entropy features for a client's journal history."""
        from mental_entropy.temporal.te import te_features

        # Parse timestamps and build input tuples
        entries: list[tuple[datetime, float, None]] = []
        for e in req.entries:
            try:
                ts = datetime.fromisoformat(e.timestamp)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid timestamp: {e.timestamp}. Use ISO 8601 format.",
                )
            entries.append((ts, e.mes_score, None))

        try:
            features = te_features(entries, sort=True)
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Temporal error: {e}")

        # Build trend summary
        direction = features.get("te_trend_direction", 0)
        if direction == -1:
            trend = "improving"
        elif direction == 1:
            trend = "worsening"
        else:
            trend = "stable"

        slope_week = features.get("te_slope_per_week", 0.0)
        volatility = features.get("te_volatility", 0.0)
        n = features.get("te_n_entries", 0)
        span = features.get("te_time_span_days", 0.0)

        summary = (
            f"Based on {n} entries over {span:.0f} days: "
            f"trend is {trend} ({slope_week:+.1f} points/week), "
            f"volatility {'high' if volatility > 0.3 else 'moderate' if volatility > 0.15 else 'low'} "
            f"(CV={volatility:.2f})."
        )

        return TemporalResponse(
            temporal_features=features,
            trend=trend,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # POST /user-state
    # ------------------------------------------------------------------
    @app.post("/user-state", response_model=UserStateResponse)
    async def user_state(req: UserStateRequest) -> UserStateResponse:
        """Compute a user's current mental entropy state.

        Uses exponential decay weighting so recent journal entries count
        more than old ones. Returns a current MES score, confidence level,
        trend, and status indicator.
        """
        from mental_entropy.temporal.te import compute_user_state

        # Parse timestamps
        entries: list[tuple[datetime, float, None]] = []
        for e in req.entries:
            try:
                ts = datetime.fromisoformat(e.timestamp)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid timestamp: {e.timestamp}. Use ISO 8601 format.",
                )
            entries.append((ts, e.mes_score, None))

        try:
            result = compute_user_state(
                entries,
                half_life_days=req.half_life_days,
            )
        except Exception as e:
            logger.error(f"User state computation failed: {e}")
            raise HTTPException(status_code=500, detail=f"User state error: {e}")

        return UserStateResponse(**result)

    # ------------------------------------------------------------------
    # POST /classify
    # ------------------------------------------------------------------
    @app.post("/classify", response_model=ClassifyResponse)
    async def classify_state(req: ClassifyRequest) -> ClassifyResponse:
        """Classify a user's mental state from MES signals."""
        from mental_entropy.insight.classify import classify_mental_state

        subscores: dict[str, float] | None = None
        if req.current_subscores:
            subscores = {
                k: v
                for k, v in req.current_subscores.model_dump().items()
                if v is not None
            }

        volatility = req.temporal_features.te_volatility if req.temporal_features else None
        rolling_std = req.temporal_features.te_rolling_7d_std if req.temporal_features else None

        try:
            result = classify_mental_state(
                current_mes=req.user_state.current_mes,
                confidence=req.user_state.confidence,
                trend=req.user_state.trend,
                trend_slope=req.user_state.trend_slope,
                baseline_mes=req.user_state.baseline_mes,
                n_entries=req.user_state.n_entries,
                volatility=volatility,
                rolling_7d_std=rolling_std,
                current_subscores=subscores,
            )
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Classification error: {e}")

        return ClassifyResponse(
            state=result.state.value,
            state_display=result.state_display,
            description=result.description,
            confidence=result.confidence,
            primary_signals=result.primary_signals,
            weakest_dimensions=[
                WeakDimensionResponse(
                    name=d.name,
                    value=d.value,
                    display_name=d.display_name,
                    description=d.description,
                )
                for d in result.weakest_dimensions
            ],
            directive=result.directive,
            avoid=result.avoid,
            target_direction=result.target_direction,
        )

    # ------------------------------------------------------------------
    # POST /insight
    # ------------------------------------------------------------------
    @app.post("/insight", response_model=InsightResponse)
    async def insight(req: InsightRequest) -> InsightResponse:
        """Assemble a state-aware LLM prompt for insight generation."""
        from mental_entropy.insight.prompts import assemble_insight_prompt

        subscores: dict[str, float] | None = None
        if req.current_subscores:
            subscores = {
                k: v
                for k, v in req.current_subscores.model_dump().items()
                if v is not None
            }

        volatility = req.temporal_features.te_volatility if req.temporal_features else None
        rolling_std = req.temporal_features.te_rolling_7d_std if req.temporal_features else None

        prev_insight: dict[str, str] | None = None
        if req.previous_insight:
            prev_insight = {
                k: v
                for k, v in req.previous_insight.model_dump().items()
                if v is not None
            }

        try:
            classification, prompt, feedback_ctx = assemble_insight_prompt(
                current_mes=req.user_state.current_mes,
                confidence=req.user_state.confidence,
                trend=req.user_state.trend,
                trend_slope=req.user_state.trend_slope,
                baseline_mes=req.user_state.baseline_mes,
                n_entries=req.user_state.n_entries,
                volatility=volatility,
                rolling_7d_std=rolling_std,
                current_subscores=subscores,
                journal_text=req.journal_text,
                previous_insight=prev_insight,
                timestamp=req.timestamp,
            )
        except Exception as e:
            logger.error(f"Insight assembly failed: {e}")
            raise HTTPException(status_code=500, detail=f"Insight error: {e}")

        return InsightResponse(
            classification=ClassifyResponse(
                state=classification.state.value,
                state_display=classification.state_display,
                description=classification.description,
                confidence=classification.confidence,
                primary_signals=classification.primary_signals,
                weakest_dimensions=[
                    WeakDimensionResponse(
                        name=d.name,
                        value=d.value,
                        display_name=d.display_name,
                        description=d.description,
                    )
                    for d in classification.weakest_dimensions
                ],
                directive=classification.directive,
                avoid=classification.avoid,
                target_direction=classification.target_direction,
            ),
            prompt=InsightPromptResponse(
                system=prompt.system,
                user=prompt.user,
                model_recommendation=prompt.model_recommendation,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
            ),
            feedback_context=InsightFeedbackContextResponse(
                timestamp=feedback_ctx.timestamp,
                state=feedback_ctx.state,
                mes_at_insight=feedback_ctx.mes_at_insight,
                subscores_at_insight=feedback_ctx.subscores_at_insight,
                target_direction=feedback_ctx.target_direction,
            ),
        )

    # ------------------------------------------------------------------
    # POST /insight/feedback
    # ------------------------------------------------------------------
    @app.post("/insight/feedback", response_model=FeedbackResponse)
    async def insight_feedback(req: FeedbackRequest) -> FeedbackResponse:
        """Measure effectiveness of a previous insight."""
        from mental_entropy.insight.feedback import compute_feedback
        from mental_entropy.insight.types import FeedbackContext

        ctx = FeedbackContext(
            timestamp=req.feedback_context.timestamp,
            state=req.feedback_context.state,
            mes_at_insight=req.feedback_context.mes_at_insight,
            subscores_at_insight=req.feedback_context.subscores_at_insight,
            target_direction=req.feedback_context.target_direction,
        )

        try:
            result = compute_feedback(
                feedback_context=ctx,
                new_mes=req.new_mes,
                new_subscores=req.new_subscores,
            )
        except Exception as e:
            logger.error(f"Feedback computation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Feedback error: {e}")

        return FeedbackResponse(
            effective=result.effective,
            mes_delta=result.mes_delta,
            target_direction=result.target_direction,
            moved_correctly=result.moved_correctly,
            subscore_deltas={
                k: SubscoreDeltaResponse(delta=v.delta, improved=v.improved)
                for k, v in result.subscore_deltas.items()
            },
            interpretation=result.interpretation,
        )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check with model status."""
        manifest = _get_manifest()

        metrics = manifest.get("metrics", {})
        arch = manifest.get("architecture", "unknown")
        # v7 stores human_corr under a different key
        h_corr = metrics.get("v7_human_corr", metrics.get("oof_human_corr", 0.0))
        return HealthResponse(
            status="ok",
            model=arch,
            architecture=arch,
            n_training_entries=manifest.get("n_entries", 0),
            human_corr=round(h_corr, 4),
            version="1.0.0",
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()


def main() -> None:
    """Run the MES API server."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    host = os.environ.get("MES_HOST", "0.0.0.0")
    # Railway sets PORT env var; fall back to MES_PORT then 8000
    port = int(os.environ.get("PORT", os.environ.get("MES_PORT", "8000")))
    workers = int(os.environ.get("MES_WORKERS", "1"))

    logger.info(f"Starting MES API on {host}:{port} (workers={workers})")
    uvicorn.run(
        "mental_entropy.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
