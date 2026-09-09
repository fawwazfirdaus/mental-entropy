"""Tests for the MES REST API."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mental_entropy.api import create_app


@pytest.fixture(scope="module")
def client():
    """Create a test client with preloaded models."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["architecture"] in ("hybrid_v6", "subscore_embedding_v7")
        assert data["n_training_entries"] > 0
        assert data["human_corr"] > 0

    def test_health_has_version(self, client: TestClient):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


# ---------------------------------------------------------------------------
# POST /score
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_valid_text(self, client: TestClient):
        resp = client.post("/score", json={
            "text": "Today I went for a walk in the park. The weather was nice and I felt calm."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["mes_score"] <= 100
        assert len(data["interpretation"]) > 0
        assert "features" in data
        assert len(data["features"]) > 0

    def test_score_high_entropy_text(self, client: TestClient):
        resp = client.post("/score", json={
            "text": (
                "I cant think straight everything jumping between work and bills "
                "and that argument and did I forget something I cant remember "
                "what was I doing the laundry or was it dishes no wait the car "
                "needs fixing too and my head hurts and nothing makes sense"
            )
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mes_score"] > 30  # Should be elevated

    def test_score_low_entropy_text(self, client: TestClient):
        resp = client.post("/score", json={
            "text": (
                "Today was a productive day. I started with a morning run, "
                "then completed three tasks at work. After lunch I had a "
                "good meeting with my team. In the evening I read a book "
                "and went to bed early feeling satisfied with the day."
            )
        })
        assert resp.status_code == 200
        data = resp.json()
        # Low entropy text should score lower than high entropy
        assert data["mes_score"] < 70

    def test_score_too_short(self, client: TestClient):
        resp = client.post("/score", json={"text": "Hi"})
        assert resp.status_code == 422  # Validation error

    def test_score_empty_text(self, client: TestClient):
        resp = client.post("/score", json={"text": ""})
        assert resp.status_code == 422

    def test_score_missing_text(self, client: TestClient):
        resp = client.post("/score", json={})
        assert resp.status_code == 422

    def test_score_features_are_floats(self, client: TestClient):
        resp = client.post("/score", json={
            "text": "Today was a calm day. I organized my thoughts and wrote in my journal."
        })
        data = resp.json()
        for k, v in data["features"].items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"


# ---------------------------------------------------------------------------
# POST /score/batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_multiple_entries(self, client: TestClient):
        resp = client.post("/score/batch", json={
            "entries": [
                {"id": "a", "text": "Today was calm and organized. I felt at peace with myself."},
                {"id": "b", "text": "Everything is chaotic and nothing makes sense anymore I cant focus."},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "a"
        assert data["results"][1]["id"] == "b"
        # Both should have valid scores
        for r in data["results"]:
            assert 0 <= r["mes_score"] <= 100
            assert len(r["interpretation"]) > 0

    def test_batch_empty_list(self, client: TestClient):
        resp = client.post("/score/batch", json={"entries": []})
        assert resp.status_code == 422

    def test_batch_single_entry(self, client: TestClient):
        resp = client.post("/score/batch", json={
            "entries": [
                {"id": "x", "text": "A simple journal entry about my day at work."},
            ]
        })
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1


# ---------------------------------------------------------------------------
# POST /temporal
# ---------------------------------------------------------------------------


class TestTemporal:
    def test_temporal_valid_history(self, client: TestClient):
        resp = client.post("/temporal", json={
            "entries": [
                {"timestamp": "2024-01-01T10:00:00", "mes_score": 45.0},
                {"timestamp": "2024-01-02T10:00:00", "mes_score": 50.0},
                {"timestamp": "2024-01-03T10:00:00", "mes_score": 42.0},
                {"timestamp": "2024-01-04T10:00:00", "mes_score": 38.0},
                {"timestamp": "2024-01-05T10:00:00", "mes_score": 35.0},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "temporal_features" in data
        assert len(data["temporal_features"]) == 21
        assert data["trend"] in ("improving", "worsening", "stable")
        assert len(data["summary"]) > 0

    def test_temporal_improving_trend(self, client: TestClient):
        # Scores going down = improving (lower entropy)
        resp = client.post("/temporal", json={
            "entries": [
                {"timestamp": "2024-01-01T10:00:00", "mes_score": 80.0},
                {"timestamp": "2024-01-02T10:00:00", "mes_score": 70.0},
                {"timestamp": "2024-01-03T10:00:00", "mes_score": 60.0},
                {"timestamp": "2024-01-04T10:00:00", "mes_score": 50.0},
                {"timestamp": "2024-01-05T10:00:00", "mes_score": 40.0},
            ]
        })
        data = resp.json()
        assert data["trend"] == "improving"

    def test_temporal_too_few_entries(self, client: TestClient):
        resp = client.post("/temporal", json={
            "entries": [
                {"timestamp": "2024-01-01T10:00:00", "mes_score": 45.0},
            ]
        })
        assert resp.status_code == 422  # Min 2 entries

    def test_temporal_bad_timestamp(self, client: TestClient):
        resp = client.post("/temporal", json={
            "entries": [
                {"timestamp": "not-a-date", "mes_score": 45.0},
                {"timestamp": "2024-01-02T10:00:00", "mes_score": 50.0},
            ]
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_classify_baseline(self, client: TestClient):
        resp = client.post("/classify", json={
            "user_state": {
                "current_mes": 45.0,
                "confidence": 0.8,
                "trend": "stable",
                "trend_slope": 0.0,
                "baseline_mes": 45.0,
                "n_entries": 10,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "baseline"
        assert data["state_display"] == "Balanced"
        assert len(data["primary_signals"]) > 0
        assert data["target_direction"] == "stable"

    def test_classify_overwhelm(self, client: TestClient):
        resp = client.post("/classify", json={
            "user_state": {
                "current_mes": 65.0,
                "confidence": 0.8,
                "trend": "worsening",
                "trend_slope": 2.0,
                "baseline_mes": 45.0,
                "n_entries": 10,
            },
            "temporal_features": {
                "te_volatility": 0.35,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "overwhelm"
        assert data["target_direction"] == "decrease"

    def test_classify_with_subscores(self, client: TestClient):
        resp = client.post("/classify", json={
            "user_state": {
                "current_mes": 45.0,
                "confidence": 0.8,
                "trend": "stable",
                "trend_slope": 0.0,
                "baseline_mes": 45.0,
                "n_entries": 10,
            },
            "current_subscores": {
                "prediction_coherence": 1.5,
                "compression_progress": 2.0,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["weakest_dimensions"]) == 2

    def test_classify_insufficient_data(self, client: TestClient):
        resp = client.post("/classify", json={
            "user_state": {
                "current_mes": 45.0,
                "confidence": 0.8,
                "trend": "stable",
                "trend_slope": 0.0,
                "baseline_mes": 45.0,
                "n_entries": 1,
            },
        })
        assert resp.status_code == 200
        assert resp.json()["state"] == "insufficient_data"


# ---------------------------------------------------------------------------
# POST /insight
# ---------------------------------------------------------------------------


class TestInsight:
    def test_insight_returns_all_sections(self, client: TestClient):
        resp = client.post("/insight", json={
            "user_state": {
                "current_mes": 65.0,
                "confidence": 0.8,
                "trend": "worsening",
                "trend_slope": 2.0,
                "baseline_mes": 45.0,
                "n_entries": 10,
            },
            "temporal_features": {
                "te_volatility": 0.35,
            },
            "journal_text": "Everything is falling apart and I can't focus on anything.",
            "timestamp": "2026-04-01T10:00:00Z",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "classification" in data
        assert "prompt" in data
        assert "feedback_context" in data
        assert data["classification"]["state"] == "overwhelm"
        assert "Free Energy" in data["prompt"]["system"]
        assert "falling apart" in data["prompt"]["user"]
        assert data["feedback_context"]["target_direction"] == "decrease"

    def test_insight_with_previous(self, client: TestClient):
        resp = client.post("/insight", json={
            "user_state": {
                "current_mes": 45.0,
                "confidence": 0.8,
                "trend": "stable",
                "trend_slope": 0.0,
                "baseline_mes": 45.0,
                "n_entries": 10,
            },
            "journal_text": "Today was calm and organized.",
            "previous_insight": {
                "text": "Last time we noticed a pattern.",
                "effectiveness": "effective",
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "noticed a pattern" in data["prompt"]["user"]

    def test_insight_text_too_short(self, client: TestClient):
        resp = client.post("/insight", json={
            "user_state": {
                "current_mes": 45.0,
                "confidence": 0.8,
                "trend": "stable",
                "trend_slope": 0.0,
                "baseline_mes": 45.0,
                "n_entries": 10,
            },
            "journal_text": "Hi",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /insight/feedback
# ---------------------------------------------------------------------------


class TestInsightFeedback:
    def test_feedback_effective(self, client: TestClient):
        resp = client.post("/insight/feedback", json={
            "feedback_context": {
                "timestamp": "2026-04-01T10:00:00Z",
                "state": "overwhelm",
                "mes_at_insight": 62.5,
                "subscores_at_insight": {
                    "prediction_coherence": 2.8,
                    "compression_progress": 1.9,
                },
                "target_direction": "decrease",
            },
            "new_mes": 57.0,
            "new_subscores": {
                "prediction_coherence": 3.2,
                "compression_progress": 2.5,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effective"] is True
        assert data["moved_correctly"] is True
        assert data["mes_delta"] == -5.5
        assert "prediction_coherence" in data["subscore_deltas"]

    def test_feedback_ineffective(self, client: TestClient):
        resp = client.post("/insight/feedback", json={
            "feedback_context": {
                "timestamp": "2026-04-01T10:00:00Z",
                "state": "overwhelm",
                "mes_at_insight": 62.5,
                "subscores_at_insight": {},
                "target_direction": "decrease",
            },
            "new_mes": 68.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effective"] is False
        assert data["moved_correctly"] is False

    def test_feedback_stable_target(self, client: TestClient):
        resp = client.post("/insight/feedback", json={
            "feedback_context": {
                "timestamp": "2026-04-01T10:00:00Z",
                "state": "baseline",
                "mes_at_insight": 45.0,
                "subscores_at_insight": {},
                "target_direction": "stable",
            },
            "new_mes": 46.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effective"] is True
