"""Tests for global disorder feature extraction."""

from __future__ import annotations

import json

from mental_entropy.embedding.types import EmbeddingResult, SentenceEmbedding
from mental_entropy.features.gd import (
    GD_FEATURE_KEYS,
    gd_features,
    gd_features_from_result,
)


def make_result(texts: list[str]) -> EmbeddingResult:
    sentences = [
        SentenceEmbedding(id=i, text=text, embedding=[1.0, 0.0])
        for i, text in enumerate(texts)
    ]
    return EmbeddingResult(sentences=sentences, doc_embedding=[1.0, 0.0])


def test_gd_features_returns_exact_keys() -> None:
    features = gd_features(["This is one thought.", "It resolves cleanly."])

    assert set(features) == GD_FEATURE_KEYS
    assert len(features) == 17


def test_gd_features_json_serializable() -> None:
    features = gd_features(["Anyway, this changed.", "I don't know."])

    json.dumps(features)
    for value in features.values():
        assert isinstance(value, float)


def test_gd_features_empty_input_defaults_to_zero() -> None:
    features = gd_features([])

    assert set(features) == GD_FEATURE_KEYS
    assert all(value == 0.0 for value in features.values())


def test_gd_features_emotional_but_coherent_stays_low() -> None:
    sentences = [
        "I was scared after the argument.",
        "I realized the fear came from an old pattern.",
        "That helps me know what boundary I need next.",
    ]

    features = gd_features(sentences)

    assert features["gd_global_disorder_score"] < 0.2
    assert features["gd_topic_reset_rate"] == 0.0
    assert features["gd_corruption_rate"] == 0.0


def test_gd_features_detects_corrupted_fragmented_text() -> None:
    sentences = [
        'alt="quiz result" border=0> Ding ding.',
        "Take the weird obsession quiz by somebody.",
        "I don't even know what this is.",
        "Anyway, not full time anyhow.",
    ]

    features = gd_features(sentences)

    assert features["gd_corruption_rate"] >= 0.5
    assert features["gd_topic_reset_rate"] >= 0.25
    assert features["gd_global_disorder_score"] >= 0.35


def test_gd_features_detects_runon_loops_and_no_compression_ending() -> None:
    sentences = [
        "I think I think I think I should explain because and because and because everything keeps going and going and going and I cannot land the thought at all",
        "He said that and they did this and it was there and this was that.",
        "Whatever, I won't elaborate here.",
    ]

    features = gd_features(sentences)

    assert features["gd_runon_chain_rate"] > 0.0
    assert features["gd_loop_repetition_rate"] > 0.0
    assert features["gd_dangling_referent_rate"] > 0.0
    assert features["gd_no_compression_ending"] == 1.0
    assert features["gd_disorder_marker_count_norm"] >= 0.5


def test_gd_features_from_result_uses_sentence_text() -> None:
    result = make_result(["First thought.", "Anyway, second thought."])

    assert gd_features_from_result(result) == gd_features([
        "First thought.",
        "Anyway, second thought.",
    ])


def test_gd_features_detects_block_resets_and_abandoned_setup() -> None:
    text = [
        "Apologies for the format because I'm on mobile.",
        "For context, my cousin uses my account and my dog knows Reddit.",
        "",
        "Anyway, the real issue started last year.",
        "There was a room and they kept saying it would happen.",
        "",
        "Also I forgot to explain the family situation.",
        "I don't know."
    ]

    features = gd_features(text)

    assert features["gd_block_reset_rate"] > 0.0
    assert features["gd_abandoned_setup_count_norm"] >= 0.25
    assert features["gd_compression_failure_score"] >= 0.4


def test_gd_features_detects_entity_drift() -> None:
    sentences = [
        "Sarah called yesterday about the lease.",
        "Mark texted about a car.",
        "Jennifer said the office was closing.",
        "Daniel was angry about the bus.",
        "Anyway, I don't know what any of this means.",
    ]

    features = gd_features(sentences)

    assert features["gd_single_use_entity_rate"] >= 0.5
    assert features["gd_entity_drift_score"] >= 0.4


def test_gd_features_long_entry_density_catches_repeated_small_failures() -> None:
    sentences = [
        "Anyway, I don't know.",
        "Also this changed.",
        "Wait, that was different.",
        "He said it was there.",
        "Take the strange quiz result.",
        "I think I think I should stop.",
        "Whatever, I won't elaborate here.",
    ]

    features = gd_features(sentences)

    assert features["gd_disorder_markers_per_500w"] > 10.0
    assert features["gd_global_disorder_score"] >= 0.45
