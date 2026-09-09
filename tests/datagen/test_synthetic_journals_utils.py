"""Utility tests for synthetic journal generation."""

from __future__ import annotations

import json

import pytest

from mental_entropy.datagen.synthetic_journals import (
    Context,
    EntrySpec,
    GeneratorConfig,
    Persona,
    _parse_args,
    build_user_prompt,
    count_em_dashes,
    contains_disallowed,
    count_paragraphs,
    count_sentences,
    extract_journal_text,
    generate_synthetic_journals,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two? Three!", 3),
        ("Single sentence only.", 1),
    ],
)
def test_count_sentences_basic(text: str, expected: int) -> None:
    assert count_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", 3),
        ("Only one paragraph here.", 1),
    ],
)
def test_count_paragraphs_basic(text: str, expected: int) -> None:
    assert count_paragraphs(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"journal_text": "Hello world."}', "Hello world."),
        ("```json\n{\"journal_text\": \"Hello again.\"}\n```", "Hello again."),
    ],
)
def test_extract_journal_text_from_json(raw: str, expected: str) -> None:
    assert extract_journal_text(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I mentioned MES once.", True),
        ("I had a normal day.", False),
    ],
)
def test_contains_disallowed_terms(text: str, expected: bool) -> None:
    assert contains_disallowed(text) is expected


def test_parse_args_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch-size must be positive"):
        _parse_args(["--output", "out.jsonl", "--batch-size", "0"])


def test_count_em_dashes() -> None:
    assert count_em_dashes("No em dash here.") == 0
    assert count_em_dashes("One — em dash.") == 1


def test_generate_retries_when_paragraph_count_mismatch(monkeypatch, tmp_path) -> None:
    spec = EntrySpec(
        entry_id=1,
        batch_number=1,
        batch_size=1,
        target_sentence_count=2,
        target_paragraph_count=2,
        allow_bullets=False,
        entropy_bucket="low",
        entropy_driver="routine day, single setting, linear timeline, stable mood",
        word_count_range=(2, 40),
        imperfection_count=0,
        opening_style="Start with a concrete physical detail in the present moment.",
        persona=Persona(
            age_range="25-34",
            gender=None,
            occupation_or_role="engineer",
            personality_traits=("reflective", "practical"),
            baseline_writing_style="reflective",
            life_context="lives alone, steady routine",
            cognitive_state="calm",
        ),
        context=Context(
            primary_bucket="normal day reflection",
            secondary_bucket=None,
            setting="late night at home",
            timeframe="today",
        ),
    )

    class _FakeClient:
        calls = 0

        def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
            del base_url, api_key, model

        def generate(
            self,
            messages: list[dict[str, str]],
            temperature: float,
            max_tokens: int,
            seed: int | None,
        ) -> str:
            del messages, temperature, max_tokens, seed
            type(self).calls += 1
            if type(self).calls == 1:
                return '{"journal_text": "First sentence. Second sentence."}'
            return '{"journal_text": "First sentence.\\n\\nSecond sentence."}'

    monkeypatch.setattr("mental_entropy.datagen.synthetic_journals.OpenAICompatClient", _FakeClient)
    monkeypatch.setattr("mental_entropy.datagen.synthetic_journals.build_entry_specs", lambda config: iter([spec]))

    output_path = tmp_path / "generated.jsonl"
    config = GeneratorConfig(
        total_entries=1,
        batch_size=1,
        global_seed=123,
        output_path=str(output_path),
        provider="openai_compat",
        model="test-model",
        base_url="https://example.com/v1/chat/completions",
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        max_retries=1,
        sentence_tolerance=1,
        words_per_sentence_range=(1, 20),
        entropy_weights={"low": 1.0, "mid": 0.0, "high": 0.0},
        log_every=1,
        send_seed=False,
    )

    generate_synthetic_journals(config)

    assert _FakeClient.calls == 2
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["metadata"]["paragraph_count"] == 2


def test_generate_continues_when_entry_fails(monkeypatch, tmp_path) -> None:
    failing_spec = EntrySpec(
        entry_id=1,
        batch_number=1,
        batch_size=2,
        target_sentence_count=2,
        target_paragraph_count=1,
        allow_bullets=False,
        entropy_bucket="low",
        entropy_driver="routine day, single setting, linear timeline, stable mood",
        word_count_range=(2, 20),
        imperfection_count=0,
        opening_style="Start with a practical concern from the day.",
        persona=Persona(
            age_range="25-34",
            gender=None,
            occupation_or_role="engineer",
            personality_traits=("reflective", "practical"),
            baseline_writing_style="reflective",
            life_context="lives alone, steady routine",
            cognitive_state="calm",
        ),
        context=Context(
            primary_bucket="normal day reflection",
            secondary_bucket=None,
            setting="late night at home",
            timeframe="today",
        ),
    )
    good_spec = EntrySpec(
        entry_id=2,
        batch_number=1,
        batch_size=2,
        target_sentence_count=2,
        target_paragraph_count=1,
        allow_bullets=False,
        entropy_bucket="low",
        entropy_driver="routine day, single setting, linear timeline, stable mood",
        word_count_range=(2, 20),
        imperfection_count=0,
        opening_style="Start with a practical concern from the day.",
        persona=failing_spec.persona,
        context=failing_spec.context,
    )

    class _FakeClient:
        calls = 0

        def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
            del base_url, api_key, model

        def generate(
            self,
            messages: list[dict[str, str]],
            temperature: float,
            max_tokens: int,
            seed: int | None,
        ) -> str:
            del messages, temperature, max_tokens, seed
            type(self).calls += 1
            if type(self).calls == 1:
                return '{"journal_text": ""}'
            return '{"journal_text": "First sentence. Second sentence."}'

    monkeypatch.setattr("mental_entropy.datagen.synthetic_journals.OpenAICompatClient", _FakeClient)
    monkeypatch.setattr(
        "mental_entropy.datagen.synthetic_journals.build_entry_specs",
        lambda config: iter([failing_spec, good_spec]),
    )

    output_path = tmp_path / "generated_partial.jsonl"
    config = GeneratorConfig(
        total_entries=2,
        batch_size=2,
        global_seed=123,
        output_path=str(output_path),
        provider="openai_compat",
        model="test-model",
        base_url="https://example.com/v1/chat/completions",
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        max_retries=0,
        sentence_tolerance=1,
        words_per_sentence_range=(1, 20),
        entropy_weights={"low": 1.0, "mid": 0.0, "high": 0.0},
        log_every=1,
        send_seed=True,
        continue_on_failure=True,
    )

    generate_synthetic_journals(config)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == 2


def test_generate_append_skips_existing_ids(monkeypatch, tmp_path) -> None:
    spec1 = EntrySpec(
        entry_id=1,
        batch_number=1,
        batch_size=2,
        target_sentence_count=2,
        target_paragraph_count=1,
        allow_bullets=False,
        entropy_bucket="low",
        entropy_driver="routine day, single setting, linear timeline, stable mood",
        word_count_range=(2, 20),
        imperfection_count=0,
        opening_style="Start with an action the writer just took.",
        persona=Persona(
            age_range="25-34",
            gender=None,
            occupation_or_role="engineer",
            personality_traits=("reflective", "practical"),
            baseline_writing_style="reflective",
            life_context="lives alone, steady routine",
            cognitive_state="calm",
        ),
        context=Context(
            primary_bucket="normal day reflection",
            secondary_bucket=None,
            setting="late night at home",
            timeframe="today",
        ),
    )
    spec2 = EntrySpec(
        entry_id=2,
        batch_number=1,
        batch_size=2,
        target_sentence_count=2,
        target_paragraph_count=1,
        allow_bullets=False,
        entropy_bucket="low",
        entropy_driver="routine day, single setting, linear timeline, stable mood",
        word_count_range=(2, 20),
        imperfection_count=0,
        opening_style="Start with an action the writer just took.",
        persona=spec1.persona,
        context=spec1.context,
    )

    class _FakeClient:
        calls = 0

        def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
            del base_url, api_key, model

        def generate(
            self,
            messages: list[dict[str, str]],
            temperature: float,
            max_tokens: int,
            seed: int | None,
        ) -> str:
            del messages, temperature, max_tokens, seed
            type(self).calls += 1
            return '{"journal_text": "First sentence. Second sentence."}'

    monkeypatch.setattr("mental_entropy.datagen.synthetic_journals.OpenAICompatClient", _FakeClient)
    monkeypatch.setattr("mental_entropy.datagen.synthetic_journals.build_entry_specs", lambda config: iter([spec1, spec2]))

    output_path = tmp_path / "generated_append.jsonl"
    output_path.write_text('{"id": 1, "journal_text": "already here"}\n', encoding="utf-8")
    config = GeneratorConfig(
        total_entries=2,
        batch_size=2,
        global_seed=123,
        output_path=str(output_path),
        provider="openai_compat",
        model="test-model",
        base_url="https://example.com/v1/chat/completions",
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        max_retries=0,
        sentence_tolerance=1,
        words_per_sentence_range=(1, 20),
        entropy_weights={"low": 1.0, "mid": 0.0, "high": 0.0},
        log_every=1,
        send_seed=False,
        append_output=True,
    )

    generate_synthetic_journals(config)

    assert _FakeClient.calls == 1
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == [1, 2]


def test_build_user_prompt_uses_opening_style_and_no_entropy_instruction() -> None:
    spec = EntrySpec(
        entry_id=1,
        batch_number=1,
        batch_size=1,
        target_sentence_count=10,
        target_paragraph_count=1,
        allow_bullets=False,
        entropy_bucket="low",
        entropy_driver="routine day, single setting, linear timeline, stable mood",
        word_count_range=(120, 240),
        imperfection_count=0,
        opening_style="Start with an action the writer just took.",
        persona=Persona(
            age_range="25-34",
            gender=None,
            occupation_or_role="engineer",
            personality_traits=("reflective", "practical"),
            baseline_writing_style="reflective",
            life_context="lives alone, steady routine",
            cognitive_state="calm",
        ),
        context=Context(
            primary_bucket="normal day reflection",
            secondary_bucket=None,
            setting="late night at home",
            timeframe="today",
        ),
    )

    prompt = build_user_prompt(spec, sentence_tolerance=1)
    assert "Opening style target: Start with an action the writer just took." in prompt
    assert "Prompt family: general_reflection" in prompt
    assert "Avoid opener template: Do not start with \"I'm sitting...\"." in prompt
    assert "Avoid em dashes" in prompt
    assert "Entropy driver" not in prompt
