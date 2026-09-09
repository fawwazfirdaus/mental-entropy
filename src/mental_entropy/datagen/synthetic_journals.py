"""Synthetic journal dataset generation.

This module builds realistic human journal entries using a configurable
prompt/spec and an OpenAI-compatible chat completion endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Persona:
    age_range: str
    gender: str | None
    occupation_or_role: str
    personality_traits: tuple[str, ...]
    baseline_writing_style: str
    life_context: str
    cognitive_state: str


@dataclass(frozen=True, slots=True)
class Context:
    primary_bucket: str
    secondary_bucket: str | None
    setting: str
    timeframe: str


@dataclass(frozen=True, slots=True)
class EntrySpec:
    entry_id: int
    batch_number: int
    batch_size: int
    target_sentence_count: int
    target_paragraph_count: int
    allow_bullets: bool
    entropy_bucket: str
    entropy_driver: str
    word_count_range: tuple[int, int]
    imperfection_count: int
    opening_style: str
    persona: Persona
    context: Context
    prompt_family: str = "general_reflection"
    opener_avoid: str = "Do not start with \"I'm sitting...\"."


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    total_entries: int
    batch_size: int
    global_seed: int
    output_path: str
    provider: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float
    max_tokens: int
    max_retries: int
    sentence_tolerance: int
    words_per_sentence_range: tuple[int, int]
    entropy_weights: dict[str, float]
    log_every: int
    send_seed: bool
    max_em_dashes: int = 1
    append_output: bool = False
    continue_on_failure: bool = True


# ---------------------------------------------------------------------------
# Defaults and pools
# ---------------------------------------------------------------------------

SENTENCE_BINS: list[tuple[int, int, float]] = [
    (8, 9, 0.10),
    (10, 13, 0.55),
    (14, 16, 0.30),
    (17, 18, 0.05),
]

PARAGRAPH_WEIGHTS: dict[int, float] = {
    1: 0.50,
    2: 0.35,
    3: 0.12,
    4: 0.03,
}

GENDER_OPTIONS: list[tuple[str | None, float]] = [
    (None, 0.20),
    ("female", 0.40),
    ("male", 0.35),
    ("nonbinary", 0.05),
]

AGE_RANGES: tuple[str, ...] = (
    "18-24",
    "25-34",
    "35-44",
    "45-60",
)

OCCUPATIONS: tuple[str, ...] = (
    "student",
    "nurse",
    "engineer",
    "teacher",
    "retail associate",
    "barista",
    "parent",
    "caregiver",
    "designer",
    "software developer",
    "project manager",
    "administrative assistant",
    "accountant",
    "writer",
    "artist",
    "freelancer",
    "founder",
    "research assistant",
    "counselor",
    "therapist",
    "sales associate",
    "chef",
    "mechanic",
    "warehouse worker",
    "grad student",
    "unemployed",
    "job seeker",
    "shift supervisor",
    "customer support",
)

PERSONALITY_TRAITS: tuple[str, ...] = (
    "reflective",
    "observant",
    "practical",
    "warm",
    "restless",
    "patient",
    "private",
    "curious",
    "anxious",
    "optimistic",
    "skeptical",
    "organized",
    "scatterbrained",
    "sensitive",
    "dry-humored",
    "direct",
    "gentle",
    "intense",
    "self-critical",
    "thoughtful",
    "hopeful",
    "overthinking",
    "grounded",
    "spontaneous",
)

WRITING_STYLES: tuple[str, ...] = (
    "reflective",
    "blunt",
    "rambling",
    "minimalist",
    "analytical",
    "messy-notes",
    "stream-of-consciousness",
    "dry",
    "expressive",
)

LIFE_CONTEXTS: tuple[str, ...] = (
    "lives alone, steady routine",
    "shares an apartment, juggling schedules",
    "new relationship, unsure where it is going",
    "long-term relationship, working on communication",
    "lives with family, limited privacy",
    "new city, still building a social circle",
    "recently moved, feeling ungrounded",
    "financially tight, tracking expenses closely",
    "caring for a parent, stretched thin",
    "coparenting, coordinating calendars",
    "remote work, blurred boundaries",
    "in school, balancing work and classes",
    "freelancing, income inconsistent",
)

COGNITIVE_STATES: tuple[str, ...] = (
    "calm",
    "focused",
    "tired",
    "exhausted",
    "scattered",
    "anxious",
    "rushing",
    "overwhelmed",
    "numb",
    "restless",
    "hyperfocused",
    "foggy",
)

CONTEXT_BUCKETS: tuple[str, ...] = (
    "normal day reflection",
    "work/school pressure",
    "relationship tension or repair",
    "family / grief / health",
    "identity / meaning / uncertainty",
    "logistics overload",
    "social friction",
    "positive but busy",
)

SETTINGS: tuple[str, ...] = (
    "late night at home",
    "early morning kitchen",
    "bus ride home",
    "quiet lunch break",
    "bedroom desk",
    "parking lot before work",
    "living room couch",
    "coffee shop corner",
    "after dinner at the table",
    "walk around the block",
)

SETTING_WEIGHTS: dict[str, float] = {
    "late night at home": 1.0,
    "early morning kitchen": 1.0,
    "bus ride home": 0.9,
    "quiet lunch break": 1.0,
    "bedroom desk": 1.0,
    "parking lot before work": 0.9,
    "living room couch": 1.0,
    "coffee shop corner": 0.45,
    "after dinner at the table": 1.0,
    "walk around the block": 0.9,
}

TIMEFRAMES: tuple[str, ...] = (
    "today",
    "tonight",
    "this morning",
    "this afternoon",
    "late last night",
    "just now",
)

ENTROPY_DRIVERS: dict[str, tuple[str, ...]] = {
    "low": (
        "routine day, single setting, linear timeline, stable mood",
        "one main concern, steady pace, minimal interruptions",
        "focused on one thread, calm tone, few topic shifts",
    ),
    "mid": (
        "two settings, mild interruptions, some topic drift",
        "routine with one curveball, occasional mental jumps",
        "mostly linear with brief digressions",
    ),
    "high": (
        "interruptions, topic shifts, conflicting obligations, non-linear jumps",
        "fragmented day, scattered attention, abrupt transitions",
        "multiple threads competing, emotional whiplash",
    ),
}

ENTROPY_COGNITIVE_MAP: dict[str, tuple[str, ...]] = {
    "low": ("calm", "focused", "tired"),
    "mid": ("tired", "foggy", "hyperfocused", "restless"),
    "high": ("scattered", "anxious", "rushing", "overwhelmed", "restless"),
}

# Banned terms that should never appear in journal text
BANNED_TERMS: tuple[str, ...] = (
    "mental entropy",
    "mes",
    "entropy",
    "coherence score",
    "subscore",
    "subscores",
    "classification",
    "scoring",
    "language model",
)

OPENING_STYLES: tuple[str, ...] = (
    "Start with a concrete physical detail in the present moment.",
    "Start with an action the writer just took.",
    "Start with a plain emotional state without dramatic wording.",
    "Start with a practical concern from the day.",
    "Start with a short memory from earlier today, then move to now.",
    "Start with a brief observation about the setting sounds or light.",
)

PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "daily_log": (
        "Prioritize concrete events and practical details over introspection.",
        "Use a plain, matter-of-fact tone.",
    ),
    "emotional_processing": (
        "Center on one emotionally meaningful moment from the day.",
        "Keep emotional language direct and grounded.",
    ),
    "logistics_overload": (
        "Emphasize scheduling, responsibilities, and unfinished tasks.",
        "Let stress appear through details, not dramatic wording.",
    ),
    "interpersonal_reflection": (
        "Focus on one conversation, misunderstanding, or social interaction.",
        "Include uncertainty without over-analyzing every sentence.",
    ),
    "mixed_reflection": (
        "Blend concrete day events with short reflective thoughts.",
        "Keep pacing natural with occasional digression.",
    ),
}

OPENER_AVOID_PATTERNS: tuple[str, ...] = (
    "I'm sitting",
    "Sitting in",
    "It's [time] and I'm",
    "The hum of",
    "Just closed my laptop",
)

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are generating realistic human journal entries for ML training.

ABSOLUTE RULES (DO NOT VIOLATE):
- Do NOT mention "mental entropy", "MES", "entropy", "coherence score", or "subscores".
- Do NOT include any scores, ratings, classifications, or analysis.
- Output must be a single JSON object and nothing else.
- The JSON object must contain a "journal_text" field only.
- No markdown, no headings, no commentary.
"""


def load_dotenv(path: str) -> None:
    """Load environment variables from a .env file without overriding existing env."""
    if not path or not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _peek_env_file(argv: list[str]) -> str:
    """Peek env file path from argv without full parsing."""
    for idx, arg in enumerate(argv):
        if arg == "--env-file" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--env-file="):
            return arg.split("=", 1)[1]
    return ".env"


def build_user_prompt(spec: EntrySpec, sentence_tolerance: int) -> str:
    """Build a per-entry user prompt."""
    persona = spec.persona
    context = spec.context
    gender_value = "null" if persona.gender is None else json.dumps(persona.gender)
    secondary_value = "null" if context.secondary_bucket is None else json.dumps(context.secondary_bucket)
    traits_value = json.dumps(list(persona.personality_traits))
    age_value = json.dumps(persona.age_range)
    occupation_value = json.dumps(persona.occupation_or_role)
    style_value = json.dumps(persona.baseline_writing_style)
    life_context_value = json.dumps(persona.life_context)
    cognitive_value = json.dumps(persona.cognitive_state)
    primary_value = json.dumps(context.primary_bucket)
    setting_value = json.dumps(context.setting)
    timeframe_value = json.dumps(context.timeframe)
    family_guidance = PROMPT_FAMILIES.get(spec.prompt_family, PROMPT_FAMILIES["mixed_reflection"])
    family_line_a, family_line_b = family_guidance

    return f"""\
Generate ONE private, candid journal entry in English.

Persona (use exactly as written; do not invent new fields):
{{
  "age_range": {age_value},
  "gender": {gender_value},
  "occupation_or_role": {occupation_value},
  "personality_traits": {traits_value},
  "baseline_writing_style": {style_value},
  "life_context": {life_context_value},
  "cognitive_state": {cognitive_value}
}}

Context (use exactly as written; do not label it inside the journal):
{{
  "primary_bucket": {primary_value},
  "secondary_bucket": {secondary_value},
  "setting": {setting_value},
  "timeframe": {timeframe_value}
}}

Writing constraints:
- Sentence count target: {spec.target_sentence_count} (+/-{sentence_tolerance}).
- Paragraph count target: {spec.target_paragraph_count}. Use blank lines if multiple paragraphs.
- Word count target: {spec.word_count_range[0]}-{spec.word_count_range[1]}.
- Bullets: {"allowed" if spec.allow_bullets else "avoid entirely"}.
- Use 0-2 emojis max.
- Prefer natural phrasing over polished literary style.
- Include concrete details only when they fit naturally.
- Avoid specific personal identifiers (full names, addresses, phone numbers, emails).
- Avoid poetic over-stylization or artificial polish.
- Avoid em dashes ("—"). Prefer periods, commas, or parentheses.
- Let coherence/disorganization emerge naturally from the persona and context. Do not intentionally inject, force, or target any entropy pattern.
- Prompt family: {spec.prompt_family}
- Family guidance: {family_line_a}
- Family guidance: {family_line_b}
- Opening style target: {spec.opening_style}
- Avoid opener template: {spec.opener_avoid}
- Keep references to coffee shops/cafes only when they fit the provided setting.

Output format:
- Return a single JSON object with exactly one key: "journal_text".
- Example: {{"journal_text": "..."}} (no extra keys).
"""


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------


class OpenAICompatClient:
    """Minimal OpenAI-compatible Chat Completions client using urllib."""

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def generate(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None) -> str:
        """Generate a completion and return raw text."""
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request = Request(self._base_url, data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"LLM request failed: HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response was not valid JSON.") from exc

        choices = parsed.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response had no choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM response content missing or invalid.")
        return content.strip()


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def _weighted_choice(rng: random.Random, items: list[tuple[str | int | None, float]]) -> str | int | None:
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    pick = rng.random() * total
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if pick <= cumulative:
            return value
    return items[-1][0]


def _sample_sentence_count(rng: random.Random) -> int:
    weighted_bins = [(f"{lo}-{hi}", weight) for lo, hi, weight in SENTENCE_BINS]
    pick = _weighted_choice(rng, weighted_bins)
    if pick is None:
        raise ValueError("Sentence bin selection failed.")
    lo_str, hi_str = str(pick).split("-")
    lo, hi = int(lo_str), int(hi_str)
    return rng.randint(lo, hi)


def _sample_paragraph_count(rng: random.Random) -> int:
    weighted = [(count, weight) for count, weight in PARAGRAPH_WEIGHTS.items()]
    pick = _weighted_choice(rng, weighted)
    if pick is None:
        raise ValueError("Paragraph selection failed.")
    return int(pick)


def _sample_entropy_bucket(rng: random.Random, weights: dict[str, float]) -> str:
    weighted = [(bucket, weight) for bucket, weight in weights.items()]
    pick = _weighted_choice(rng, weighted)
    if pick is None:
        raise ValueError("Entropy bucket selection failed.")
    return str(pick)


def _sample_persona(rng: random.Random, entropy_bucket: str) -> Persona:
    gender = _weighted_choice(rng, GENDER_OPTIONS)
    trait_count = rng.randint(2, 4)
    traits = rng.sample(PERSONALITY_TRAITS, trait_count)
    cognitive_pool = ENTROPY_COGNITIVE_MAP.get(entropy_bucket, COGNITIVE_STATES)
    cognitive_state = rng.choice(cognitive_pool)

    return Persona(
        age_range=rng.choice(AGE_RANGES),
        gender=gender if isinstance(gender, str) or gender is None else None,
        occupation_or_role=rng.choice(OCCUPATIONS),
        personality_traits=tuple(traits),
        baseline_writing_style=rng.choice(WRITING_STYLES),
        life_context=rng.choice(LIFE_CONTEXTS),
        cognitive_state=cognitive_state,
    )


def _sample_context(rng: random.Random) -> Context:
    primary = rng.choice(CONTEXT_BUCKETS)
    secondary = None
    if rng.random() < 0.35:
        options = [bucket for bucket in CONTEXT_BUCKETS if bucket != primary]
        secondary = rng.choice(options)
    weighted_settings = [(setting, SETTING_WEIGHTS.get(setting, 1.0)) for setting in SETTINGS]
    setting_pick = _weighted_choice(rng, weighted_settings)
    if not isinstance(setting_pick, str):
        raise ValueError("Setting selection failed.")
    return Context(
        primary_bucket=primary,
        secondary_bucket=secondary,
        setting=setting_pick,
        timeframe=rng.choice(TIMEFRAMES),
    )


def _batch_seed(global_seed: int, batch_number: int) -> str:
    return f"{global_seed}::batch={batch_number}"


def build_entry_specs(config: GeneratorConfig) -> Iterator[EntrySpec]:
    """Generate entry specs with deterministic per-batch randomness."""
    total_entries = config.total_entries
    batch_size = config.batch_size

    for batch_number in range(1, (total_entries - 1) // batch_size + 2):
        batch_seed = _batch_seed(config.global_seed, batch_number)
        rng = random.Random(batch_seed)
        seen_signatures: set[tuple[str, str, str, str]] = set()

        start_id = (batch_number - 1) * batch_size + 1
        end_id = min(batch_number * batch_size, total_entries)
        for entry_id in range(start_id, end_id + 1):
            entropy_bucket = _sample_entropy_bucket(rng, config.entropy_weights)
            entropy_driver = rng.choice(ENTROPY_DRIVERS[entropy_bucket])
            target_sentence_count = _sample_sentence_count(rng)
            target_paragraph_count = _sample_paragraph_count(rng)
            allow_bullets = rng.random() < 0.05
            imperfection_count = rng.randint(0, 4)
            opening_style = rng.choice(OPENING_STYLES)
            prompt_family = rng.choice(tuple(PROMPT_FAMILIES.keys()))
            opener_avoid = rng.choice(OPENER_AVOID_PATTERNS)
            min_wps, max_wps = config.words_per_sentence_range
            word_count_range = (
                target_sentence_count * min_wps,
                target_sentence_count * max_wps,
            )

            # Avoid near-identical persona within batch
            for _ in range(12):
                persona = _sample_persona(rng, entropy_bucket)
                signature = (
                    persona.age_range,
                    persona.occupation_or_role,
                    persona.baseline_writing_style,
                    persona.cognitive_state,
                )
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    break
            else:
                persona = _sample_persona(rng, entropy_bucket)

            context = _sample_context(rng)

            yield EntrySpec(
                entry_id=entry_id,
                batch_number=batch_number,
                batch_size=batch_size,
                target_sentence_count=target_sentence_count,
                target_paragraph_count=target_paragraph_count,
                allow_bullets=allow_bullets,
                entropy_bucket=entropy_bucket,
                entropy_driver=entropy_driver,
                word_count_range=word_count_range,
                imperfection_count=imperfection_count,
                opening_style=opening_style,
                persona=persona,
                context=context,
                prompt_family=prompt_family,
                opener_avoid=opener_avoid,
            )


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+\w+(?:\s+\w+){0,3}\s+(?:St|Street|Ave|Avenue|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct)\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s+", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: str) -> int:
    """Approximate sentence count using punctuation heuristics."""
    cleaned = text.strip()
    if not cleaned:
        return 0
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return len([part for part in parts if part.strip()])


def count_paragraphs(text: str) -> int:
    """Count paragraph blocks separated by blank lines."""
    cleaned = text.strip()
    if not cleaned:
        return 0
    blocks = re.split(r"\n\s*\n", cleaned)
    return len([block for block in blocks if block.strip()])


def contains_bullets(text: str) -> bool:
    return bool(_BULLET_RE.search(text))


def contains_disallowed(text: str) -> bool:
    lower = text.lower()
    for term in BANNED_TERMS:
        if " " in term:
            if term in lower:
                return True
        else:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, lower):
                return True
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _ADDRESS_RE.search(text):
        return True
    return False


def word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def count_em_dashes(text: str) -> int:
    return text.count("—")


def extract_journal_text(raw: str) -> str:
    """Extract journal_text from a JSON object or raw text."""
    stripped = raw.strip()
    if not stripped:
        return ""

    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return ""
        if isinstance(data, dict):
            journal_text = data.get("journal_text")
            if isinstance(journal_text, str):
                return journal_text.strip()

    # Attempt to extract JSON object from mixed text
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                return ""
            if isinstance(data, dict):
                journal_text = data.get("journal_text")
                if isinstance(journal_text, str):
                    return journal_text.strip()

    return ""


def _load_existing_entry_ids(path: str) -> set[int]:
    """Load existing entry IDs from a JSONL file."""
    if not os.path.exists(path):
        return set()

    ids: set[int] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in existing output at line {line_number}.") from exc
            if not isinstance(payload, dict):
                continue
            entry_id = payload.get("id")
            if isinstance(entry_id, int):
                ids.add(entry_id)
    return ids


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_synthetic_journals(config: GeneratorConfig) -> None:
    """Generate synthetic journals and write to a JSONL file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logger = logging.getLogger("synthetic_journals")

    if config.provider != "openai_compat":
        raise ValueError(f"Unsupported provider: {config.provider}")

    if not config.api_key:
        logger.warning("No API key provided. Requests may fail if the endpoint requires auth.")

    client = OpenAICompatClient(config.base_url, config.api_key, config.model)

    total = config.total_entries
    existing_ids: set[int] = set()
    if config.append_output:
        existing_ids = _load_existing_entry_ids(config.output_path)
    written = len(existing_ids)
    failures = 0
    started = time.time()

    file_mode = "a" if config.append_output else "w"
    with open(config.output_path, file_mode, encoding="utf-8") as handle:
        for spec in build_entry_specs(config):
            if spec.entry_id in existing_ids:
                continue
            seed_material = f"{config.global_seed}:{spec.entry_id}".encode("utf-8")
            entry_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
            attempt = 0
            last_error = None

            while attempt <= config.max_retries:
                attempt += 1
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(spec, config.sentence_tolerance)},
                ]

                try:
                    raw = client.generate(
                        messages=messages,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        seed=entry_seed if config.send_seed else None,
                    )
                except RuntimeError as exc:
                    last_error = str(exc)
                    continue

                journal_text = extract_journal_text(raw)
                if not journal_text:
                    last_error = "Empty journal text."
                    continue

                if contains_disallowed(journal_text):
                    last_error = "Disallowed content detected."
                    continue

                sentence_count = count_sentences(journal_text)
                if abs(sentence_count - spec.target_sentence_count) > config.sentence_tolerance:
                    last_error = (
                        f"Sentence count {sentence_count} outside tolerance "
                        f"for target {spec.target_sentence_count}."
                    )
                    continue

                wc = word_count(journal_text)
                if not (spec.word_count_range[0] <= wc <= spec.word_count_range[1]):
                    last_error = f"Word count {wc} outside target range."
                    continue

                em_dash_count = count_em_dashes(journal_text)
                if em_dash_count > config.max_em_dashes:
                    last_error = (
                        f"Em dash count {em_dash_count} exceeded max "
                        f"{config.max_em_dashes}."
                    )
                    continue

                paragraph_count = count_paragraphs(journal_text)
                if paragraph_count != spec.target_paragraph_count:
                    last_error = (
                        f"Paragraph count {paragraph_count} did not match "
                        f"target {spec.target_paragraph_count}."
                    )
                    continue

                bullet_flag = contains_bullets(journal_text)
                if bullet_flag and not spec.allow_bullets:
                    last_error = "Bullets detected when not allowed."
                    continue

                persona = spec.persona
                context = spec.context

                entry = {
                    "id": spec.entry_id,
                    "persona": {
                        "age_range": persona.age_range,
                        "gender": persona.gender,
                        "occupation_or_role": persona.occupation_or_role,
                        "personality_traits": list(persona.personality_traits),
                        "baseline_writing_style": persona.baseline_writing_style,
                        "life_context": persona.life_context,
                        "cognitive_state": persona.cognitive_state,
                    },
                    "context": {
                        "primary_bucket": context.primary_bucket,
                        "secondary_bucket": context.secondary_bucket,
                        "setting": context.setting,
                        "timeframe": context.timeframe,
                    },
                    "journal_text": journal_text,
                    "metadata": {
                        "sentence_count": int(sentence_count),
                        "paragraph_count": int(paragraph_count),
                        "contains_bullets": bool(bullet_flag),
                        "batch_number": spec.batch_number,
                        "batch_size": spec.batch_size,
                    },
                }

                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

                if written % config.log_every == 0 or written == total:
                    elapsed = time.time() - started
                    logger.info(f"Wrote {written}/{total} entries ({elapsed:.1f}s)")
                break

            else:
                if config.continue_on_failure:
                    failures += 1
                    logger.warning(f"Skipping entry {spec.entry_id} after retries: {last_error}")
                    continue
                raise RuntimeError(f"Failed to generate entry {spec.entry_id}: {last_error}")

    if failures:
        logger.warning(f"Completed with {failures} skipped entries out of {total}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> GeneratorConfig:
    env_file = _peek_env_file(argv)
    load_dotenv(env_file)

    parser = argparse.ArgumentParser(description="Generate synthetic journal JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--env-file", default=env_file, help="Path to .env file to load.")
    parser.add_argument("--total-entries", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--provider", default="openai_compat", choices=["openai_compat"])
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini")),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "OPENAI_BASE_URL",
            os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"),
        ),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sentence-tolerance", type=int, default=1)
    parser.add_argument("--min-words-per-sentence", type=int, default=14)
    parser.add_argument("--max-words-per-sentence", type=int, default=20)
    parser.add_argument("--entropy-low", type=float, default=0.35)
    parser.add_argument("--entropy-mid", type=float, default=0.30)
    parser.add_argument("--entropy-high", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--send-seed", action="store_true", help="Send seed to the LLM if supported.")
    parser.add_argument("--max-em-dashes", type=int, default=1)
    parser.add_argument("--append", action="store_true", help="Append to output and skip existing entry IDs.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort run if an entry fails after retries.")

    args = parser.parse_args(argv)

    entropy_weights = {
        "low": args.entropy_low,
        "mid": args.entropy_mid,
        "high": args.entropy_high,
    }
    total_weight = sum(entropy_weights.values())
    if total_weight <= 0:
        raise ValueError("Entropy weights must sum to a positive number.")

    if args.min_words_per_sentence <= 0 or args.max_words_per_sentence <= 0:
        raise ValueError("Words-per-sentence bounds must be positive.")
    if args.min_words_per_sentence > args.max_words_per_sentence:
        raise ValueError("min-words-per-sentence cannot exceed max-words-per-sentence.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.sentence_tolerance < 0:
        raise ValueError("sentence-tolerance cannot be negative.")
    if args.max_em_dashes < 0:
        raise ValueError("max-em-dashes cannot be negative.")

    api_key = os.environ.get(args.api_key_env)

    return GeneratorConfig(
        total_entries=args.total_entries,
        batch_size=args.batch_size,
        global_seed=args.seed,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        sentence_tolerance=args.sentence_tolerance,
        words_per_sentence_range=(args.min_words_per_sentence, args.max_words_per_sentence),
        entropy_weights=entropy_weights,
        log_every=max(1, args.log_every),
        send_seed=bool(args.send_seed),
        max_em_dashes=args.max_em_dashes,
        append_output=bool(args.append),
        continue_on_failure=not bool(args.fail_fast),
    )


def main(argv: list[str] | None = None) -> None:
    config = _parse_args(argv or sys.argv[1:])
    generate_synthetic_journals(config)


if __name__ == "__main__":
    main()
