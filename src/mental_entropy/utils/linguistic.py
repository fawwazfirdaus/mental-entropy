"""Linguistic analysis utilities for feature extraction.

This module provides common linguistic analysis functions, regex patterns, and word/phrase
lists used across multiple feature modules (NE, CLE) to detect linguistic patterns
in journal text.
"""

import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Regex patterns for verb detection
# ---------------------------------------------------------------------------

# Auxiliary verbs
AUXILIARY_VERBS = re.compile(
    r"\b(am|is|are|was|were|be|been|being|have|has|had|do|does|did)\b",
    re.IGNORECASE,
)

# Modal verbs
MODAL_VERBS = re.compile(
    r"\b(will|would|shall|should|can|could|may|might|must)\b",
    re.IGNORECASE,
)

# Words ending in common verb suffixes (covers most regular verbs)
VERB_SUFFIX_PATTERN = re.compile(
    r"\b\w{2,}(ed|ing|es)\b",  # walked, thinking, goes (min 2 chars before suffix)
    re.IGNORECASE,
)

# Common base-form verbs (helps avoid false "no verb" on short sentences)
BASE_VERBS = re.compile(
    r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)\b("
    r"go|come|feel|think|want|need|make|take|see|hear|say|get|know|tell|put|"
    r"keep|leave|bring|work|sleep|eat|drink|walk|run|write|read|call|try|look|"
    r"use|help|move|live|stay|start|stop|plan|finish|hope|wait|hold|turn|open|"
    r"close|learn|remember|forget|ask|answer|reply|find|lose|pay|play|speak|"
    r"talk|sit|stand|drive|ride|meet|build|fix|change|check|focus|rest|wake|"
    r"grow)\b",
    re.IGNORECASE,
)
# Common irregular past tense verbs
IRREGULAR_PAST = re.compile(
    r"\b(went|thought|felt|knew|saw|came|got|made|said|took|told|found|gave|"
    r"left|put|read|ran|sat|stood|understood|wrote|began|broke|chose|"
    r"spoke|woke|drove|ate|fell|forgot|grew|hid|held|kept|led|lost|"
    r"met|paid|sent|set|shot|showed|shut|sang|slept|spent|spread|"
    r"taught|threw|wore|won|meant|heard|brought|caught|bought|built|dealt|"
    r"hung|laid|let|lit|rose|shook|shone|sank|spun|split|stuck|stung|struck|"
    r"swam|swung|tore|wound|became|bit|blew|drew|flew|froze|hurt|knelt|"
    r"rang|rode|sought|slid|sold|sprang|stole|swept|wept)\b",
    re.IGNORECASE,
)

# Common third-person singular present tense verbs (ending in 's' but not 'es')
# These are missed by the *es pattern but common in journal entries
THIRD_PERSON_VERBS = re.compile(
    r"\b(thinks|knows|feels|wants|needs|understands|believes|seems|appears|"
    r"works|helps|makes|takes|gives|says|tells|asks|looks|finds|uses|shows|"
    r"tries|means|keeps|lets|begins|becomes|remains|provides|requires|"
    r"includes|allows|suggests|considers|expects|creates|offers|produces|"
    r"serves|represents|supports|contains|involves|continues|follows|leads|"
    r"gets|puts|sits|stands|runs|comes|sees|hears|talks|walks|plays|stays|"
    r"calls|turns|holds|brings|writes|reads|learns|remembers|forgets|likes|"
    r"loves|hates|cares|matters|happens|exists|lives|dies|starts|stops|ends|"
    r"sleeps|eats|drinks|speaks|breaks|falls|grows|pays|sends|spends|meets)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Common punctuation patterns
# ---------------------------------------------------------------------------

ENDS_WITH_ELLIPSIS = re.compile(r"(?:\.{3,}|…)\s*$")
ENDS_WITH_SENTENCE_PUNCT = re.compile(r"[.!?]\s*$")
ENDS_WITH_DASH = re.compile(r"(—|--)\s*$")
STARTS_WITH_CONJUNCTION = re.compile(r"^\s*(but|and|so)\b", re.IGNORECASE)
MULTIPLE_DASHES = re.compile(r"--|—|–")  # em-dash, en-dash, or double hyphen
ELLIPSIS_CLUSTER = re.compile(r"(?:\.{3,}|…+)")  # 3+ dots or repeated ellipsis
MULTIPLE_EXCLAMATIONS = re.compile(r"!{3,}")
MULTIPLE_QUESTIONS = re.compile(r"\?{3,}")
MID_SENTENCE_DASH = re.compile(r"(—|--)")


# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------


def tokenize_simple(text: str) -> list[str]:
    """Simple word tokenization via regex (split on whitespace/punctuation)."""
    return re.findall(r"\b\w+\b", text)


@lru_cache(maxsize=64)
def _compile_phrase_pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a word-boundary regex for a phrase list (cached)."""
    if not phrases:
        return re.compile(r"(?!x)x")
    escaped = [re.escape(phrase) for phrase in phrases]
    pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    """Check if text contains any of the phrases (case-insensitive, word-boundary)."""
    return bool(_compile_phrase_pattern(phrases).search(text))


def has_verb(text: str) -> bool:
    """
    Check if text contains a verb-like pattern.

    Uses expanded detection:
    - Auxiliary verbs (am, is, are, was, were, have, has, had, do, does, did)
    - Modal verbs (will, would, can, could, should, etc.)
    - Words ending in verb suffixes (ed, ing, es)
    - Common irregular past tense verbs (went, thought, felt, knew, etc.)
    - Common base-form verbs (with a simple determiner guard)
    - Common third-person singular present tense verbs (thinks, knows, feels, etc.)
    """
    return (
        bool(AUXILIARY_VERBS.search(text))
        or bool(MODAL_VERBS.search(text))
        or bool(VERB_SUFFIX_PATTERN.search(text))
        or bool(IRREGULAR_PAST.search(text))
        or bool(BASE_VERBS.search(text))
        or bool(THIRD_PERSON_VERBS.search(text))
    )


# ---------------------------------------------------------------------------
# Word and phrase lists for feature extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NE (Narrative Entropy) Word Lists
# ---------------------------------------------------------------------------

# Temporal markers for ne_temporal_markers_rate
TEMPORAL_MARKERS: tuple[str, ...] = (
    "today",
    "yesterday",
    "tomorrow",
    "this morning",
    "this afternoon",
    "this evening",
    "tonight",
    "later",
    "later on",
    "earlier",
    "earlier today",
    "after",
    "afterwards",
    "before",
    "then",
    "when",
    "while",
    "now",
    "right now",
    "recently",
    "lately",
    "these days",
    "last night",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "next year",
    "at the time",
    "at first",
    "in the end",
    "eventually",
    "the other day",
)

# Strict past/future markers for jump detection (ne_temporal_jump_rate)
PAST_MARKERS: tuple[str, ...] = (
    "yesterday",
    "earlier",
    "earlier today",
    "last night",
    "last week",
    "last month",
    "last year",
)
FUTURE_MARKERS: tuple[str, ...] = (
    "tomorrow",
    "later",
    "later on",
    "next week",
    "next month",
    "next year",
    "soon",
)

# Glue connectors that bridge temporal jumps
GLUE_CONNECTORS: tuple[str, ...] = (
    "because",
    "so",
    "but",
    "then",
    "after",
    "before",
    "when",
    "while",
)

# Discourse connectors for ne_connectors_rate
CONNECTORS: tuple[str, ...] = (
    "because",
    "so",
    "but",
    "however",
    "although",
    "though",
    "therefore",
    "then",
    "after",
    "before",
    "instead",
    "still",
    "yet",
    "also",
    "meanwhile",
    "finally",
    "nevertheless",
    "regardless",
    "otherwise",
    "despite",
    "as a result",
)

# Reflection phrases for ne_reflection_rate (not emotions)
# These capture self-awareness, insight, perspective-taking, and growth recognition.
# Phrases are chosen to be specific enough to avoid false positives from casual usage.
REFLECTION_PHRASES: tuple[str, ...] = (
    # Realizations with context (avoid bare "i realized" which can be casual)
    "i realized that",
    "i realized how",
    "i realized why",
    "i realized what",
    "now i realize",
    "i've realized",
    "made me realize",
    # Learning about self (not just "i learned" which can be factual)
    "i've learned that",
    "i learned that",
    "i've learned to",
    "i learned to",
    "taught me that",
    "taught me about",
    "taught me a lot",
    "taught me so much",
    # Present tense insight (more specific)
    "i see now",
    "now i see",
    "i see where i",
    "i see why",
    # Progressive insight (natural journal phrasing)
    "starting to understand myself",
    "starting to understand why",
    "starting to understand how",
    "starting to see myself",
    "starting to see why",
    "starting to see how",
    "beginning to see myself",
    "beginning to see why",
    "beginning to see how",
    "beginning to understand myself",
    "beginning to understand why",
    "beginning to understand how",
    # Growth/change contractions (specific to personal growth)
    "i've grown",
    "i've come to",
    "i've been thinking",
    "i've changed",
    "i've noticed that i",
    "i've noticed how i",
    "i've noticed my",
    # Thinking/understanding (more specific)
    "it made me think",
    "helped me understand",
    "helped me see",
    "i understand now",
    "now i understand",
    # Retrospective self-analysis
    "in hindsight",
    "upon reflection",
    "time to reflect",
    "looking back,",  # comma indicates reflective pause, not literal looking
    "looking back at my",
    "looking back on",
    "i used to think",
    "i used to believe",
    "i used to be",
    "i was so certain",
    "i was afraid of",
    "i was wrong about",
    # Perspective/insight (specific enough)
    "gave me perspective",
    "given me perspective",
    "put things in perspective",
    "puts things in perspective",
    "opened my eyes",
    "it hit me that",
    "it clicked",
    "it dawned on me",
    # Change/growth recognition
    "something changed in me",
    "something shifted",
    "changed something in me",
    "changed me",
    "i'm finally learning",
    "finally understand",
    # Self-pattern recognition
    "my pattern",
    "my patterns",
    # Meaning-making (specific)
    "what matters is",
    "what's important is",
    "the point is",
    # Humility/growth markers
    "is humbling",
    "was humbling",
    "has been humbling",
    # Forward-looking reflection (with context)
    "going forward",
    "from now on",
)

# ---------------------------------------------------------------------------
# Narrative Closure Word Lists
# ---------------------------------------------------------------------------

# Closure cues: language signaling resolution, conclusion, or intentional ending.
CLOSURE_CUES: tuple[str, ...] = (
    # Resolution/conclusion
    "in the end",
    "at the end of the day",
    "all in all",
    "all things considered",
    "the bottom line",
    "the point is",
    "what matters is",
    "what's important is",
    # Acceptance/affirmation
    "that's enough",
    "that's okay",
    "that's alright",
    "that's fine",
    "it's okay",
    "it's alright",
    "it's fine",
    "it'll be okay",
    "everything's okay",
    "everything will be",
    "i'm okay with",
    "i'm at peace",
    "i can accept",
    "i accept that",
    # Achievement/milestone
    "i did it",
    "i made it",
    "i got the",
    "i finally",
    "we did it",
    "we made it",
    # Forward-looking resolution
    "from now on",
    "going forward",
    "moving forward",
    "tomorrow i'll",
    "tomorrow i will",
    "next time",
    "i'm ready to",
    "i'm going to",
    "time to move on",
    # Gratitude/contentment ending
    "i'm grateful",
    "i'm thankful",
    "grateful for",
    "thankful for",
    "that's all for now",
    "enough for now",
    "enough for today",
    "good night",
    "goodnight",
    # Summary phrases
    "so overall",
    "overall i",
    "to sum up",
    "in summary",
    "long story short",
    "the takeaway",
)

# Abandonment markers: language signaling unresolved/incomplete endings.
ABANDONMENT_MARKERS: tuple[str, ...] = (
    # Open questions (unresolved)
    "what do i do",
    "what am i supposed to",
    "what am i going to",
    "what should i",
    "what now",
    "now what",
    "where do i go",
    "how do i",
    "why does this",
    "why do i",
    "why can't i",
    "will it ever",
    "when will",
    "is it ever going to",
    "what's the point",
    "what's wrong with me",
    # Trailing-off language
    "i don't even know",
    "i just don't know",
    "i have no idea",
    "who knows",
    "who cares",
    "whatever",
    "i give up",
    "i can't anymore",
    "i can't do this",
    "so yeah",
    "but yeah",
    "but anyway",
    "anyway",
    "idk",
    "ugh",
)

# ---------------------------------------------------------------------------
# CLE (Cognitive Load Entropy) Word Lists
# ---------------------------------------------------------------------------

# Restart cues: explicit cognitive resets
RESTART_CUES: tuple[str, ...] = (
    "anyway",
    "wait,",
    "wait!",
    "wait.",
    "wait no",
    "or wait,",
    "hold on,",
    "hold on.",
    "i mean",
    "no,",
    "actually",
    "never mind",
    "nevermind",
    "scratch that",
    "just forget it",
    "ugh forget it",
    "so yeah",
    "where was i",
    "what was i saying",
    "let me start over",
)

# Hedge cues: uncertainty/unstable commitment
HEDGE_CUES: tuple[str, ...] = (
    "maybe",
    "i guess",
    "i suppose",
    "i think",
    "i'm not sure",
    "not sure",
    "i'm unsure",
    "i don't know",
    "kind of",
    "sort of",
    "probably",
    "perhaps",
    "possibly",
    "might be",
    "who knows",
    "it feels like",
)

# ---------------------------------------------------------------------------
# BC (Belief Conflict) Word Lists
# ---------------------------------------------------------------------------

# Belief statement patterns (sentence prefixes indicating self-referential beliefs)
BELIEF_PATTERNS: tuple[str, ...] = (
    # Identity statements
    "i am ",
    "i'm ",
    "i feel ",
    "i'm feeling ",
    "i feel like ",
    # Modal commitments
    "i will ",
    "i'll ",
    "i won't ",
    "i can ",
    "i can't ",
    "i cannot ",
    "i could ",
    "i couldn't ",
    # Habitual self-perception
    "i always ",
    "i never ",
    "i usually ",
    # Value/relationship statements
    "i love ",
    "i hate ",
    "i want ",
    "i need ",
    "i should ",
    "i shouldn't ",
    "i must ",
    # Self-knowledge claims
    "i know ",
    "i believe ",
    "i think that ",
    "i'm sure ",
    "i'm certain ",
    "i'm confident ",
    "i'm afraid ",
    "i'm scared ",
    "i'm worried ",
    "i'm happy ",
    "i'm sad ",
    "i'm angry ",
)

# Negation words for polarity detection
NEGATION_WORDS: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "cannot",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "nothing",
    "nobody",
    "nowhere",
    "none",
)

# Polar word pairs (each key maps to its opposite)
POLAR_WORD_PAIRS: dict[str, str] = {
    # Confidence/worth
    "confident": "worthless",
    "worthless": "confident",
    "capable": "incapable",
    "incapable": "capable",
    "strong": "weak",
    "weak": "strong",
    "enough": "inadequate",
    "inadequate": "enough",
    # Emotions
    "love": "hate",
    "hate": "love",
    "happy": "sad",
    "sad": "happy",
    "hopeful": "hopeless",
    "hopeless": "hopeful",
    # Success/failure
    "success": "failure",
    "failure": "success",
    "succeed": "fail",
    "fail": "succeed",
    # Certainty
    "certain": "uncertain",
    "uncertain": "certain",
    "sure": "unsure",
    "unsure": "sure",
    # Morality
    "right": "wrong",
    "wrong": "right",
    "good": "bad",
    "bad": "good",
    # Connection
    "connected": "alone",
    "alone": "connected",
    "loved": "unloved",
    "unloved": "loved",
}

# Integration/resolution markers (signal that contradictions are being processed)
INTEGRATION_MARKERS: tuple[str, ...] = (
    # Explicit reconciliation
    "but i've realized",
    "but i realize",
    "but now i see",
    "and yet i see now",
    "and yet i understand",
    "both are true",
    "both can be true",
    "i can hold both",
    "i can see both",
    "it's complicated",
    "it's complex",
    "it's not that simple",
    # Temporal resolution (past vs present self)
    "i used to think",
    "i used to believe",
    "i used to feel",
    "now i see",
    "now i understand",
    "now i realize",
    "i've come to see",
    "i've come to understand",
    "i've come to realize",
    # Paradox acknowledgment
    "at the same time",
    "on the other hand",
    "even though",
    "despite",
    "although",
    "nevertheless",
    "nonetheless",
    "while also",
    # Growth/change markers
    "i've changed",
    "i've grown",
    "i've learned",
    "i'm learning to",
    "i'm starting to see",
    "i'm beginning to understand",
    "looking back",
    # Synthesis language
    "i can be both",
    "both sides",
    "makes sense now",
    "it all makes sense",
)
