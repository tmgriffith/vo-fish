"""Quality gate utilities — Whisper word extraction, silence gaps, anchor matching."""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Word", "extract_words", "largest_word_gap",
    "find_anchor_starts", "QualityCheck", "evaluate",
]

_WORD_CHAR_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str  # already normalized (lower, alnum + ')


def _normalize(s: str) -> str:
    return "".join(_WORD_CHAR_RE.findall((s or "").lower()))


def _segments(stt_result) -> list:
    """Pull a list of segments out of a result that may be a dict or object."""
    if isinstance(stt_result, dict):
        return stt_result.get("segments", []) or []
    segs = getattr(stt_result, "segments", None)
    if segs is None and hasattr(stt_result, "__dict__"):
        segs = stt_result.__dict__.get("segments", [])
    return segs or []


def _seg_words(seg) -> list:
    if isinstance(seg, dict):
        return seg.get("words", []) or []
    w = getattr(seg, "words", None)
    if w is None and hasattr(seg, "__dict__"):
        w = seg.__dict__.get("words", [])
    return w or []


def extract_words(stt_result) -> list[Word]:
    """Flatten Whisper output into a list of normalized Word triples."""
    out: list[Word] = []
    for seg in _segments(stt_result):
        for w in _seg_words(seg):
            data = w if isinstance(w, dict) else w.__dict__
            ws = data.get("start")
            we = data.get("end")
            raw = data.get("word") or data.get("text") or ""
            text = _normalize(raw)
            if ws is not None and we is not None and text:
                out.append(Word(float(ws), float(we), text))
    return out


def largest_word_gap(words: list[Word]) -> float:
    """Largest inter-word silence in seconds. Returns +inf for <2 words."""
    if len(words) < 2:
        return float("inf")
    return max(words[i].start - words[i - 1].end for i in range(1, len(words)))


def find_anchor_starts(words, anchors):
    """Placeholder — implemented in next task."""
    raise NotImplementedError


@dataclass
class QualityCheck:
    passed: bool
    max_gap: float
    anchor_starts: list[float] | None
    reason: str = ""


def evaluate(words, max_silence_gap, anchors=None):
    raise NotImplementedError
