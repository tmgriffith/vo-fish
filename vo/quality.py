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


def find_anchor_starts(words: list[Word], anchors: list[list[str]]) -> list[float] | None:
    """For each anchor (a list of consecutive content words), find the start
    time of its first word in `words`. Anchors are matched in order; the
    search cursor advances after each match so anchors stay monotonic.

    Up to 3 stray words are allowed between consecutive anchor words to absorb
    Whisper filler/punctuation glitches.

    Returns:
      list of floats (one per anchor). If a specific anchor can't be matched
      it gets backfilled by averaging its surrounding matched anchors. If no
      anchor matches at all, returns None.
    """
    norm = [(w.start, w.end, w.text) for w in words]
    starts: list[float | None] = []
    cursor = 0
    for anchor in anchors:
        if not anchor:
            starts.append(None)
            continue
        head = anchor[0]
        found = None
        i = cursor
        while i < len(norm):
            if norm[i][2] == head:
                # try to match the rest of the anchor within a small window
                j = i
                ok = True
                for tok in anchor[1:]:
                    advanced = False
                    for k in range(j + 1, min(j + 5, len(norm))):
                        if norm[k][2] == tok:
                            j = k
                            advanced = True
                            break
                    if not advanced:
                        ok = False
                        break
                if ok:
                    found = norm[i][0]
                    cursor = j + 1
                    break
            i += 1
        starts.append(found)

    if all(s is None for s in starts):
        return None

    last_end = norm[-1][1] if norm else 0.0
    for idx in range(len(starts)):
        if starts[idx] is None:
            prev = next((starts[k] for k in range(idx - 1, -1, -1) if starts[k] is not None), 0.0)
            nxt = next((starts[k] for k in range(idx + 1, len(starts)) if starts[k] is not None), last_end)
            starts[idx] = (prev + nxt) / 2

    # enforce monotonic increase
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            starts[i] = starts[i - 1] + 0.1
    return [float(s) for s in starts]  # type: ignore[arg-type]


@dataclass
class QualityCheck:
    passed: bool
    max_gap: float
    anchor_starts: list[float] | None
    reason: str = ""


def evaluate(words, max_silence_gap, anchors=None):
    raise NotImplementedError
