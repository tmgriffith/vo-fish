"""Tag-mode handling for the renderer.

- auto:     no-op (script already has tags from Claude / caller)
- explicit: no-op (script has tags, leave them alone)
- none:     strip all [bracketed] tags
"""
from __future__ import annotations

import re

__all__ = ["apply_tag_mode", "TagModeError", "VALID_TAG_MODES"]

VALID_TAG_MODES = ("auto", "explicit", "none")

# Matches [...] not preceded by < and not containing < or > so we don't eat
# <|speaker:i|> speaker tokens. Greedy-free.
_TAG_RE = re.compile(r"\[[^\[\]<>]*\]")
_WS_RE = re.compile(r"[ \t]+")


class TagModeError(ValueError):
    """Unknown tag_mode value."""


def apply_tag_mode(script: str, mode: str) -> str:
    if mode not in VALID_TAG_MODES:
        raise TagModeError(
            f"unknown tag_mode {mode!r} — valid: {', '.join(VALID_TAG_MODES)}"
        )
    if mode in ("auto", "explicit"):
        return script
    # mode == "none": strip bracketed tags, collapse whitespace per line
    stripped = _TAG_RE.sub("", script)
    out_lines = []
    for line in stripped.splitlines():
        out_lines.append(_WS_RE.sub(" ", line).strip())
    return "\n".join(l for l in out_lines if l)
