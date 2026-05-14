"""Resolve --voice / --ref-audio into a concrete (audio path, transcript)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vo.registries import (
    DEFAULT_VOICES_PATH, load_voices, get_voice, Voice
)

__all__ = ["ResolvedVoice", "VoiceResolutionError", "resolve_voice"]


class VoiceResolutionError(Exception):
    pass


@dataclass
class ResolvedVoice:
    voice_id: str | None        # None for ad-hoc or model-default
    audio_path: Path | None     # None for model-default
    transcript: str | None      # None for model-default
    source: str                 # "registry" | "ad_hoc" | "model_default"


def resolve_voice(
    *,
    voice_id: str | None,
    ref_audio: Path | None,
    ref_text: str | None,
    voices_path: Path | None = DEFAULT_VOICES_PATH,
    transcribe_fn: Callable[[Path], str] | None = None,
) -> ResolvedVoice:
    if voice_id is not None and ref_audio is not None:
        raise VoiceResolutionError(
            "cannot supply both --voice and --ref-audio"
        )

    if voice_id is not None:
        if voices_path is None:
            raise VoiceResolutionError("voices_path required when voice_id is given")
        v = get_voice(voice_id, voices_path)
        return ResolvedVoice(voice_id=v.id, audio_path=v.audio_path,
                             transcript=v.transcript, source="registry")

    if ref_audio is not None:
        if not ref_audio.exists():
            raise VoiceResolutionError(f"ref_audio not found: {ref_audio}")
        if ref_text is None:
            if transcribe_fn is None:
                # Lazy import so tests can fully mock when desired.
                from vo.transcribe import transcribe as _t
                transcribe_fn = _t
            ref_text = transcribe_fn(ref_audio)
        return ResolvedVoice(voice_id=None, audio_path=ref_audio,
                             transcript=ref_text, source="ad_hoc")

    # Neither given — try registry default
    if voices_path is not None and voices_path.exists():
        raw = json.loads(voices_path.read_text())
        default_id = raw.get("default")
        if default_id:
            v = get_voice(default_id, voices_path)
            return ResolvedVoice(voice_id=v.id, audio_path=v.audio_path,
                                 transcript=v.transcript, source="registry")

    return ResolvedVoice(voice_id=None, audio_path=None,
                         transcript=None, source="model_default")
