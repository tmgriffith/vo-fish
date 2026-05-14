"""Whisper transcription helper for reference audio."""
from __future__ import annotations

from pathlib import Path

__all__ = ["transcribe", "STT_MODEL"]

STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"

_STT = None


def _get_stt():
    global _STT
    if _STT is None:
        from mlx_audio.stt import load
        _STT = load(STT_MODEL)
    return _STT


def transcribe(audio_path: Path | str) -> str:
    """Return the Whisper transcript of `audio_path`."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    stt = _get_stt()
    r = stt.generate(str(audio_path))
    return getattr(r, "text", None) or str(r)
