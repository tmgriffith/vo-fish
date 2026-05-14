"""Fish Audio S2 Pro renderer — the only module that touches MLX."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

from vo import _version_check  # noqa: F401
from vo.registries import DEFAULT_VOICES_PATH
from vo.tags import apply_tag_mode
from vo.voice_resolver import resolve_voice
from vo.quality import extract_words, evaluate, QualityCheck

__all__ = ["render", "RenderResult"]

# ---- Lazy model loaders (monkeypatchable in tests) ---------------------

_MODEL = None
_STT = None
_MODEL_PATH = "~/Library/Caches/mlx-audio/fish-audio-s2-pro-bf16"


def _get_model():
    global _MODEL
    if _MODEL is None:
        import os
        from mlx_audio.tts.utils import load_model
        _MODEL = load_model(os.path.expanduser(_MODEL_PATH))
    return _MODEL


def _get_stt():
    global _STT
    if _STT is None:
        from mlx_audio.stt import load
        _STT = load("mlx-community/whisper-large-v3-turbo-asr-fp16")
    return _STT


def _load_ref_audio(ref_path: Path, sample_rate: int):
    from mlx_audio.utils import load_audio
    return load_audio(str(ref_path), sample_rate=sample_rate, volume_normalize=False)


def _write_audio(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    from mlx_audio.audio_io import write as audio_write
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_write(str(path), np.array(audio), sample_rate, format="wav")


# ---- Public API --------------------------------------------------------

@dataclass
class RenderResult:
    wav_path: Path
    words_path: Path | None
    tagged_path: Path
    voice_id: str | None
    duration_s: float
    attempts_used: int
    quality_passed: bool


def render(
    *,
    script: str,
    out_path: Path,
    voice: str | None = None,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    voices_path: Path = DEFAULT_VOICES_PATH,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 30,
    speed: float = 1.0,
    max_tokens: int = 4096,
    chunk_length: int = 300,
    no_stt: bool = False,
    multi_speaker: bool = False,
    language: str = "en",
    tag_mode: str = "auto",
    anchors: list[list[str]] | None = None,
    max_retries: int = 1,
    max_silence_gap: float = 2.5,
) -> RenderResult:
    out_path = Path(out_path)

    # 1. Tag preprocessing
    final_script = apply_tag_mode(script, tag_mode)

    # 2. Resolve voice
    rv = resolve_voice(
        voice_id=voice, ref_audio=ref_audio, ref_text=ref_text,
        voices_path=voices_path,
    )

    model = _get_model()
    sr = int(model.sample_rate)

    ref_arr = _load_ref_audio(rv.audio_path, sr) if rv.audio_path else None

    # 3. Generate
    results = list(model.generate(
        text=final_script,
        ref_audio=ref_arr,
        ref_text=rv.transcript,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        chunk_length=chunk_length,
        speed=speed,
    ))
    if not results:
        raise RuntimeError("Fish Speech returned no audio segments")

    import mlx.core as mx
    # Concatenate any segments
    if len(results) == 1:
        audio = results[0].audio
    else:
        audio = mx.concatenate([r.audio for r in results])

    # Materialise before numpy conversion in _write_audio.
    mx.eval(audio)

    _write_audio(out_path, audio, sr)
    duration_s = float(audio.shape[0]) / sr

    # 4. Sidecars (sttless path is short-circuited later in this plan)
    tagged_path = out_path.with_suffix(".tagged.txt")
    tagged_path.write_text(final_script)

    words_path: Path | None = None
    quality_passed = True
    if not no_stt:
        stt = _get_stt()
        r = stt.generate(str(out_path), word_timestamps=True,
                         return_timestamps=True, condition_on_previous_text=False)
        words = extract_words(r)
        check: QualityCheck = evaluate(words, max_silence_gap=max_silence_gap,
                                       anchors=anchors)
        quality_passed = check.passed
        words_path = out_path.with_suffix(".words.json")
        words_path.write_text(json.dumps({
            "duration_s": duration_s,
            "max_gap": check.max_gap,
            "quality_passed": check.passed,
            "quality_reason": check.reason,
            "anchor_starts": check.anchor_starts,
            "words": [{"start": w.start, "end": w.end, "text": w.text}
                      for w in words],
        }, indent=2))

    return RenderResult(
        wav_path=out_path,
        words_path=words_path,
        tagged_path=tagged_path,
        voice_id=rv.voice_id,
        duration_s=duration_s,
        attempts_used=1,
        quality_passed=quality_passed,
    )
