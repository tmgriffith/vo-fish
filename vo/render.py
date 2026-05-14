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
    max_retries: int = 4,
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

    # 3. Generation with quality-gate retry loop
    import mlx.core as mx

    attempts_used = 0
    final_audio = None
    final_words = None
    final_check: QualityCheck | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        attempts_used = attempt
        # Perturb temperature on retry to break out of stuck states
        attempt_temp = temperature + 0.05 * (attempt - 1)
        results = list(model.generate(
            text=final_script,
            ref_audio=ref_arr,
            ref_text=rv.transcript,
            temperature=attempt_temp,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            chunk_length=chunk_length,
            speed=speed,
        ))
        if not results:
            continue
        audio = results[0].audio if len(results) == 1 else mx.concatenate([r.audio for r in results])
        mx.eval(audio)
        _write_audio(out_path, audio, sr)
        final_audio = audio

        if no_stt:
            final_check = QualityCheck(passed=True, max_gap=0.0,
                                       anchor_starts=None, reason="")
            break

        stt = _get_stt()
        r = stt.generate(str(out_path), word_timestamps=True,
                         return_timestamps=True, condition_on_previous_text=False)
        words = extract_words(r)
        check = evaluate(words, max_silence_gap=max_silence_gap, anchors=anchors)
        final_words = words
        final_check = check
        if check.passed:
            break

    if final_audio is None:
        raise RuntimeError("Fish Speech returned no audio segments across all retries")
    assert final_check is not None

    duration_s = float(final_audio.shape[0]) / sr

    # 4. Sidecars
    tagged_path = out_path.with_suffix(".tagged.txt")
    tagged_path.write_text(final_script)

    words_path: Path | None = None
    if not no_stt:
        words_path = out_path.with_suffix(".words.json")
        words_path.write_text(json.dumps({
            "duration_s": duration_s,
            "max_gap": final_check.max_gap,
            "quality_passed": final_check.passed,
            "quality_reason": final_check.reason,
            "anchor_starts": final_check.anchor_starts,
            "words": [{"start": w.start, "end": w.end, "text": w.text}
                      for w in (final_words or [])],
        }, indent=2))

    return RenderResult(
        wav_path=out_path,
        words_path=words_path,
        tagged_path=tagged_path,
        voice_id=rv.voice_id,
        duration_s=duration_s,
        attempts_used=attempts_used,
        quality_passed=final_check.passed,
    )


# ---- CLI ---------------------------------------------------------------

import argparse
import json as _cli_json

from vo.registries import (
    DEFAULT_PRESETS_PATH, load_presets, Preset,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vo",
        description="Fish Audio S2 Pro voiceover renderer.",
    )
    p.add_argument("--script", help="Script file (- for stdin).")
    p.add_argument("--out", help="Output WAV path.")

    # Voice selection (mutually exclusive enforced post-parse, not via argparse,
    # because the spec requires a custom error message).
    p.add_argument("--voice", help="Voice ID from voices.json.")
    p.add_argument("--ref-audio", help="Ad-hoc reference audio path.")
    p.add_argument("--ref-text", help="Transcript of --ref-audio. Auto-Whispered if missing.")

    # Preset
    p.add_argument("--preset", help="Preset name (defaults from presets.json).")
    p.add_argument("--no-preset", action="store_true",
                   help="Ignore any preset defaults.")

    # Sampling
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--speed", type=float)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--chunk-length", type=int, default=300)

    # Quality gate
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--max-silence-gap", type=float, default=2.5)
    p.add_argument("--no-stt", action="store_true")
    p.add_argument("--anchors-json",
                   help="JSON-encoded list of anchor phrase lists, e.g. "
                        "'[[\"most\",\"content\"],[\"that\\u2019s\",\"not\"]]'.")

    # Features
    p.add_argument("--multi-speaker", action="store_true")
    p.add_argument("--language", default="en")
    p.add_argument("--tag-mode", choices=["auto", "explicit", "none"], default="auto")
    p.add_argument("--seed", type=int)

    # Registry paths (mainly for tests; can also be overridden in practice)
    p.add_argument("--voices-path", default=str(DEFAULT_VOICES_PATH))
    p.add_argument("--presets-path", default=str(DEFAULT_PRESETS_PATH))

    # Admin paths (filled in next task)
    p.add_argument("--save-voice")
    p.add_argument("--label")
    p.add_argument("--notes")
    p.add_argument("--save-preset")
    p.add_argument("--preset-notes")
    p.add_argument("--add-voice")
    p.add_argument("--audio")
    p.add_argument("--transcript")
    p.add_argument("--add-preset")
    p.add_argument("--json")
    p.add_argument("--transcribe")

    return p


def _read_script(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text()


def _apply_preset(args, preset: Preset) -> None:
    """Fill un-set sampling args from preset defaults."""
    if args.voice is None and preset.voice is not None:
        args.voice = preset.voice
    if args.temperature is None:
        args.temperature = preset.temperature
    if args.top_p is None:
        args.top_p = preset.top_p
    if args.top_k is None:
        args.top_k = preset.top_k
    if args.speed is None:
        args.speed = preset.speed
    if args.language == "en" and preset.language != "en":
        args.language = preset.language


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Admin paths handled in a later task; here we only do the render path.
    if any([args.add_voice, args.add_preset, args.transcribe]):
        return _admin_main(args)

    if not args.script or not args.out:
        parser.error("--script and --out are required for rendering")

    if args.voice and args.ref_audio:
        print("error: cannot supply both --voice and --ref-audio",
              file=sys.stderr)
        return 2

    # Preset defaults (--no-preset wins)
    if args.preset and not args.no_preset:
        from vo.registries import RegistryError as _RegistryError
        try:
            preset = load_presets(Path(args.presets_path))[args.preset]
        except KeyError:
            print(f"error: unknown preset {args.preset!r}", file=sys.stderr)
            return 2
        except _RegistryError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        _apply_preset(args, preset)

    # Final fallback defaults
    if args.temperature is None: args.temperature = 0.7
    if args.top_p is None:       args.top_p = 0.7
    if args.top_k is None:       args.top_k = 30
    if args.speed is None:       args.speed = 1.0

    anchors = None
    if args.anchors_json:
        anchors = _cli_json.loads(args.anchors_json)

    script_text = _read_script(args.script)

    try:
        result = render(
            script=script_text,
            out_path=Path(args.out),
            voice=args.voice,
            ref_audio=Path(args.ref_audio) if args.ref_audio else None,
            ref_text=args.ref_text,
            voices_path=Path(args.voices_path),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            speed=args.speed,
            max_tokens=args.max_tokens,
            chunk_length=args.chunk_length,
            no_stt=args.no_stt,
            multi_speaker=args.multi_speaker,
            language=args.language,
            tag_mode=args.tag_mode,
            anchors=anchors,
            max_retries=args.max_retries,
            max_silence_gap=args.max_silence_gap,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(_cli_json.dumps({
        "wav": str(result.wav_path),
        "words": str(result.words_path) if result.words_path else None,
        "tagged": str(result.tagged_path),
        "voice_id": result.voice_id,
        "duration_s": result.duration_s,
        "attempts_used": result.attempts_used,
        "quality_passed": result.quality_passed,
    }))

    if args.save_voice and result.quality_passed and args.ref_audio:
        from vo.registries import Voice as _Voice, add_voice as _add_voice
        # Reuse the transcript we already resolved (via resolve_voice in render);
        # re-resolve here to avoid threading it through RenderResult.
        from vo.voice_resolver import resolve_voice as _resolve
        rv = _resolve(voice_id=None, ref_audio=Path(args.ref_audio),
                      ref_text=args.ref_text, voices_path=Path(args.voices_path))
        v = _Voice(
            id=args.save_voice,
            label=args.label or args.save_voice,
            audio=str(Path(args.ref_audio)),
            transcript=rv.transcript or "",
            notes=args.notes or "",
        )
        _add_voice(v, Path(args.voices_path))
        print(_cli_json.dumps({"saved_voice": args.save_voice}))

    if args.save_preset and result.quality_passed:
        from vo.registries import Preset as _Preset, add_preset as _add_preset
        p = _Preset(
            name=args.save_preset,
            voice=args.voice,
            tag_hints=[],
            tag_density="medium",
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            speed=args.speed,
            language=args.language,
            notes=args.preset_notes or "",
        )
        _add_preset(p, Path(args.presets_path))
        print(_cli_json.dumps({"saved_preset": args.save_preset}))

    return 0 if result.quality_passed else 5


def _admin_main(args) -> int:
    from vo.registries import add_voice, add_preset, Voice, Preset

    if args.transcribe:
        from vo.transcribe import transcribe
        audio = Path(args.transcribe)
        if not audio.exists():
            print(f"error: not found: {audio}", file=sys.stderr)
            return 2
        text = transcribe(audio)
        print(text.strip())
        return 0

    if args.add_voice:
        if not args.audio:
            print("error: --add-voice requires --audio", file=sys.stderr)
            return 2
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"error: --audio not found: {audio_path}", file=sys.stderr)
            return 2
        transcript = args.transcript
        if transcript is None:
            from vo.transcribe import transcribe
            transcript = transcribe(audio_path).strip()
        v = Voice(
            id=args.add_voice,
            label=args.label or args.add_voice,
            audio=str(audio_path),
            transcript=transcript,
            notes=args.notes or "",
        )
        add_voice(v, Path(args.voices_path))
        print(_cli_json.dumps({"added_voice": args.add_voice,
                               "path": args.voices_path}))
        return 0

    if args.add_preset:
        if not args.json:
            print("error: --add-preset requires --json", file=sys.stderr)
            return 2
        try:
            payload = _cli_json.loads(args.json)
        except _cli_json.JSONDecodeError as e:
            print(f"error: --json invalid: {e}", file=sys.stderr)
            return 2
        p = Preset(
            name=args.add_preset,
            voice=payload.get("voice"),
            tag_hints=list(payload.get("tag_hints", [])),
            tag_density=payload.get("tag_density", "medium"),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.7)),
            top_k=int(payload.get("top_k", 30)),
            speed=float(payload.get("speed", 1.0)),
            language=payload.get("language", "en"),
            notes=args.preset_notes or payload.get("notes", ""),
        )
        add_preset(p, Path(args.presets_path))
        print(_cli_json.dumps({"added_preset": args.add_preset,
                               "path": args.presets_path}))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
