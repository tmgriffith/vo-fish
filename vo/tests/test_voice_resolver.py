"""Tests for vo.voice_resolver."""
from pathlib import Path

import pytest

from vo.voice_resolver import resolve_voice, ResolvedVoice, VoiceResolutionError


def test_resolve_voice_from_registry(sample_voices_path):
    rv = resolve_voice(voice_id="excited", ref_audio=None, ref_text=None,
                       voices_path=sample_voices_path, transcribe_fn=None)
    assert isinstance(rv, ResolvedVoice)
    assert rv.voice_id == "excited"
    assert rv.transcript == "Test transcript."
    assert rv.audio_path.name == "Excited VO clone sample.m4a"


def test_resolve_voice_ad_hoc_with_transcript(tmp_path):
    audio = tmp_path / "ad.wav"
    audio.write_bytes(b"")
    rv = resolve_voice(voice_id=None, ref_audio=audio, ref_text="hi",
                       voices_path=None, transcribe_fn=None)
    assert rv.voice_id is None
    assert rv.transcript == "hi"
    assert rv.audio_path == audio


def test_resolve_voice_ad_hoc_auto_transcribes(tmp_path):
    audio = tmp_path / "ad.wav"
    audio.write_bytes(b"")
    called = {}
    def fake_transcribe(p):
        called["p"] = p
        return "whisper-derived transcript"
    rv = resolve_voice(voice_id=None, ref_audio=audio, ref_text=None,
                       voices_path=None, transcribe_fn=fake_transcribe)
    assert rv.transcript == "whisper-derived transcript"
    assert called["p"] == audio


def test_resolve_voice_neither_returns_default_voice_marker(sample_voices_path):
    """When neither voice_id nor ref_audio is supplied, fall back to the registry
    default if present."""
    rv = resolve_voice(voice_id=None, ref_audio=None, ref_text=None,
                       voices_path=sample_voices_path, transcribe_fn=None)
    assert rv.voice_id == "excited"


def test_resolve_voice_both_raises(tmp_path):
    audio = tmp_path / "ad.wav"
    audio.write_bytes(b"")
    with pytest.raises(VoiceResolutionError, match="cannot supply both"):
        resolve_voice(voice_id="excited", ref_audio=audio, ref_text=None,
                      voices_path=None, transcribe_fn=None)


def test_resolve_voice_ad_hoc_missing_audio_raises(tmp_path):
    with pytest.raises(VoiceResolutionError, match="ref_audio not found"):
        resolve_voice(voice_id=None, ref_audio=tmp_path / "nope.wav",
                      ref_text="t", voices_path=None, transcribe_fn=None)


def test_resolve_voice_no_default_no_args_returns_model_default(tmp_path):
    """If no default and no args, return a marker for the model's built-in voice."""
    import json
    p = tmp_path / "voices.json"
    p.write_text(json.dumps({"version": 1, "voices": {}}))
    rv = resolve_voice(voice_id=None, ref_audio=None, ref_text=None,
                       voices_path=p, transcribe_fn=None)
    assert rv.voice_id is None
    assert rv.audio_path is None
    assert rv.transcript is None
