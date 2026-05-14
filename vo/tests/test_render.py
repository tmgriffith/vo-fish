"""Tests for vo.render — happy path with mocked model."""
import json
from pathlib import Path

import pytest
import mlx.core as mx

from vo.tests.conftest import FakeFishModel, FakeWhisper


@pytest.fixture
def fake_renderer(monkeypatch, patch_render_audio_io):
    import vo.render as render
    tts = FakeFishModel()
    stt = FakeWhisper(transcript="hello world", words=tts.words)
    monkeypatch.setattr(render, "_get_model", lambda: tts)
    monkeypatch.setattr(render, "_get_stt", lambda: stt)
    return render, tts, stt


def test_render_happy_path_writes_wav(fake_renderer, tmp_path, sample_voices_path):
    render, tts, stt = fake_renderer
    out = tmp_path / "out.wav"
    result = render.render(
        script="Hello [excited] world.",
        out_path=out,
        voice="excited",
        voices_path=sample_voices_path,
    )
    assert out.exists()
    assert result.wav_path == out
    assert result.duration_s == pytest.approx(2.0)
    assert result.attempts_used == 1
    assert result.quality_passed is True


def test_render_passes_ref_audio_and_text_to_model(fake_renderer, tmp_path, sample_voices_path):
    render, tts, _ = fake_renderer
    out = tmp_path / "out.wav"
    render.render(script="hi", out_path=out, voice="excited",
                  voices_path=sample_voices_path)
    call = tts.calls[0]
    assert call["text"] == "hi"
    assert call["ref_text"] == "Test transcript."
    # ref_audio is the loaded mx.array, not a path
    assert call["ref_audio"] is not None


def test_render_strips_tags_when_tag_mode_none(fake_renderer, tmp_path, sample_voices_path):
    render, tts, _ = fake_renderer
    out = tmp_path / "out.wav"
    render.render(script="Hi [excited] there.", out_path=out, voice="excited",
                  voices_path=sample_voices_path, tag_mode="none")
    assert tts.calls[0]["text"] == "Hi there."


def test_render_model_default_voice(fake_renderer, tmp_path):
    """No voice + no ref → ref_audio=None, ref_text=None."""
    render, tts, _ = fake_renderer
    out = tmp_path / "out.wav"
    # Use an empty registry so there's no default
    empty = tmp_path / "voices.json"
    empty.write_text(json.dumps({"version": 1, "voices": {}}))
    render.render(script="hi", out_path=out, voices_path=empty)
    call = tts.calls[0]
    assert call["ref_audio"] is None
    assert call["ref_text"] is None


def test_render_retries_on_silence_gap(monkeypatch, tmp_path, sample_voices_path, patch_render_audio_io):
    import vo.render as render
    bad_words = [
        {"start": 0.0, "end": 1.0, "word": "hello"},
        {"start": 6.0, "end": 7.0, "word": "world"},  # 5s gap
    ]
    good_words = [
        {"start": 0.0, "end": 0.5, "word": "hello"},
        {"start": 0.6, "end": 1.0, "word": "world"},
    ]
    tts = FakeFishModel()
    # Whisper returns bad words on first call, good on second
    stt = FakeWhisper(words=bad_words)
    call_count = {"n": 0}
    _orig_gen = stt.generate
    def stt_generate(audio, **kw):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            stt._words = good_words
        return _orig_gen(audio, **kw)
    stt.generate = stt_generate
    monkeypatch.setattr(render, "_get_model", lambda: tts)
    monkeypatch.setattr(render, "_get_stt", lambda: stt)

    out = tmp_path / "out.wav"
    result = render.render(
        script="hi", out_path=out, voice="excited",
        voices_path=sample_voices_path,
        max_retries=3, max_silence_gap=2.5,
    )
    assert result.attempts_used == 2
    assert result.quality_passed is True
    assert len(tts.calls) == 2


def test_render_exhausts_retries_and_marks_failed(monkeypatch, tmp_path, sample_voices_path, patch_render_audio_io):
    import vo.render as render
    bad_words = [
        {"start": 0.0, "end": 1.0, "word": "a"},
        {"start": 6.0, "end": 7.0, "word": "b"},
    ]
    tts = FakeFishModel()
    stt = FakeWhisper(words=bad_words)
    monkeypatch.setattr(render, "_get_model", lambda: tts)
    monkeypatch.setattr(render, "_get_stt", lambda: stt)
    out = tmp_path / "out.wav"
    result = render.render(
        script="hi", out_path=out, voice="excited",
        voices_path=sample_voices_path,
        max_retries=3, max_silence_gap=2.5,
    )
    assert result.attempts_used == 3
    assert result.quality_passed is False


def test_render_no_stt_skips_quality_gate(monkeypatch, tmp_path, sample_voices_path, patch_render_audio_io):
    import vo.render as render
    tts = FakeFishModel()
    stt_called = {"n": 0}
    def boom():
        stt_called["n"] += 1
        raise RuntimeError("should not be called")
    monkeypatch.setattr(render, "_get_model", lambda: tts)
    monkeypatch.setattr(render, "_get_stt", boom)
    out = tmp_path / "out.wav"
    result = render.render(script="hi", out_path=out, voice="excited",
                           voices_path=sample_voices_path, no_stt=True)
    assert stt_called["n"] == 0
    assert result.words_path is None
    assert result.quality_passed is True
