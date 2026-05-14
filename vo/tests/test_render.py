"""Tests for vo.render — happy path with mocked model."""
import json
from pathlib import Path

import pytest
import mlx.core as mx

from vo.tests.conftest import FakeFishModel, FakeWhisper


@pytest.fixture
def fake_renderer(monkeypatch):
    import vo.render as render
    tts = FakeFishModel()
    stt = FakeWhisper(transcript="hello world", words=tts.words)
    monkeypatch.setattr(render, "_get_model", lambda: tts)
    monkeypatch.setattr(render, "_get_stt", lambda: stt)
    # Stub out the real audio loader so we don't need a real file on disk
    monkeypatch.setattr(render, "_load_ref_audio", lambda path, sr: mx.zeros((sr,)))
    # Stub out the audio writer so we don't need mlx_audio.audio_io
    def _fake_write(path, audio, sample_rate):
        import wave, struct
        path.parent.mkdir(parents=True, exist_ok=True)
        n_samples = audio.shape[0] if hasattr(audio, "shape") else len(audio)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    monkeypatch.setattr(render, "_write_audio", _fake_write)
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
