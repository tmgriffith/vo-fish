"""Tests for vo.render CLI."""
import sys
from pathlib import Path

import pytest

import vo.render as render
from vo.tests.conftest import FakeFishModel, FakeWhisper


@pytest.fixture
def cli_setup(monkeypatch, sample_voices_path, tmp_path, patch_render_audio_io):
    tts = FakeFishModel()
    stt = FakeWhisper(words=tts.words)
    monkeypatch.setattr(render, "_get_model", lambda: tts)
    monkeypatch.setattr(render, "_get_stt", lambda: stt)
    return tts, stt, sample_voices_path, tmp_path


def test_cli_renders_via_script_file(cli_setup, capsys):
    tts, stt, voices, tmp = cli_setup
    script = tmp / "script.txt"
    script.write_text("Hello world.")
    out = tmp / "out.wav"
    rc = render.main([
        "--script", str(script),
        "--out", str(out),
        "--voice", "excited",
        "--voices-path", str(voices),
    ])
    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr().out
    assert "out.wav" in captured


def test_cli_reads_script_from_stdin(cli_setup, monkeypatch, capsys):
    tts, stt, voices, tmp = cli_setup
    out = tmp / "out.wav"
    monkeypatch.setattr("sys.stdin", _StringIO("Hi from stdin."))
    rc = render.main([
        "--script", "-",
        "--out", str(out),
        "--voice", "excited",
        "--voices-path", str(voices),
    ])
    assert rc == 0
    assert tts.calls[0]["text"] == "Hi from stdin."


def test_cli_rejects_voice_and_ref_audio(cli_setup, tmp_path):
    tts, stt, voices, tmp = cli_setup
    ref = tmp / "r.wav"
    ref.write_bytes(b"")
    script = tmp / "s.txt"
    script.write_text("hi")
    out = tmp / "o.wav"
    rc = render.main([
        "--script", str(script), "--out", str(out),
        "--voice", "excited", "--ref-audio", str(ref),
        "--voices-path", str(voices),
    ])
    assert rc != 0


def test_cli_uses_preset_defaults(cli_setup, tmp_path):
    import json
    tts, stt, voices, tmp = cli_setup
    presets = tmp / "presets.json"
    presets.write_text(json.dumps({"version": 1, "presets": {
        "hype": {"voice": "excited", "temperature": 0.9, "speed": 1.2,
                 "tag_density": "high"}
    }}))
    script = tmp / "s.txt"
    script.write_text("hi")
    out = tmp / "o.wav"
    rc = render.main([
        "--script", str(script), "--out", str(out),
        "--preset", "hype",
        "--voices-path", str(voices),
        "--presets-path", str(presets),
    ])
    assert rc == 0
    call = tts.calls[0]
    assert call["temperature"] == pytest.approx(0.9)
    assert call["speed"] == pytest.approx(1.2)


def test_cli_explicit_flags_override_preset(cli_setup, tmp_path):
    import json
    tts, stt, voices, tmp = cli_setup
    presets = tmp / "presets.json"
    presets.write_text(json.dumps({"version": 1, "presets": {
        "hype": {"voice": "excited", "temperature": 0.9}
    }}))
    script = tmp / "s.txt"
    script.write_text("hi")
    out = tmp / "o.wav"
    rc = render.main([
        "--script", str(script), "--out", str(out),
        "--preset", "hype", "--temperature", "0.3",
        "--voices-path", str(voices), "--presets-path", str(presets),
    ])
    assert rc == 0
    assert tts.calls[0]["temperature"] == pytest.approx(0.3)


class _StringIO:
    def __init__(self, s): self.s = s
    def read(self): return self.s
