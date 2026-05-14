"""Tests for vo.transcribe — Whisper helper."""
from pathlib import Path

import pytest

from vo import transcribe as t
from vo.tests.conftest import FakeWhisper


def test_transcribe_returns_text(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    fake = FakeWhisper(transcript="just a test")
    monkeypatch.setattr(t, "_get_stt", lambda: fake)
    assert t.transcribe(audio).strip() == "just a test"
    assert fake.calls[0][0] == str(audio)


def test_transcribe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        t.transcribe(tmp_path / "nope.wav")
