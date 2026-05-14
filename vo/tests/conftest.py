# vo/tests/conftest.py
"""Shared pytest fixtures."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_voices_path(tmp_path: Path) -> Path:
    """Empty voices.json for write tests."""
    p = tmp_path / "voices.json"
    p.write_text(json.dumps({"version": 1, "voices": {}}))
    return p


@pytest.fixture
def tmp_presets_path(tmp_path: Path) -> Path:
    p = tmp_path / "presets.json"
    p.write_text(json.dumps({"version": 1, "presets": {}}))
    return p


@pytest.fixture
def sample_voices_path(tmp_path: Path) -> Path:
    """Pre-populated voices.json with one entry."""
    p = tmp_path / "voices.json"
    p.write_text(json.dumps({
        "version": 1,
        "default": "excited",
        "voices": {
            "excited": {
                "label": "Excited",
                "audio": "voice samples/Excited VO clone sample.m4a",
                "transcript": "Test transcript.",
                "notes": "test"
            }
        }
    }))
    return p


# --- mocked Whisper ----------------------------------------------------

class _FakeSegment:
    def __init__(self, text, words):
        self.text = text
        self.start = words[0]["start"] if words else 0.0
        self.end = words[-1]["end"] if words else 0.0
        from types import SimpleNamespace
        self.words = [SimpleNamespace(**w) for w in words]


class _FakeWhisperResult:
    def __init__(self, segments):
        self.segments = segments
    @property
    def text(self):
        return " ".join(s.text for s in self.segments)


class FakeWhisper:
    """Minimal stand-in for mlx_audio.stt models."""
    def __init__(self, transcript="hello world", words=None):
        self.transcript = transcript
        self.calls = []
        self._words = words or [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 0.5, "end": 1.0, "word": "world"},
        ]
    def generate(self, audio, **kw):
        self.calls.append((audio, kw))
        return _FakeWhisperResult([_FakeSegment(self.transcript, self._words)])


# --- mocked Fish Speech ------------------------------------------------

class _FakeGenResult:
    def __init__(self, audio):
        self.audio = audio


class FakeFishModel:
    """Stand-in for the loaded Fish Speech model in tests."""
    def __init__(self, sample_rate=44100, audio_seconds=2.0, words=None):
        self.sample_rate = sample_rate
        self.audio_seconds = audio_seconds
        self.calls = []
        # Predetermined Whisper output for the quality gate
        self.words = words or [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 0.5, "end": 1.0, "word": "world"},
        ]
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        import mlx.core as mx
        n = int(self.sample_rate * self.audio_seconds)
        audio = mx.zeros((n,))
        return iter([_FakeGenResult(audio)])
