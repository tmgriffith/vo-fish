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
