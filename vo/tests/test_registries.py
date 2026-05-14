"""Tests for vo.registries — voices.json and presets.json I/O."""
import json
from pathlib import Path

import pytest

from vo.registries import (
    Voice, Preset, RegistryError,
    load_voices, get_voice, add_voice,
    load_presets, get_preset, add_preset,
)


# ---------- voices.json -------------------------------------------------

def test_load_voices_returns_dict_of_voice_objects(sample_voices_path):
    voices = load_voices(sample_voices_path)
    assert "excited" in voices
    v = voices["excited"]
    assert isinstance(v, Voice)
    assert v.id == "excited"
    assert v.label == "Excited"
    assert v.transcript == "Test transcript."


def test_load_voices_tolerates_unknown_fields(tmp_path):
    p = tmp_path / "voices.json"
    p.write_text(json.dumps({
        "version": 1,
        "voices": {
            "v": {
                "label": "x", "audio": "a.wav", "transcript": "t",
                "future_field": "ignored", "created_at": "2026-05-14"
            }
        }
    }))
    voices = load_voices(p)
    assert voices["v"].label == "x"
    assert voices["v"].extra["future_field"] == "ignored"


def test_load_voices_missing_required_field_raises(tmp_path):
    p = tmp_path / "voices.json"
    p.write_text(json.dumps({"version": 1, "voices": {
        "bad": {"label": "x"}  # missing audio + transcript
    }}))
    with pytest.raises(RegistryError, match="missing required field"):
        load_voices(p)


def test_load_voices_malformed_json_raises(tmp_path):
    p = tmp_path / "voices.json"
    p.write_text("not json {")
    with pytest.raises(RegistryError, match="invalid JSON"):
        load_voices(p)


def test_load_voices_top_level_array_raises(tmp_path):
    p = tmp_path / "voices.json"
    p.write_text("[]")
    with pytest.raises(RegistryError, match="expected an object"):
        load_voices(p)


def test_load_voices_voices_field_not_object_raises(tmp_path):
    p = tmp_path / "voices.json"
    p.write_text('{"version": 1, "voices": "not a dict"}')
    with pytest.raises(RegistryError, match="'voices' must be an object"):
        load_voices(p)


def test_load_voices_missing_file_raises(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_voices(tmp_path / "nope.json")


def test_get_voice_returns_named_voice(sample_voices_path):
    v = get_voice("excited", sample_voices_path)
    assert v.id == "excited"


def test_get_voice_unknown_id_lists_available(sample_voices_path):
    with pytest.raises(RegistryError, match=r"unknown voice 'nope'.*available: excited"):
        get_voice("nope", sample_voices_path)


# ---------- add_voice --------------------------------------------------

def test_add_voice_writes_to_disk(tmp_voices_path):
    v = Voice(id="new", label="New", audio="a.wav", transcript="t")
    add_voice(v, tmp_voices_path)
    voices = load_voices(tmp_voices_path)
    assert "new" in voices
    assert voices["new"].label == "New"


def test_add_voice_preserves_existing_entries(sample_voices_path):
    v = Voice(id="new", label="N", audio="a.wav", transcript="t")
    add_voice(v, sample_voices_path)
    voices = load_voices(sample_voices_path)
    assert set(voices.keys()) == {"excited", "new"}


def test_add_voice_overwrites_same_id(sample_voices_path):
    v = Voice(id="excited", label="Updated", audio="x.wav", transcript="t2")
    add_voice(v, sample_voices_path)
    assert load_voices(sample_voices_path)["excited"].label == "Updated"


def test_add_voice_writes_extra_fields(tmp_voices_path):
    v = Voice(id="v", label="x", audio="a.wav", transcript="t",
              extra={"created_at": "2026-05-14"})
    add_voice(v, tmp_voices_path)
    raw = json.loads(tmp_voices_path.read_text())
    assert raw["voices"]["v"]["created_at"] == "2026-05-14"
