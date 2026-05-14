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


# ---------- voices.local.json overlay ----------------------------------

def test_overlay_fills_in_missing_transcript(tmp_path):
    base = tmp_path / "voices.json"
    base.write_text(json.dumps({"version": 1, "voices": {
        "excited": {"label": "Excited", "audio": "a.m4a",
                    "transcript": "<placeholder>"}
    }}))
    overlay = tmp_path / "voices.local.json"
    overlay.write_text(json.dumps({"voices": {
        "excited": {"transcript": "real spoken sentence"}
    }}))
    voices = load_voices(base)
    assert voices["excited"].transcript == "real spoken sentence"
    # Non-overridden fields stay from base.
    assert voices["excited"].label == "Excited"
    assert voices["excited"].audio == "a.m4a"


def test_overlay_can_add_new_voices(tmp_path):
    base = tmp_path / "voices.json"
    base.write_text(json.dumps({"version": 1, "voices": {}}))
    overlay = tmp_path / "voices.local.json"
    overlay.write_text(json.dumps({"voices": {
        "private": {"label": "P", "audio": "p.m4a", "transcript": "x"}
    }}))
    voices = load_voices(base)
    assert "private" in voices
    assert voices["private"].label == "P"


def test_overlay_absent_is_no_op(sample_voices_path):
    # No overlay file exists; load_voices behaves exactly like before.
    voices = load_voices(sample_voices_path)
    assert voices["excited"].transcript == "Test transcript."


def test_overlay_malformed_voices_field_raises(tmp_path):
    base = tmp_path / "voices.json"
    base.write_text(json.dumps({"version": 1, "voices": {}}))
    overlay = tmp_path / "voices.local.json"
    overlay.write_text(json.dumps({"voices": "not a dict"}))
    with pytest.raises(RegistryError, match="overlay.*must be an object"):
        load_voices(base)


# ---------- presets.json -----------------------------------------------

@pytest.fixture
def sample_presets_path(tmp_path):
    p = tmp_path / "presets.json"
    p.write_text(json.dumps({
        "version": 1,
        "presets": {
            "fb-reel-hype": {
                "voice": "excited",
                "tag_hints": ["[emphasis]", "[excited]"],
                "tag_density": "high",
                "temperature": 0.75,
                "speed": 1.05,
                "notes": "Fast-paced FB/IG narration."
            }
        }
    }))
    return p


def test_load_presets_returns_preset_objects(sample_presets_path):
    presets = load_presets(sample_presets_path)
    assert "fb-reel-hype" in presets
    p = presets["fb-reel-hype"]
    assert isinstance(p, Preset)
    assert p.voice == "excited"
    assert p.tag_density == "high"
    assert p.temperature == 0.75
    assert p.speed == 1.05


def test_load_presets_applies_defaults_for_missing_fields(tmp_path):
    p = tmp_path / "presets.json"
    p.write_text(json.dumps({"version": 1, "presets": {
        "minimal": {"notes": "just a stub"}
    }}))
    presets = load_presets(p)
    m = presets["minimal"]
    assert m.voice is None
    assert m.tag_density == "medium"
    assert m.temperature == 0.7
    assert m.speed == 1.0


def test_load_presets_tolerates_unknown_fields(tmp_path):
    p = tmp_path / "presets.json"
    p.write_text(json.dumps({"version": 1, "presets": {
        "x": {"voice": "v", "future": "ignored"}
    }}))
    presets = load_presets(p)
    assert presets["x"].extra["future"] == "ignored"


def test_get_preset_unknown_lists_available(sample_presets_path):
    with pytest.raises(RegistryError, match=r"unknown preset 'nope'.*available: fb-reel-hype"):
        get_preset("nope", sample_presets_path)


def test_add_preset_writes_and_preserves(sample_presets_path):
    p = Preset(name="new", voice="excited", tag_density="low")
    add_preset(p, sample_presets_path)
    presets = load_presets(sample_presets_path)
    assert set(presets.keys()) == {"fb-reel-hype", "new"}
    assert presets["new"].tag_density == "low"


def test_load_presets_top_level_array_raises(tmp_path):
    p = tmp_path / "presets.json"
    p.write_text("[]")
    with pytest.raises(RegistryError, match="expected an object"):
        load_presets(p)


def test_load_presets_presets_field_not_object_raises(tmp_path):
    p = tmp_path / "presets.json"
    p.write_text('{"version": 1, "presets": "not a dict"}')
    with pytest.raises(RegistryError, match="'presets' must be an object"):
        load_presets(p)
