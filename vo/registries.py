"""voices.json + presets.json registries."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Voice", "Preset", "RegistryError",
    "load_voices", "get_voice", "add_voice",
    "load_presets", "get_preset", "add_preset",
    "DEFAULT_VOICES_PATH", "DEFAULT_PRESETS_PATH",
]

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_VOICES_PATH = PKG_DIR / "voices.json"
DEFAULT_PRESETS_PATH = PKG_DIR / "presets.json"
PROJECT_ROOT = PKG_DIR.parent  # paths inside voices.json resolve from here


class RegistryError(Exception):
    """Anything wrong with a registry file."""


@dataclass
class Voice:
    id: str
    label: str
    audio: str          # path string, resolved relative to PROJECT_ROOT
    transcript: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def audio_path(self) -> Path:
        p = Path(self.audio)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


@dataclass
class Preset:
    name: str
    voice: str | None = None
    tag_hints: list[str] = field(default_factory=list)
    tag_density: str = "medium"
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 30
    speed: float = 1.0
    language: str = "en"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_REQUIRED_VOICE_FIELDS = ("label", "audio", "transcript")


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise RegistryError(f"registry not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RegistryError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise RegistryError(
            f"invalid JSON in {path}: expected an object, got {type(data).__name__}"
        )
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_voices(path: Path = DEFAULT_VOICES_PATH) -> dict[str, Voice]:
    raw = _read_json(path)
    voices_raw = raw.get("voices", {})
    if not isinstance(voices_raw, dict):
        raise RegistryError(
            f"invalid voices.json at {path}: 'voices' must be an object"
        )
    out: dict[str, Voice] = {}
    for vid, data in voices_raw.items():
        missing = [f for f in _REQUIRED_VOICE_FIELDS if f not in data]
        if missing:
            raise RegistryError(
                f"voice {vid!r} missing required field(s): {', '.join(missing)}"
            )
        extra = {k: v for k, v in data.items()
                 if k not in _REQUIRED_VOICE_FIELDS and k != "notes"}
        out[vid] = Voice(
            id=vid,
            label=data["label"],
            audio=data["audio"],
            transcript=data["transcript"],
            notes=data.get("notes", ""),
            extra=extra,
        )
    return out


def get_voice(voice_id: str, path: Path = DEFAULT_VOICES_PATH) -> Voice:
    voices = load_voices(path)
    if voice_id not in voices:
        available = ", ".join(sorted(voices)) or "(none)"
        raise RegistryError(
            f"unknown voice {voice_id!r} — available: {available}"
        )
    return voices[voice_id]


def add_voice(voice: Voice, path: Path = DEFAULT_VOICES_PATH) -> None:
    """Insert or overwrite a voice entry. Creates the file if missing."""
    if path.exists():
        data = _read_json(path)
    else:
        data = {"version": 1, "voices": {}}
    data.setdefault("voices", {})
    entry: dict[str, Any] = {
        "label": voice.label,
        "audio": voice.audio,
        "transcript": voice.transcript,
    }
    if voice.notes:
        entry["notes"] = voice.notes
    for k, v in voice.extra.items():
        if k not in entry:
            entry[k] = v
    data["voices"][voice.id] = entry
    _write_json(path, data)


_PRESET_DEFAULTS = {
    "voice": None,
    "tag_hints": [],
    "tag_density": "medium",
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 30,
    "speed": 1.0,
    "language": "en",
    "notes": "",
}
_PRESET_KNOWN = set(_PRESET_DEFAULTS) | {"name"}


def load_presets(path: Path = DEFAULT_PRESETS_PATH) -> dict[str, Preset]:
    raw = _read_json(path)
    presets_raw = raw.get("presets", {})
    out: dict[str, Preset] = {}
    for name, data in presets_raw.items():
        merged = {**_PRESET_DEFAULTS, **{k: v for k, v in data.items() if k in _PRESET_DEFAULTS}}
        extra = {k: v for k, v in data.items() if k not in _PRESET_KNOWN}
        out[name] = Preset(
            name=name,
            voice=merged["voice"],
            tag_hints=list(merged["tag_hints"]),
            tag_density=merged["tag_density"],
            temperature=float(merged["temperature"]),
            top_p=float(merged["top_p"]),
            top_k=int(merged["top_k"]),
            speed=float(merged["speed"]),
            language=merged["language"],
            notes=merged["notes"],
            extra=extra,
        )
    return out


def get_preset(name: str, path: Path = DEFAULT_PRESETS_PATH) -> Preset:
    presets = load_presets(path)
    if name not in presets:
        available = ", ".join(sorted(presets)) or "(none)"
        raise RegistryError(f"unknown preset {name!r} — available: {available}")
    return presets[name]


def add_preset(preset: Preset, path: Path = DEFAULT_PRESETS_PATH) -> None:
    if path.exists():
        data = _read_json(path)
    else:
        data = {"version": 1, "presets": {}}
    data.setdefault("presets", {})
    entry: dict[str, Any] = {
        "voice": preset.voice,
        "tag_hints": list(preset.tag_hints),
        "tag_density": preset.tag_density,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "speed": preset.speed,
        "language": preset.language,
    }
    if preset.notes:
        entry["notes"] = preset.notes
    for k, v in preset.extra.items():
        if k not in entry:
            entry[k] = v
    data["presets"][preset.name] = entry
    _write_json(path, data)
