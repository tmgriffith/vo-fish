# VO Fish Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `vo-fish` Claude Code skill plus a reusable Python renderer at `vo/` that wraps Fish Audio S2 Pro for expressive content-creator voiceovers, with flexible voice cloning, optional presets, runtime registry growth, and a Whisper-based quality gate.

**Architecture:** Two-layer split. The skill markdown is the creative brain (picks voice, injects emotion tags, calls renderer). The Python renderer at `vo/` does pure inference, registry I/O, and quality checks — split into focused modules (`registries.py`, `tags.py`, `quality.py`, `render.py`) so each unit is small enough to test in isolation. Everything runs against the existing `.venv-mlx` pinned to mlx-audio 0.4.2.

**Tech Stack:** Python 3.13 (in `.venv-mlx`), mlx-audio 0.4.2, Fish Audio S2 Pro bf16, Whisper Large v3 Turbo for alignment, pytest for tests, stdlib argparse for CLI. Project is not a git repo yet — Task 1 initializes git so subsequent commits work.

**Working directory for all commands:** `/Users/tim/business/Content Machine/Content Machine/`

---

## File Structure

```
Content Machine/
├── .claude/
│   └── skills/
│       └── vo-fish/
│           └── SKILL.md             # creative recipe Claude follows
├── vo/
│   ├── __init__.py                  # exports the public API
│   ├── _version_check.py            # asserts mlx_audio==0.4.2 at import time
│   ├── registries.py                # voices.json + presets.json I/O
│   ├── tags.py                      # tag_mode handling
│   ├── quality.py                   # Whisper word extraction, silence gap, anchors, evaluator
│   ├── transcribe.py                # Whisper helper for auto-transcribing refs
│   ├── render.py                    # render() function + CLI entry point + model loading
│   ├── voices.json                  # seeded registry
│   ├── presets.json                 # seeded presets
│   ├── README.md                    # human-readable notes
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py              # shared fixtures (temp registries, mock model)
│       ├── test_registries.py
│       ├── test_tags.py
│       ├── test_quality.py
│       ├── test_cli.py
│       └── test_smoke.py            # opt-in end-to-end (skipped by default)
└── .venv-mlx/                       # existing
```

**Decomposition rationale:**
- `registries.py` is pure JSON I/O — easy unit tests.
- `tags.py` is pure string handling — easy unit tests.
- `quality.py` is pure data processing on Whisper output — easy unit tests with canned word lists.
- `transcribe.py` thin wrapper around Whisper — mocked in tests.
- `render.py` is the only module that touches MLX. It composes everything else. Tests use a mocked model.

---

## Task 1: Project scaffolding + git init + version pin

**Files:**
- Create: `vo/__init__.py`
- Create: `vo/_version_check.py`
- Create: `vo/README.md`
- Create: `vo/tests/__init__.py`
- Create: `vo/tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git + install pytest**

```bash
git init
git add docs/
git commit -m "docs: vo-fish design spec and implementation plan"
.venv-mlx/bin/pip install pytest
```

- [ ] **Step 2: Create `.gitignore`**

Write `.gitignore`:

```
.venv-mlx/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
*.tagged.txt
*.words.json
/tmp/
```

- [ ] **Step 3: Write `vo/_version_check.py`**

```python
"""Hard-fail import if mlx-audio is not pinned to 0.4.2.

mlx-audio 0.4.3 ships a Fish Speech regression that produces degenerative
token loops. 0.4.2 is the known-good version.
"""
import mlx_audio

REQUIRED_VERSION = "0.4.2"
_actual = getattr(mlx_audio, "__version__", None)
if _actual is None:
    try:
        from importlib.metadata import version as _v
        _actual = _v("mlx-audio")
    except Exception:
        _actual = "unknown"

if _actual != REQUIRED_VERSION:
    raise ImportError(
        f"vo/ requires mlx-audio=={REQUIRED_VERSION} (found {_actual}). "
        f"Run: .venv-mlx/bin/pip install 'mlx-audio=={REQUIRED_VERSION}'"
    )
```

- [ ] **Step 4: Write `vo/__init__.py`**

```python
"""vo - Fish Audio S2 Pro voiceover renderer for Content Machine."""
from vo import _version_check  # noqa: F401  (import-time version gate)

__all__ = ["render", "RenderResult"]


def __getattr__(name: str):
    # Lazy import so just importing the package doesn't load the model.
    if name in ("render", "RenderResult"):
        from vo.render import render, RenderResult
        return {"render": render, "RenderResult": RenderResult}[name]
    raise AttributeError(name)
```

- [ ] **Step 5: Write `vo/tests/__init__.py` (empty) and `vo/tests/conftest.py`**

```python
# vo/tests/__init__.py
```

```python
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
```

- [ ] **Step 6: Write `vo/README.md`**

```markdown
# vo — Fish Audio S2 Pro renderer

Local voiceover engine for Content Machine. Wraps Fish Audio S2 Pro (via
mlx-audio 0.4.2) with voice cloning, inline emotion tags, and a Whisper
quality gate.

## Quick use

    .venv-mlx/bin/python vo/render.py \
        --script script.txt \
        --voice excited \
        --out out/vo.wav

Outputs `out/vo.wav`, `out/vo.words.json`, `out/vo.tagged.txt`.

## Voices

`vo/voices.json` is the registry. Add a new voice on the fly:

    .venv-mlx/bin/python vo/render.py \
        --ref-audio path/to/sample.m4a \
        --script script.txt \
        --out out/vo.wav \
        --save-voice my_voice --label "My new voice"

If `--ref-text` is omitted, the renderer Whisper-transcribes the reference
automatically.

## Why the venv

mlx-audio 0.4.3 breaks Fish Speech (degenerative token loops). The renderer
pins to 0.4.2 and refuses to import otherwise. See
`docs/superpowers/specs/2026-05-14-vo-fish-skill-design.md`.
```

- [ ] **Step 7: Verify the import gate works**

```bash
.venv-mlx/bin/python -c "import vo; print('import ok')"
```

Expected: `import ok` (mlx-audio is pinned to 0.4.2 in `.venv-mlx`).

- [ ] **Step 8: Commit**

```bash
git add .gitignore vo/__init__.py vo/_version_check.py vo/README.md vo/tests/__init__.py vo/tests/conftest.py
git commit -m "feat(vo): scaffold package with version pin and test conftest"
```

---

## Task 2: Voice registry — dataclass + load

**Files:**
- Create: `vo/registries.py`
- Create: `vo/tests/test_registries.py`

- [ ] **Step 1: Write failing tests in `vo/tests/test_registries.py`**

```python
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


def test_load_voices_missing_file_raises(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_voices(tmp_path / "nope.json")


def test_get_voice_returns_named_voice(sample_voices_path):
    v = get_voice("excited", sample_voices_path)
    assert v.id == "excited"


def test_get_voice_unknown_id_lists_available(sample_voices_path):
    with pytest.raises(RegistryError, match=r"unknown voice 'nope'.*available: excited"):
        get_voice("nope", sample_voices_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_registries.py -v
```

Expected: ImportError / failures — `vo.registries` doesn't exist yet.

- [ ] **Step 3: Write `vo/registries.py` with `Voice`, `RegistryError`, `load_voices`, `get_voice`**

```python
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
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RegistryError(f"invalid JSON in {path}: {e}") from e


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_voices(path: Path = DEFAULT_VOICES_PATH) -> dict[str, Voice]:
    raw = _read_json(path)
    voices_raw = raw.get("voices", {})
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
    """Stub — implemented in next task."""
    raise NotImplementedError


def load_presets(path: Path = DEFAULT_PRESETS_PATH) -> dict[str, Preset]:
    """Stub — implemented in next task."""
    raise NotImplementedError


def get_preset(name: str, path: Path = DEFAULT_PRESETS_PATH) -> Preset:
    raise NotImplementedError


def add_preset(preset: Preset, path: Path = DEFAULT_PRESETS_PATH) -> None:
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_registries.py -v -k "voices"
```

Expected: all `voices`-related tests PASS. The remaining preset tests will fail but we'll add them next.

- [ ] **Step 5: Commit**

```bash
git add vo/registries.py vo/tests/test_registries.py
git commit -m "feat(vo): add Voice dataclass and voices.json load/get"
```

---

## Task 3: Voice registry — add_voice (write path)

**Files:**
- Modify: `vo/registries.py` (replace `add_voice` stub)
- Modify: `vo/tests/test_registries.py` (append tests)

- [ ] **Step 1: Append failing tests to `vo/tests/test_registries.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_registries.py -v -k "add_voice"
```

Expected: 4 failures (`NotImplementedError`).

- [ ] **Step 3: Replace the `add_voice` stub in `vo/registries.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_registries.py -v -k "add_voice"
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/registries.py vo/tests/test_registries.py
git commit -m "feat(vo): add_voice writes to voices.json preserving existing entries"
```

---

## Task 4: Preset registry — load + get + add

**Files:**
- Modify: `vo/registries.py` (replace preset stubs)
- Modify: `vo/tests/test_registries.py` (append preset tests)

- [ ] **Step 1: Append failing tests to `vo/tests/test_registries.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_registries.py -v -k "preset"
```

Expected: 5 failures.

- [ ] **Step 3: Replace preset stubs in `vo/registries.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_registries.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/registries.py vo/tests/test_registries.py
git commit -m "feat(vo): presets.json load/get/add with field defaults"
```

---

## Task 5: Tag mode handling

**Files:**
- Create: `vo/tags.py`
- Create: `vo/tests/test_tags.py`

- [ ] **Step 1: Write failing tests in `vo/tests/test_tags.py`**

```python
"""Tests for vo.tags — tag_mode handling."""
import pytest

from vo.tags import apply_tag_mode, TagModeError


def test_auto_passes_through_unchanged():
    s = "Hello [excited] world."
    assert apply_tag_mode(s, "auto") == s


def test_explicit_passes_through_unchanged():
    s = "Hello [excited] world. [pause] More."
    assert apply_tag_mode(s, "explicit") == s


def test_none_strips_bracketed_tags():
    s = "Hello [excited] world. [short pause] More."
    assert apply_tag_mode(s, "none") == "Hello world. More."


def test_none_collapses_extra_whitespace():
    s = "Hello   [tag1]   [tag2]   world."
    assert apply_tag_mode(s, "none") == "Hello world."


def test_none_preserves_speaker_tokens():
    """`<|speaker:0|>` tokens are not square-bracket tags; keep them."""
    s = "<|speaker:0|>Hi [excited] there."
    assert apply_tag_mode(s, "none") == "<|speaker:0|>Hi there."


def test_unknown_mode_raises():
    with pytest.raises(TagModeError, match="unknown tag_mode 'wat'"):
        apply_tag_mode("hi", "wat")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_tags.py -v
```

Expected: ImportError — `vo.tags` doesn't exist.

- [ ] **Step 3: Write `vo/tags.py`**

```python
"""Tag-mode handling for the renderer.

- auto:     no-op (script already has tags from Claude / caller)
- explicit: no-op (script has tags, leave them alone)
- none:     strip all [bracketed] tags
"""
from __future__ import annotations

import re

__all__ = ["apply_tag_mode", "TagModeError", "VALID_TAG_MODES"]

VALID_TAG_MODES = ("auto", "explicit", "none")

# Matches [...] not preceded by < and not containing < or > so we don't eat
# <|speaker:i|> speaker tokens. Greedy-free.
_TAG_RE = re.compile(r"\[[^\[\]<>]*\]")
_WS_RE = re.compile(r"[ \t]+")


class TagModeError(ValueError):
    """Unknown tag_mode value."""


def apply_tag_mode(script: str, mode: str) -> str:
    if mode not in VALID_TAG_MODES:
        raise TagModeError(
            f"unknown tag_mode {mode!r} — valid: {', '.join(VALID_TAG_MODES)}"
        )
    if mode in ("auto", "explicit"):
        return script
    # mode == "none": strip bracketed tags, collapse whitespace per line
    stripped = _TAG_RE.sub("", script)
    out_lines = []
    for line in stripped.splitlines():
        out_lines.append(_WS_RE.sub(" ", line).strip())
    return "\n".join(l for l in out_lines if l)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_tags.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/tags.py vo/tests/test_tags.py
git commit -m "feat(vo): tag_mode handling (auto/explicit/none)"
```

---

## Task 6: Quality gate — Word + extract_words + largest_word_gap

**Files:**
- Create: `vo/quality.py`
- Create: `vo/tests/test_quality.py`

- [ ] **Step 1: Write failing tests in `vo/tests/test_quality.py`**

```python
"""Tests for vo.quality — Whisper word extraction and silence gap."""
from types import SimpleNamespace

import pytest

from vo.quality import Word, extract_words, largest_word_gap


def _fake_stt_result(words_data, segments_data=None):
    """Build a Whisper-result-shaped object from raw word lists."""
    segs = []
    if segments_data is not None:
        segs = [SimpleNamespace(**s) for s in segments_data]
    else:
        segs = [SimpleNamespace(
            start=words_data[0]["start"] if words_data else 0.0,
            end=words_data[-1]["end"] if words_data else 0.0,
            text=" ".join(w["word"] for w in words_data),
            words=[SimpleNamespace(**w) for w in words_data],
        )]
    return SimpleNamespace(segments=segs)


def test_extract_words_normalizes_text():
    r = _fake_stt_result([
        {"start": 0.0, "end": 0.5, "word": "Hello,"},
        {"start": 0.5, "end": 1.0, "word": " WORLD"},
    ])
    words = extract_words(r)
    assert len(words) == 2
    assert words[0].text == "hello"
    assert words[1].text == "world"


def test_extract_words_skips_pure_punctuation():
    r = _fake_stt_result([
        {"start": 0.0, "end": 0.5, "word": "Hi"},
        {"start": 0.5, "end": 0.6, "word": ","},
        {"start": 0.6, "end": 1.0, "word": "there"},
    ])
    words = extract_words(r)
    assert [w.text for w in words] == ["hi", "there"]


def test_extract_words_handles_dict_shapes():
    r = {"segments": [{
        "start": 0.0, "end": 1.0, "text": "hi",
        "words": [{"start": 0.0, "end": 0.5, "word": "hi"}],
    }]}
    words = extract_words(r)
    assert words[0] == Word(start=0.0, end=0.5, text="hi")


def test_extract_words_empty_result_returns_empty():
    r = SimpleNamespace(segments=[])
    assert extract_words(r) == []


def test_largest_word_gap_normal_speech():
    words = [
        Word(0.0, 0.5, "a"),
        Word(0.6, 1.0, "b"),
        Word(1.1, 1.5, "c"),
    ]
    assert largest_word_gap(words) == pytest.approx(0.1)


def test_largest_word_gap_finds_collapse():
    words = [
        Word(0.0, 1.0, "a"),
        Word(5.0, 6.0, "b"),  # 4s gap
        Word(6.1, 7.0, "c"),
    ]
    assert largest_word_gap(words) == pytest.approx(4.0)


def test_largest_word_gap_single_word_is_inf():
    assert largest_word_gap([Word(0.0, 1.0, "a")]) == float("inf")


def test_largest_word_gap_empty_is_inf():
    assert largest_word_gap([]) == float("inf")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_quality.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `vo/quality.py` (partial — Word + extract_words + largest_word_gap)**

```python
"""Quality gate utilities — Whisper word extraction, silence gaps, anchor matching."""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Word", "extract_words", "largest_word_gap",
    "find_anchor_starts", "QualityCheck", "evaluate",
]

_WORD_CHAR_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str  # already normalized (lower, alnum + ')


def _normalize(s: str) -> str:
    return "".join(_WORD_CHAR_RE.findall((s or "").lower()))


def _segments(stt_result) -> list:
    """Pull a list of segments out of a result that may be a dict or object."""
    if isinstance(stt_result, dict):
        return stt_result.get("segments", []) or []
    segs = getattr(stt_result, "segments", None)
    if segs is None and hasattr(stt_result, "__dict__"):
        segs = stt_result.__dict__.get("segments", [])
    return segs or []


def _seg_words(seg) -> list:
    if isinstance(seg, dict):
        return seg.get("words", []) or []
    w = getattr(seg, "words", None)
    if w is None and hasattr(seg, "__dict__"):
        w = seg.__dict__.get("words", [])
    return w or []


def extract_words(stt_result) -> list[Word]:
    """Flatten Whisper output into a list of normalized Word triples."""
    out: list[Word] = []
    for seg in _segments(stt_result):
        for w in _seg_words(seg):
            data = w if isinstance(w, dict) else w.__dict__
            ws = data.get("start")
            we = data.get("end")
            raw = data.get("word") or data.get("text") or ""
            text = _normalize(raw)
            if ws is not None and we is not None and text:
                out.append(Word(float(ws), float(we), text))
    return out


def largest_word_gap(words: list[Word]) -> float:
    """Largest inter-word silence in seconds. Returns +inf for <2 words."""
    if len(words) < 2:
        return float("inf")
    return max(words[i].start - words[i - 1].end for i in range(1, len(words)))


def find_anchor_starts(words, anchors):
    """Placeholder — implemented in next task."""
    raise NotImplementedError


@dataclass
class QualityCheck:
    passed: bool
    max_gap: float
    anchor_starts: list[float] | None
    reason: str = ""


def evaluate(words, max_silence_gap, anchors=None):
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_quality.py -v -k "extract_words or largest_word_gap"
```

Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/quality.py vo/tests/test_quality.py
git commit -m "feat(vo): Word/extract_words/largest_word_gap for quality gate"
```

---

## Task 7: Quality gate — anchor matching

**Files:**
- Modify: `vo/quality.py` (replace `find_anchor_starts` stub)
- Modify: `vo/tests/test_quality.py` (append anchor tests)

- [ ] **Step 1: Append failing tests to `vo/tests/test_quality.py`**

```python
# ---------- anchor matching --------------------------------------------

def _words(*pairs):
    """Helper: [(t_start, 'text'), ...] -> [Word(...)]."""
    out = []
    for i, (start, txt) in enumerate(pairs):
        out.append(Word(start, start + 0.4, txt))
    return out


def test_find_anchor_starts_locates_each_anchor():
    from vo.quality import find_anchor_starts
    words = _words(
        (0.0, "most"), (0.5, "content"), (1.0, "creators"), (1.5, "are"),
        (3.0, "that's"), (3.5, "not"), (4.0, "a"), (4.5, "storage"),
        (6.0, "you"), (6.5, "shot"), (7.0, "it"),
    )
    anchors = [
        ["most", "content", "creators"],
        ["that's", "not", "a", "storage"],
        ["you", "shot", "it"],
    ]
    starts = find_anchor_starts(words, anchors)
    assert starts == [0.0, 3.0, 6.0]


def test_find_anchor_starts_tolerates_extra_filler_word():
    from vo.quality import find_anchor_starts
    # "the fix really isn't" — "really" is an extra word
    words = _words(
        (0.0, "the"), (0.4, "fix"), (0.8, "really"), (1.2, "isn't"),
    )
    starts = find_anchor_starts(words, [["the", "fix", "isn't"]])
    assert starts == [0.0]


def test_find_anchor_starts_returns_none_if_none_match():
    from vo.quality import find_anchor_starts
    words = _words((0.0, "wat"), (0.5, "huh"))
    assert find_anchor_starts(words, [["nope", "nope"]]) is None


def test_find_anchor_starts_backfills_missing_with_neighbours():
    from vo.quality import find_anchor_starts
    # Middle anchor missing, neighbours present
    words = _words(
        (0.0, "alpha"), (1.0, "beta"),
        # gamma missing
        (3.0, "delta"),
    )
    anchors = [["alpha"], ["gamma"], ["delta"]]
    starts = find_anchor_starts(words, anchors)
    assert starts is not None
    assert starts[0] == 0.0
    assert starts[2] == 3.0
    assert 0.0 < starts[1] < 3.0


def test_find_anchor_starts_enforces_monotonic_increase():
    from vo.quality import find_anchor_starts
    # Second anchor matches a word that occurs BEFORE the first anchor.
    # Implementation should refuse to go backwards.
    words = _words((0.0, "alpha"), (1.0, "beta"), (2.0, "alpha"))
    starts = find_anchor_starts(words, [["beta"], ["alpha"]])
    # second anchor must find the post-cursor "alpha" at 2.0
    assert starts == [1.0, 2.0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_quality.py -v -k "anchor"
```

Expected: 5 failures (`NotImplementedError`).

- [ ] **Step 3: Replace `find_anchor_starts` in `vo/quality.py`**

```python
def find_anchor_starts(words: list[Word], anchors: list[list[str]]) -> list[float] | None:
    """For each anchor (a list of consecutive content words), find the start
    time of its first word in `words`. Anchors are matched in order; the
    search cursor advances after each match so anchors stay monotonic.

    Up to 3 stray words are allowed between consecutive anchor words to absorb
    Whisper filler/punctuation glitches.

    Returns:
      list of floats (one per anchor). If a specific anchor can't be matched
      it gets backfilled by averaging its surrounding matched anchors. If no
      anchor matches at all, returns None.
    """
    norm = [(w.start, w.end, w.text) for w in words]
    starts: list[float | None] = []
    cursor = 0
    for anchor in anchors:
        if not anchor:
            starts.append(None)
            continue
        head = anchor[0]
        found = None
        i = cursor
        while i < len(norm):
            if norm[i][2] == head:
                # try to match the rest of the anchor within a small window
                j = i
                ok = True
                for tok in anchor[1:]:
                    advanced = False
                    for k in range(j + 1, min(j + 5, len(norm))):
                        if norm[k][2] == tok:
                            j = k
                            advanced = True
                            break
                    if not advanced:
                        ok = False
                        break
                if ok:
                    found = norm[i][0]
                    cursor = j + 1
                    break
            i += 1
        starts.append(found)

    if all(s is None for s in starts):
        return None

    last_end = norm[-1][1] if norm else 0.0
    for idx in range(len(starts)):
        if starts[idx] is None:
            prev = next((starts[k] for k in range(idx - 1, -1, -1) if starts[k] is not None), 0.0)
            nxt = next((starts[k] for k in range(idx + 1, len(starts)) if starts[k] is not None), last_end)
            starts[idx] = (prev + nxt) / 2

    # enforce monotonic increase
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            starts[i] = starts[i - 1] + 0.1
    return [float(s) for s in starts]  # type: ignore[arg-type]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_quality.py -v
```

Expected: all PASS (extract_words + largest_word_gap + anchor tests).

- [ ] **Step 5: Commit**

```bash
git add vo/quality.py vo/tests/test_quality.py
git commit -m "feat(vo): anchor-phrase matching with filler tolerance"
```

---

## Task 8: Quality evaluator (gap + anchors → pass/fail)

**Files:**
- Modify: `vo/quality.py` (replace `evaluate` stub)
- Modify: `vo/tests/test_quality.py` (append evaluator tests)

- [ ] **Step 1: Append failing tests**

```python
# ---------- evaluator --------------------------------------------------

def test_evaluate_passes_with_clean_words():
    from vo.quality import evaluate
    words = _words((0.0, "a"), (0.5, "b"), (1.0, "c"))
    result = evaluate(words, max_silence_gap=2.5, anchors=None)
    assert result.passed is True
    assert result.reason == ""


def test_evaluate_fails_on_silence_gap():
    from vo.quality import evaluate
    words = [Word(0.0, 1.0, "a"), Word(5.0, 6.0, "b")]
    result = evaluate(words, max_silence_gap=2.5, anchors=None)
    assert result.passed is False
    assert "silence gap" in result.reason
    assert result.max_gap == pytest.approx(4.0)


def test_evaluate_fails_when_anchor_missing():
    from vo.quality import evaluate
    words = _words((0.0, "a"), (0.5, "b"))
    result = evaluate(words, max_silence_gap=2.5, anchors=[["nope"]])
    assert result.passed is False
    assert "anchor" in result.reason


def test_evaluate_passes_with_anchors_found():
    from vo.quality import evaluate
    words = _words((0.0, "alpha"), (0.5, "beta"), (1.0, "gamma"))
    result = evaluate(words, max_silence_gap=2.5, anchors=[["alpha"], ["gamma"]])
    assert result.passed is True
    assert result.anchor_starts == [0.0, 1.0]


def test_evaluate_fails_on_empty_words():
    from vo.quality import evaluate
    result = evaluate([], max_silence_gap=2.5, anchors=None)
    assert result.passed is False
    assert "no words" in result.reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_quality.py -v -k "evaluate"
```

Expected: 5 failures.

- [ ] **Step 3: Replace `evaluate` in `vo/quality.py`**

```python
def evaluate(
    words: list[Word],
    max_silence_gap: float,
    anchors: list[list[str]] | None = None,
) -> QualityCheck:
    """Run the quality gate. Returns a QualityCheck."""
    if not words:
        return QualityCheck(passed=False, max_gap=float("inf"),
                            anchor_starts=None, reason="no words transcribed")

    gap = largest_word_gap(words)
    anchor_starts: list[float] | None = None
    reasons: list[str] = []

    if gap > max_silence_gap:
        reasons.append(
            f"silence gap {gap:.2f}s exceeds threshold {max_silence_gap:.2f}s"
        )

    if anchors:
        anchor_starts = find_anchor_starts(words, anchors)
        if anchor_starts is None:
            reasons.append("no anchor phrases located in transcript")

    passed = not reasons
    return QualityCheck(
        passed=passed,
        max_gap=gap,
        anchor_starts=anchor_starts,
        reason="; ".join(reasons),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_quality.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/quality.py vo/tests/test_quality.py
git commit -m "feat(vo): quality evaluator combines gap + anchor checks"
```

---

## Task 9: Whisper transcription helper

**Files:**
- Create: `vo/transcribe.py`
- Modify: `vo/tests/conftest.py` (add stt fixture)
- Create: `vo/tests/test_transcribe.py`

- [ ] **Step 1: Append to `vo/tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write failing tests in `vo/tests/test_transcribe.py`**

```python
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
```

- [ ] **Step 3: Write `vo/transcribe.py`**

```python
"""Whisper transcription helper for reference audio."""
from __future__ import annotations

from pathlib import Path

__all__ = ["transcribe", "STT_MODEL"]

STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"

_STT = None


def _get_stt():
    global _STT
    if _STT is None:
        from mlx_audio.stt import load
        _STT = load(STT_MODEL)
    return _STT


def transcribe(audio_path: Path | str) -> str:
    """Return the Whisper transcript of `audio_path`."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    stt = _get_stt()
    r = stt.generate(str(audio_path))
    return getattr(r, "text", None) or str(r)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_transcribe.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/transcribe.py vo/tests/test_transcribe.py vo/tests/conftest.py
git commit -m "feat(vo): transcribe() Whisper helper with mockable stt loader"
```

---

## Task 10: Voice resolution helper (registry vs ad-hoc ref_audio)

**Files:**
- Create: `vo/voice_resolver.py`
- Create: `vo/tests/test_voice_resolver.py`

- [ ] **Step 1: Write failing tests in `vo/tests/test_voice_resolver.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_voice_resolver.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `vo/voice_resolver.py`**

```python
"""Resolve --voice / --ref-audio into a concrete (audio path, transcript)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vo.registries import (
    DEFAULT_VOICES_PATH, load_voices, get_voice, Voice
)

__all__ = ["ResolvedVoice", "VoiceResolutionError", "resolve_voice"]


class VoiceResolutionError(Exception):
    pass


@dataclass
class ResolvedVoice:
    voice_id: str | None        # None for ad-hoc or model-default
    audio_path: Path | None     # None for model-default
    transcript: str | None      # None for model-default
    source: str                 # "registry" | "ad_hoc" | "model_default"


def resolve_voice(
    *,
    voice_id: str | None,
    ref_audio: Path | None,
    ref_text: str | None,
    voices_path: Path | None = DEFAULT_VOICES_PATH,
    transcribe_fn: Callable[[Path], str] | None = None,
) -> ResolvedVoice:
    if voice_id is not None and ref_audio is not None:
        raise VoiceResolutionError(
            "cannot supply both --voice and --ref-audio"
        )

    if voice_id is not None:
        if voices_path is None:
            raise VoiceResolutionError("voices_path required when voice_id is given")
        v = get_voice(voice_id, voices_path)
        return ResolvedVoice(voice_id=v.id, audio_path=v.audio_path,
                             transcript=v.transcript, source="registry")

    if ref_audio is not None:
        if not ref_audio.exists():
            raise VoiceResolutionError(f"ref_audio not found: {ref_audio}")
        if ref_text is None:
            if transcribe_fn is None:
                # Lazy import so tests can fully mock when desired.
                from vo.transcribe import transcribe as _t
                transcribe_fn = _t
            ref_text = transcribe_fn(ref_audio)
        return ResolvedVoice(voice_id=None, audio_path=ref_audio,
                             transcript=ref_text, source="ad_hoc")

    # Neither given — try registry default
    if voices_path is not None and voices_path.exists():
        raw = json.loads(voices_path.read_text())
        default_id = raw.get("default")
        if default_id:
            v = get_voice(default_id, voices_path)
            return ResolvedVoice(voice_id=v.id, audio_path=v.audio_path,
                                 transcript=v.transcript, source="registry")

    return ResolvedVoice(voice_id=None, audio_path=None,
                         transcript=None, source="model_default")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_voice_resolver.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/voice_resolver.py vo/tests/test_voice_resolver.py
git commit -m "feat(vo): voice resolver bridges registry and ad-hoc refs"
```

---

## Task 11: Renderer — single-call generation (happy path, mocked)

**Files:**
- Create: `vo/render.py`
- Modify: `vo/tests/conftest.py` (add fake TTS model)
- Create: `vo/tests/test_render.py`

- [ ] **Step 1: Append to `vo/tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write failing tests in `vo/tests/test_render.py`**

```python
"""Tests for vo.render — happy path with mocked model."""
import json
from pathlib import Path

import pytest

from vo.tests.conftest import FakeFishModel, FakeWhisper


@pytest.fixture
def fake_renderer(monkeypatch):
    from vo import render
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
```

- [ ] **Step 3: Write `vo/render.py`**

```python
"""Fish Audio S2 Pro renderer — the only module that touches MLX."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from vo import _version_check  # noqa: F401
from vo.registries import DEFAULT_VOICES_PATH
from vo.tags import apply_tag_mode
from vo.voice_resolver import resolve_voice
from vo.quality import extract_words, evaluate, QualityCheck

__all__ = ["render", "RenderResult"]

# ---- Lazy model loaders (monkeypatchable in tests) ---------------------

_MODEL = None
_STT = None
_MODEL_PATH = "~/Library/Caches/mlx-audio/fish-audio-s2-pro-bf16"


def _get_model():
    global _MODEL
    if _MODEL is None:
        import os
        from mlx_audio.tts.utils import load_model
        _MODEL = load_model(os.path.expanduser(_MODEL_PATH))
    return _MODEL


def _get_stt():
    global _STT
    if _STT is None:
        from mlx_audio.stt import load
        _STT = load("mlx-community/whisper-large-v3-turbo-asr-fp16")
    return _STT


def _load_ref_audio(ref_path: Path, sample_rate: int):
    from mlx_audio.utils import load_audio
    return load_audio(str(ref_path), sample_rate=sample_rate, volume_normalize=False)


def _write_audio(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    from mlx_audio.audio_io import write as audio_write
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_write(str(path), np.array(audio), sample_rate, format="wav")


# ---- Public API --------------------------------------------------------

@dataclass
class RenderResult:
    wav_path: Path
    words_path: Path | None
    tagged_path: Path
    voice_id: str | None
    duration_s: float
    attempts_used: int
    quality_passed: bool


def render(
    *,
    script: str,
    out_path: Path,
    voice: str | None = None,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    voices_path: Path = DEFAULT_VOICES_PATH,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 30,
    speed: float = 1.0,
    max_tokens: int = 4096,
    chunk_length: int = 300,
    no_stt: bool = False,
    multi_speaker: bool = False,
    language: str = "en",
    tag_mode: str = "auto",
    anchors: list[list[str]] | None = None,
    max_retries: int = 1,
    max_silence_gap: float = 2.5,
) -> RenderResult:
    out_path = Path(out_path)

    # 1. Tag preprocessing
    final_script = apply_tag_mode(script, tag_mode)

    # 2. Resolve voice
    rv = resolve_voice(
        voice_id=voice, ref_audio=ref_audio, ref_text=ref_text,
        voices_path=voices_path,
    )

    model = _get_model()
    sr = int(model.sample_rate)

    ref_arr = _load_ref_audio(rv.audio_path, sr) if rv.audio_path else None

    # 3. Generate
    results = list(model.generate(
        text=final_script,
        ref_audio=ref_arr,
        ref_text=rv.transcript,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        chunk_length=chunk_length,
        speed=speed,
    ))
    if not results:
        raise RuntimeError("Fish Speech returned no audio segments")

    # Concatenate any segments
    if len(results) == 1:
        audio = results[0].audio
    else:
        import mlx.core as mx
        audio = mx.concatenate([r.audio for r in results])

    try:
        import mlx.core as mx
        mx.eval(audio)
    except Exception:
        pass

    _write_audio(out_path, audio, sr)
    duration_s = float(audio.shape[0]) / sr

    # 4. Sidecars (sttless path is short-circuited later in this plan)
    tagged_path = out_path.with_suffix(".tagged.txt")
    tagged_path.write_text(final_script)

    words_path: Path | None = None
    quality_passed = True
    if not no_stt:
        stt = _get_stt()
        r = stt.generate(str(out_path), word_timestamps=True,
                         return_timestamps=True, condition_on_previous_text=False)
        words = extract_words(r)
        check: QualityCheck = evaluate(words, max_silence_gap=max_silence_gap,
                                       anchors=anchors)
        quality_passed = check.passed
        words_path = out_path.with_suffix(".words.json")
        import json
        words_path.write_text(json.dumps({
            "duration_s": duration_s,
            "max_gap": check.max_gap,
            "quality_passed": check.passed,
            "quality_reason": check.reason,
            "anchor_starts": check.anchor_starts,
            "words": [{"start": w.start, "end": w.end, "text": w.text}
                      for w in words],
        }, indent=2))

    return RenderResult(
        wav_path=out_path,
        words_path=words_path,
        tagged_path=tagged_path,
        voice_id=rv.voice_id,
        duration_s=duration_s,
        attempts_used=1,
        quality_passed=quality_passed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_render.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/render.py vo/tests/test_render.py vo/tests/conftest.py
git commit -m "feat(vo): render() happy path with mocked model"
```

---

## Task 12: Renderer — quality-gate retry loop

**Files:**
- Modify: `vo/render.py`
- Modify: `vo/tests/test_render.py`

- [ ] **Step 1: Append failing tests to `vo/tests/test_render.py`**

```python
def test_render_retries_on_silence_gap(monkeypatch, tmp_path, sample_voices_path):
    from vo import render
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


def test_render_exhausts_retries_and_marks_failed(monkeypatch, tmp_path, sample_voices_path):
    from vo import render
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


def test_render_no_stt_skips_quality_gate(monkeypatch, tmp_path, sample_voices_path):
    from vo import render
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
```

- [ ] **Step 2: Run tests to verify the retry tests fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_render.py -v -k "retries or exhaust or no_stt"
```

Expected: failures (current render has `max_retries=1` and doesn't loop).

- [ ] **Step 3: Replace the generation+sidecar section in `vo/render.py`**

Replace everything from `# 3. Generate` through the `return RenderResult(...)` at the end of `render()` with this retry-loop version:

```python
    # 3. Generation with quality-gate retry loop
    import json as _json
    import mlx.core as mx

    attempts_used = 0
    final_audio = None
    final_words = None
    final_check: QualityCheck | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        attempts_used = attempt
        # Perturb temperature on retry to break out of stuck states
        attempt_temp = temperature + 0.05 * (attempt - 1)
        results = list(model.generate(
            text=final_script,
            ref_audio=ref_arr,
            ref_text=rv.transcript,
            temperature=attempt_temp,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            chunk_length=chunk_length,
            speed=speed,
        ))
        if not results:
            continue
        audio = results[0].audio if len(results) == 1 else mx.concatenate([r.audio for r in results])
        try:
            mx.eval(audio)
        except Exception:
            pass
        _write_audio(out_path, audio, sr)
        final_audio = audio

        if no_stt:
            final_check = QualityCheck(passed=True, max_gap=0.0,
                                       anchor_starts=None, reason="")
            break

        stt = _get_stt()
        r = stt.generate(str(out_path), word_timestamps=True,
                         return_timestamps=True, condition_on_previous_text=False)
        words = extract_words(r)
        check = evaluate(words, max_silence_gap=max_silence_gap, anchors=anchors)
        final_words = words
        final_check = check
        if check.passed:
            break

    if final_audio is None:
        raise RuntimeError("Fish Speech returned no audio segments across all retries")
    assert final_check is not None

    duration_s = float(final_audio.shape[0]) / sr

    # 4. Sidecars
    tagged_path = out_path.with_suffix(".tagged.txt")
    tagged_path.write_text(final_script)

    words_path: Path | None = None
    if not no_stt:
        words_path = out_path.with_suffix(".words.json")
        words_path.write_text(_json.dumps({
            "duration_s": duration_s,
            "max_gap": final_check.max_gap,
            "quality_passed": final_check.passed,
            "quality_reason": final_check.reason,
            "anchor_starts": final_check.anchor_starts,
            "words": [{"start": w.start, "end": w.end, "text": w.text}
                      for w in (final_words or [])],
        }, indent=2))

    return RenderResult(
        wav_path=out_path,
        words_path=words_path,
        tagged_path=tagged_path,
        voice_id=rv.voice_id,
        duration_s=duration_s,
        attempts_used=attempts_used,
        quality_passed=final_check.passed,
    )
```

Also bump the default `max_retries` from 1 to 4 in the signature:

```python
    max_retries: int = 4,
```

- [ ] **Step 4: Run all render tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_render.py -v
```

Expected: all PASS (4 happy-path + 3 retry tests).

- [ ] **Step 5: Commit**

```bash
git add vo/render.py vo/tests/test_render.py
git commit -m "feat(vo): quality-gate retry loop with temperature perturbation"
```

---

## Task 13: CLI — primary render path

**Files:**
- Modify: `vo/render.py` (append `main()` + argparse setup; add `if __name__ == "__main__":` block)
- Create: `vo/tests/test_cli.py`

- [ ] **Step 1: Write failing tests in `vo/tests/test_cli.py`**

```python
"""Tests for vo.render CLI."""
import sys
from pathlib import Path

import pytest

from vo import render
from vo.tests.conftest import FakeFishModel, FakeWhisper


@pytest.fixture
def cli_setup(monkeypatch, sample_voices_path, tmp_path):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_cli.py -v
```

Expected: failures (`render.main` not defined yet).

- [ ] **Step 3: Append CLI to `vo/render.py`**

Add at the bottom of `vo/render.py`:

```python
# ---- CLI ---------------------------------------------------------------

import argparse
import json as _cli_json
import sys

from vo.registries import (
    DEFAULT_PRESETS_PATH, load_presets, Preset,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vo",
        description="Fish Audio S2 Pro voiceover renderer.",
    )
    p.add_argument("--script", help="Script file (- for stdin).")
    p.add_argument("--out", help="Output WAV path.")

    # Voice selection (mutually exclusive enforced post-parse, not via argparse,
    # because the spec requires a custom error message).
    p.add_argument("--voice", help="Voice ID from voices.json.")
    p.add_argument("--ref-audio", help="Ad-hoc reference audio path.")
    p.add_argument("--ref-text", help="Transcript of --ref-audio. Auto-Whispered if missing.")

    # Preset
    p.add_argument("--preset", help="Preset name (defaults from presets.json).")
    p.add_argument("--no-preset", action="store_true",
                   help="Ignore any preset defaults.")

    # Sampling
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--speed", type=float)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--chunk-length", type=int, default=300)

    # Quality gate
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--max-silence-gap", type=float, default=2.5)
    p.add_argument("--no-stt", action="store_true")
    p.add_argument("--anchors-json",
                   help="JSON-encoded list of anchor phrase lists, e.g. "
                        "'[[\"most\",\"content\"],[\"that\\u2019s\",\"not\"]]'.")

    # Features
    p.add_argument("--multi-speaker", action="store_true")
    p.add_argument("--language", default="en")
    p.add_argument("--tag-mode", choices=["auto", "explicit", "none"], default="auto")
    p.add_argument("--seed", type=int)

    # Registry paths (mainly for tests; can also be overridden in practice)
    p.add_argument("--voices-path", default=str(DEFAULT_VOICES_PATH))
    p.add_argument("--presets-path", default=str(DEFAULT_PRESETS_PATH))

    # Admin paths (filled in next task)
    p.add_argument("--save-voice")
    p.add_argument("--label")
    p.add_argument("--notes")
    p.add_argument("--save-preset")
    p.add_argument("--preset-notes")
    p.add_argument("--add-voice")
    p.add_argument("--audio")
    p.add_argument("--transcript")
    p.add_argument("--add-preset")
    p.add_argument("--json")
    p.add_argument("--transcribe")

    return p


def _read_script(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text()


def _apply_preset(args, preset: Preset) -> None:
    """Fill un-set sampling args from preset defaults."""
    if args.voice is None and preset.voice is not None:
        args.voice = preset.voice
    if args.temperature is None:
        args.temperature = preset.temperature
    if args.top_p is None:
        args.top_p = preset.top_p
    if args.top_k is None:
        args.top_k = preset.top_k
    if args.speed is None:
        args.speed = preset.speed
    if args.language == "en" and preset.language != "en":
        args.language = preset.language


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Admin paths handled in a later task; here we only do the render path.
    if any([args.add_voice, args.add_preset, args.transcribe]):
        return _admin_main(args)

    if not args.script or not args.out:
        parser.error("--script and --out are required for rendering")

    if args.voice and args.ref_audio:
        print("error: cannot supply both --voice and --ref-audio",
              file=sys.stderr)
        return 2

    # Preset defaults (--no-preset wins)
    if args.preset and not args.no_preset:
        try:
            preset = load_presets(Path(args.presets_path))[args.preset]
        except KeyError:
            print(f"error: unknown preset {args.preset!r}", file=sys.stderr)
            return 2
        _apply_preset(args, preset)

    # Final fallback defaults
    if args.temperature is None: args.temperature = 0.7
    if args.top_p is None:       args.top_p = 0.7
    if args.top_k is None:       args.top_k = 30
    if args.speed is None:       args.speed = 1.0

    anchors = None
    if args.anchors_json:
        anchors = _cli_json.loads(args.anchors_json)

    script_text = _read_script(args.script)

    try:
        result = render(
            script=script_text,
            out_path=Path(args.out),
            voice=args.voice,
            ref_audio=Path(args.ref_audio) if args.ref_audio else None,
            ref_text=args.ref_text,
            voices_path=Path(args.voices_path),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            speed=args.speed,
            max_tokens=args.max_tokens,
            chunk_length=args.chunk_length,
            no_stt=args.no_stt,
            multi_speaker=args.multi_speaker,
            language=args.language,
            tag_mode=args.tag_mode,
            anchors=anchors,
            max_retries=args.max_retries,
            max_silence_gap=args.max_silence_gap,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(_cli_json.dumps({
        "wav": str(result.wav_path),
        "words": str(result.words_path) if result.words_path else None,
        "tagged": str(result.tagged_path),
        "voice_id": result.voice_id,
        "duration_s": result.duration_s,
        "attempts_used": result.attempts_used,
        "quality_passed": result.quality_passed,
    }))
    return 0 if result.quality_passed else 5


def _admin_main(args) -> int:
    """Stub — implemented in the next task."""
    print("error: admin paths not yet implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_cli.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/render.py vo/tests/test_cli.py
git commit -m "feat(vo): CLI render path with preset defaults and flag overrides"
```

---

## Task 14: CLI — admin paths (`--add-voice`, `--add-preset`, `--transcribe`)

**Files:**
- Modify: `vo/render.py` (replace `_admin_main` stub)
- Modify: `vo/tests/test_cli.py` (append admin tests)

- [ ] **Step 1: Append failing tests**

```python
def test_cli_add_voice_writes_registry(cli_setup, tmp_path):
    _, _, voices, tmp = cli_setup
    audio = tmp / "ref.wav"
    audio.write_bytes(b"")
    rc = render.main([
        "--add-voice", "newone",
        "--audio", str(audio),
        "--transcript", "the transcript",
        "--label", "Newone",
        "--voices-path", str(voices),
    ])
    assert rc == 0
    import json
    data = json.loads(voices.read_text())
    assert data["voices"]["newone"]["label"] == "Newone"
    assert data["voices"]["newone"]["transcript"] == "the transcript"


def test_cli_add_voice_auto_transcribes(cli_setup, monkeypatch, tmp_path):
    _, _, voices, tmp = cli_setup
    audio = tmp / "ref.wav"
    audio.write_bytes(b"")
    monkeypatch.setattr("vo.transcribe.transcribe", lambda p: "auto txt")
    rc = render.main([
        "--add-voice", "v2",
        "--audio", str(audio),
        "--voices-path", str(voices),
    ])
    assert rc == 0
    import json
    assert json.loads(voices.read_text())["voices"]["v2"]["transcript"] == "auto txt"


def test_cli_add_preset_writes_registry(cli_setup, tmp_path):
    _, _, _, tmp = cli_setup
    presets = tmp / "presets.json"
    presets.write_text('{"version":1,"presets":{}}')
    payload = '{"voice":"excited","temperature":0.8,"tag_density":"high"}'
    rc = render.main([
        "--add-preset", "newp",
        "--json", payload,
        "--presets-path", str(presets),
    ])
    assert rc == 0
    import json
    data = json.loads(presets.read_text())
    assert data["presets"]["newp"]["temperature"] == 0.8
    assert data["presets"]["newp"]["tag_density"] == "high"


def test_cli_transcribe_prints_transcript(cli_setup, monkeypatch, tmp_path, capsys):
    _, _, _, tmp = cli_setup
    audio = tmp / "a.wav"
    audio.write_bytes(b"")
    monkeypatch.setattr("vo.transcribe.transcribe", lambda p: "transcribed!")
    rc = render.main(["--transcribe", str(audio)])
    assert rc == 0
    assert "transcribed!" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_cli.py -v -k "add_voice or add_preset or transcribe"
```

Expected: 4 failures.

- [ ] **Step 3: Replace `_admin_main` in `vo/render.py`**

```python
def _admin_main(args) -> int:
    from vo.registries import add_voice, add_preset, Voice, Preset

    if args.transcribe:
        from vo.transcribe import transcribe
        audio = Path(args.transcribe)
        if not audio.exists():
            print(f"error: not found: {audio}", file=sys.stderr)
            return 2
        text = transcribe(audio)
        print(text.strip())
        return 0

    if args.add_voice:
        if not args.audio:
            print("error: --add-voice requires --audio", file=sys.stderr)
            return 2
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"error: --audio not found: {audio_path}", file=sys.stderr)
            return 2
        transcript = args.transcript
        if transcript is None:
            from vo.transcribe import transcribe
            transcript = transcribe(audio_path).strip()
        v = Voice(
            id=args.add_voice,
            label=args.label or args.add_voice,
            audio=str(audio_path),
            transcript=transcript,
            notes=args.notes or "",
        )
        add_voice(v, Path(args.voices_path))
        print(_cli_json.dumps({"added_voice": args.add_voice,
                               "path": args.voices_path}))
        return 0

    if args.add_preset:
        if not args.json:
            print("error: --add-preset requires --json", file=sys.stderr)
            return 2
        try:
            payload = _cli_json.loads(args.json)
        except _cli_json.JSONDecodeError as e:
            print(f"error: --json invalid: {e}", file=sys.stderr)
            return 2
        p = Preset(
            name=args.add_preset,
            voice=payload.get("voice"),
            tag_hints=list(payload.get("tag_hints", [])),
            tag_density=payload.get("tag_density", "medium"),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.7)),
            top_k=int(payload.get("top_k", 30)),
            speed=float(payload.get("speed", 1.0)),
            language=payload.get("language", "en"),
            notes=args.preset_notes or payload.get("notes", ""),
        )
        add_preset(p, Path(args.presets_path))
        print(_cli_json.dumps({"added_preset": args.add_preset,
                               "path": args.presets_path}))
        return 0

    return 1
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_cli.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/render.py vo/tests/test_cli.py
git commit -m "feat(vo): admin CLI paths (--add-voice/--add-preset/--transcribe)"
```

---

## Task 15: CLI — post-render registry mutations (`--save-voice`, `--save-preset`)

**Files:**
- Modify: `vo/render.py` (`main()` post-render block)
- Modify: `vo/tests/test_cli.py`

- [ ] **Step 1: Append failing tests**

```python
def test_cli_save_voice_after_ref_audio_run(cli_setup, monkeypatch, tmp_path):
    _, _, voices, tmp = cli_setup
    audio = tmp / "ref.wav"
    audio.write_bytes(b"")
    monkeypatch.setattr("vo.transcribe.transcribe", lambda p: "auto txt")
    script = tmp / "s.txt"
    script.write_text("hi")
    out = tmp / "o.wav"
    rc = render.main([
        "--script", str(script), "--out", str(out),
        "--ref-audio", str(audio),
        "--save-voice", "newone", "--label", "Newone",
        "--voices-path", str(voices),
    ])
    assert rc == 0
    import json
    data = json.loads(voices.read_text())
    assert "newone" in data["voices"]
    assert data["voices"]["newone"]["label"] == "Newone"


def test_cli_save_preset_after_run(cli_setup, tmp_path):
    import json
    _, _, voices, tmp = cli_setup
    presets = tmp / "presets.json"
    presets.write_text('{"version":1,"presets":{}}')
    script = tmp / "s.txt"; script.write_text("hi")
    out = tmp / "o.wav"
    rc = render.main([
        "--script", str(script), "--out", str(out),
        "--voice", "excited",
        "--temperature", "0.85", "--speed", "1.1",
        "--save-preset", "winning",
        "--preset-notes", "Captured from a good run",
        "--voices-path", str(voices),
        "--presets-path", str(presets),
    ])
    assert rc == 0
    data = json.loads(presets.read_text())
    p = data["presets"]["winning"]
    assert p["temperature"] == 0.85
    assert p["speed"] == 1.1
    assert p["voice"] == "excited"
    assert p["notes"] == "Captured from a good run"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv-mlx/bin/pytest vo/tests/test_cli.py -v -k "save_voice or save_preset"
```

Expected: 2 failures (save flags currently ignored).

- [ ] **Step 3: Modify the post-render block in `main()` of `vo/render.py`**

Right before the final `return 0 if result.quality_passed else 5` in `main()`, insert:

```python
    if args.save_voice and result.quality_passed and args.ref_audio:
        from vo.registries import Voice as _Voice, add_voice as _add_voice
        # Reuse the transcript we already resolved (via resolve_voice in render);
        # re-resolve here to avoid threading it through RenderResult.
        from vo.voice_resolver import resolve_voice as _resolve
        rv = _resolve(voice_id=None, ref_audio=Path(args.ref_audio),
                      ref_text=args.ref_text, voices_path=Path(args.voices_path))
        v = _Voice(
            id=args.save_voice,
            label=args.label or args.save_voice,
            audio=str(Path(args.ref_audio)),
            transcript=rv.transcript or "",
            notes=args.notes or "",
        )
        _add_voice(v, Path(args.voices_path))
        print(_cli_json.dumps({"saved_voice": args.save_voice}))

    if args.save_preset and result.quality_passed:
        from vo.registries import Preset as _Preset, add_preset as _add_preset
        p = _Preset(
            name=args.save_preset,
            voice=args.voice,
            tag_hints=[],
            tag_density="medium",
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            speed=args.speed,
            language=args.language,
            notes=args.preset_notes or "",
        )
        _add_preset(p, Path(args.presets_path))
        print(_cli_json.dumps({"saved_preset": args.save_preset}))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-mlx/bin/pytest vo/tests/test_cli.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/render.py vo/tests/test_cli.py
git commit -m "feat(vo): --save-voice / --save-preset capture successful runs"
```

---

## Task 16: Seed `voices.json` and `presets.json`

**Files:**
- Create: `vo/voices.json`
- Create: `vo/presets.json`

- [ ] **Step 1: Write `vo/voices.json`**

```json
{
  "version": 1,
  "default": "excited",
  "voices": {
    "excited": {
      "label": "Excited (energetic upbeat)",
      "audio": "voice samples/Excited VO clone sample.m4a",
      "transcript": "<paste a transcript of your own Excited VO reference here>",
      "notes": "Best for hype/announce/hook content.",
      "created_at": "2026-05-14",
      "created_by": "manual"
    },
    "aggressive": {
      "label": "Aggressive (punchy pitch)",
      "audio": "voice samples/Aggressive VO clone sample.m4a",
      "transcript": "<paste a transcript of your own Aggressive VO reference here>",
      "notes": "Best for direct sales/CTA-driven scripts.",
      "created_at": "2026-05-14",
      "created_by": "manual"
    },
    "resolute": {
      "label": "Resolute (calm authority)",
      "audio": "voice samples/Resolute (slow) VO clone sample.m4a",
      "transcript": "<paste a transcript of your own Resolute VO reference here>",
      "notes": "Best for tutorial/explainer/story content.",
      "created_at": "2026-05-14",
      "created_by": "manual"
    }
  }
}
```

- [ ] **Step 2: Write `vo/presets.json`**

```json
{
  "version": 1,
  "presets": {
    "fb-reel-hype": {
      "voice": "excited",
      "tag_hints": ["[emphasis]", "[excited]", "[short pause]"],
      "tag_density": "high",
      "temperature": 0.75,
      "top_p": 0.7,
      "top_k": 30,
      "speed": 1.05,
      "language": "en",
      "notes": "Fast-paced FB/IG reel narration with punchy emphasis."
    },
    "tutorial-explainer": {
      "voice": "resolute",
      "tag_hints": ["[pause]", "[emphasis]"],
      "tag_density": "low",
      "temperature": 0.6,
      "top_p": 0.7,
      "top_k": 30,
      "speed": 0.95,
      "language": "en",
      "notes": "Calm, measured authority. Light tagging."
    },
    "direct-pitch": {
      "voice": "aggressive",
      "tag_hints": ["[emphasis]", "[short pause]", "[loud]"],
      "tag_density": "medium",
      "temperature": 0.7,
      "top_p": 0.7,
      "top_k": 30,
      "speed": 1.0,
      "language": "en",
      "notes": "Confident sales delivery with deliberate emphasis."
    },
    "story-emotional": {
      "voice": "resolute",
      "tag_hints": ["[sigh]", "[pause]", "[low voice]", "[whisper]"],
      "tag_density": "medium",
      "temperature": 0.7,
      "top_p": 0.7,
      "top_k": 30,
      "speed": 0.95,
      "language": "en",
      "notes": "Slow, emotional storytelling."
    },
    "ad-aggressive-cta": {
      "voice": "aggressive",
      "tag_hints": ["[shouting]", "[emphasis]", "[surprised]"],
      "tag_density": "high",
      "temperature": 0.8,
      "top_p": 0.7,
      "top_k": 30,
      "speed": 1.05,
      "language": "en",
      "notes": "High-energy ad with strong CTA push."
    }
  }
}
```

- [ ] **Step 3: Quick smoke check the registries load cleanly**

```bash
.venv-mlx/bin/python -c "from vo.registries import load_voices, load_presets; v = load_voices(); p = load_presets(); print('voices:', list(v)); print('presets:', list(p))"
```

Expected:
```
voices: ['excited', 'aggressive', 'resolute']
presets: ['fb-reel-hype', 'tutorial-explainer', 'direct-pitch', 'story-emotional', 'ad-aggressive-cta']
```

- [ ] **Step 4: Run the full unit-test suite to make sure nothing regressed**

```bash
.venv-mlx/bin/pytest vo/tests/ -v --ignore=vo/tests/test_smoke.py
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vo/voices.json vo/presets.json
git commit -m "feat(vo): seed voices.json and presets.json with starter data"
```

---

## Task 17: Skill `SKILL.md`

**Files:**
- Create: `.claude/skills/vo-fish/SKILL.md`

- [ ] **Step 1: Write `.claude/skills/vo-fish/SKILL.md`**

```markdown
---
name: vo-fish
description: Use when the user wants to generate a voiceover for content (reel, ad, explainer, story, hook). Generates expressive cloned-voice audio locally via Fish Audio S2 Pro, picking voice + emotion tags + sampling parameters based on the script and the stated content goal.
---

# vo-fish

Generates a voiceover from a script using Fish Audio S2 Pro (running locally
on Apple Silicon). You read the brand brief, the script, and the user's stated
goal; you pick a voice from the registry (or accept an ad-hoc reference); you
inject inline emotion tags clause-by-clause; you call the Python renderer; you
report the output triplet back.

## When to use

The user wants a voiceover. Cues include:
- "make me a voiceover for…"
- "voice this script for…"
- "VO this in <some style>…"
- "read this in <voice>…"
- Pasting a script with no clear deliverable other than audio.

If they also want a video/reel built from the audio, generate the VO first
with this skill, then defer the muxing/sync work to the existing reel scripts
(`make-reel-synced.py`, etc.). This skill does NOT mux video.

## Inputs you need before generating

Ask for whatever isn't already in the conversation:

1. **The script.** A literal block of text you'll voice.
2. **The goal.** One sentence: "hype reel hook," "calm tutorial explainer,"
   "aggressive sales pitch," "emotional story," etc. The goal drives every
   downstream choice.
3. **Voice (optional).** If they don't name one, you pick from the registry
   based on goal; or use the registry default. If they hand you a custom
   audio file path, use that as ad-hoc `--ref-audio` (no registry edit
   required).

If the goal is unclear, ask one clarifying question before generating. Don't
guess if the answer changes the voice choice.

## Workflow

1. **Read context.**
   - `brand-brief.md` — voice rules, audience, primary CTA.
   - `vo/voices.json` — available cloned voices (id, label, notes).
   - `vo/presets.json` — starter recipes; **suggestions, not constraints**.

2. **Pick a voice.** Reach for a registry voice that matches the goal. Look at
   the `notes` field on each voice. If the user gave you a file path, use it
   via `--ref-audio PATH` instead — the registry is a convenience, not a fence.

3. **Choose / override a preset.** If a preset matches the goal cleanly, use
   `--preset NAME` for sane defaults. Override anything that doesn't fit
   (`--preset hype --voice resolute --temperature 0.6` is fine). If no preset
   matches, pass parameters directly with `--no-preset`.

4. **Inject inline emotion tags.** Read the script clause by clause and add
   Fish Speech tags where they add real expressive value. Don't carpet-bomb —
   density should match the goal (high for hype, low for explainers).
   - Curated tags include: `[pause]`, `[short pause]`, `[emphasis]`,
     `[excited]`, `[whisper]`, `[shouting]`, `[loud]`, `[low voice]`,
     `[sigh]`, `[laughing]`, `[chuckle]`, `[surprised]`, `[angry]`, `[sad]`,
     `[delight]`, `[clearing throat]`, `[exhale]`, `[inhale]`, `[volume up]`,
     `[volume down]`.
   - Free-form text tags work too: `[professional broadcast tone]`,
     `[with strong accent]`, `[low and conspiratorial]`. Use these when the
     curated set doesn't capture the vibe.
   - Tags act on the words that follow them. `Hi there. [whisper] Don't tell
     anyone.` whispers only the second sentence.

5. **Write the tagged script to a temp file.** Don't modify the user's
   original script in place.

6. **Run the renderer.** Always go through the project venv:

   ```bash
   .venv-mlx/bin/python vo/render.py \
       --script /tmp/script.txt \
       --out posts/<date>/<platform>/vo.wav \
       --voice <id_or_omit_for_ref> \
       --preset <name_or_omit>
   ```

   Add any explicit overrides on the end. The renderer prints a JSON line
   with output paths; parse and surface those to the user.

7. **Verify and report.** The renderer's quality gate auto-retries on Fish
   Speech collapses. If `quality_passed` is false in the JSON output, tell the
   user and offer to re-roll.

8. **Offer to grow the registries.** When the run was good:
   - If the user supplied ad-hoc `--ref-audio`, offer to save it via
     `--save-voice <id>`. Confirm with them first.
   - If you discovered a tag/parameter recipe that worked well for a content
     type they use often, offer to snapshot it via `--save-preset <name>`.
   - Never silently write to either registry.

## Flexibility principles (load-bearing)

- **Presets are starting points, not constraints.** Override anything that
  doesn't fit. `--preset` + explicit `--voice`/`--temperature`/etc. work fine
  together.
- **The voice registry is a convenience.** If the user passes a path to their
  own audio, use `--ref-audio` directly. No registry edit required to clone a
  new voice.
- **Tag injection is opt-out per run.** Add `--tag-mode explicit` if the user
  has already tagged the script and wants you to leave it alone. Add
  `--tag-mode none` to strip tags entirely.

## Output deliverables

For every run, surface all three paths to the user:

- `<out>.wav` — the audio
- `<out>.words.json` — Whisper word-level timestamps (consumable by
  `make-reel-synced.py` for slide-cued video)
- `<out>.tagged.txt` — your tag-annotated script so the user can see (and
  tweak) what you decided

## Examples

User: "make me a hype VO for our new product drop tomorrow: 'This is the
biggest thing we've ever shipped. Three years of work. Read the thread.'"

You:
1. Pick voice `excited` (matches hype goal).
2. Pick preset `fb-reel-hype`.
3. Tag the script: `[excited] This is the biggest thing we've ever shipped.
   [short pause] [emphasis] Three years of work. [pause] Read the thread.`
4. Write to `/tmp/vo_<timestamp>.txt`.
5. Run: `.venv-mlx/bin/python vo/render.py --script /tmp/vo_*.txt --out
   out/hype.wav --preset fb-reel-hype`
6. Report paths.

User: "read this calmly in [their voice sample.m4a]: 'Welcome back. Today
we're going to walk through what changed.'"

You:
1. Ad-hoc voice via `--ref-audio "<their sample.m4a>"`. No registry change.
2. Pick preset `tutorial-explainer` for sampling defaults; override `--voice`
   isn't needed because we're using `--ref-audio`.
3. Tag lightly: `Welcome back. [short pause] Today we're going to walk
   through what changed.`
4. Run with `--ref-audio` + `--preset tutorial-explainer --no-preset` if you
   want the preset's sampling but its voice-id shouldn't apply (it won't
   anyway because `--voice` and `--ref-audio` are exclusive).
5. After a successful run, offer: "Want me to save this as a registered voice
   for future runs? (`--save-voice <id>`)"
```

- [ ] **Step 2: Sanity-check the skill file exists and frontmatter parses**

```bash
.venv-mlx/bin/python -c "
import re
text = open('.claude/skills/vo-fish/SKILL.md').read()
m = re.match(r'---\n(.*?)\n---', text, re.DOTALL)
assert m, 'no frontmatter'
fm = m.group(1)
assert 'name: vo-fish' in fm
assert 'description:' in fm
print('SKILL.md OK')
"
```

Expected: `SKILL.md OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/vo-fish/SKILL.md
git commit -m "feat(vo): vo-fish skill prompt"
```

---

## Task 18: Opt-in end-to-end smoke test

**Files:**
- Create: `vo/tests/test_smoke.py`

- [ ] **Step 1: Write `vo/tests/test_smoke.py`**

```python
"""End-to-end smoke test. Skipped by default — opt in via VO_SMOKE=1.

Generates a short VO with the real Fish Speech model and verifies the
transcript matches the input within a tolerance.
"""
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("VO_SMOKE") != "1",
    reason="Set VO_SMOKE=1 to run end-to-end smoke test (slow, uses the real model).",
)


def _wer(reference: str, hypothesis: str) -> float:
    """Crude WER: edit distance over reference word count."""
    import re
    r = re.findall(r"[a-z0-9']+", reference.lower())
    h = re.findall(r"[a-z0-9']+", hypothesis.lower())
    if not r:
        return float("inf")
    # Levenshtein on word lists
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(r)


def test_excited_voice_round_trips(tmp_path):
    from vo import render
    text = "Most content creators are sitting on hundreds of hours of footage they'll never use."
    out = tmp_path / "smoke.wav"
    result = render.render(
        script=text, out_path=out, voice="excited",
        max_retries=2, max_silence_gap=2.5,
    )
    assert result.wav_path.exists()
    assert result.quality_passed, f"quality gate failed: {result}"

    # Re-transcribe and check WER
    from mlx_audio.stt import load
    stt = load("mlx-community/whisper-large-v3-turbo-asr-fp16")
    r = stt.generate(str(out))
    got = getattr(r, "text", None) or ""
    wer = _wer(text, got)
    assert wer < 0.15, f"WER {wer:.2f} too high. Got: {got!r}"
```

- [ ] **Step 2: Run the smoke test once manually**

```bash
VO_SMOKE=1 .venv-mlx/bin/pytest vo/tests/test_smoke.py -v -s
```

Expected: 1 PASS in ~30-60 seconds. If it fails, the model is degraded for that voice/input — manually re-roll once before declaring a real bug.

- [ ] **Step 3: Run the full suite without the smoke flag to confirm default skip**

```bash
.venv-mlx/bin/pytest vo/tests/ -v
```

Expected: all unit tests PASS, `test_smoke.py::test_excited_voice_round_trips` SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add vo/tests/test_smoke.py
git commit -m "test(vo): opt-in end-to-end smoke test (VO_SMOKE=1)"
```

---

## Final verification

- [ ] **All tests pass:**

```bash
.venv-mlx/bin/pytest vo/tests/ -v
```

Expected: all unit tests PASS, smoke test SKIPPED.

- [ ] **CLI is wired and shows help:**

```bash
.venv-mlx/bin/python vo/render.py --help
```

Expected: full flag list including `--voice`, `--ref-audio`, `--preset`,
`--save-voice`, `--save-preset`, `--add-voice`, `--add-preset`, `--transcribe`.

- [ ] **Skill is discoverable:**

```bash
ls .claude/skills/vo-fish/SKILL.md
```

Expected: file exists.

- [ ] **Registries load:**

```bash
.venv-mlx/bin/python -c "from vo.registries import load_voices, load_presets; print(list(load_voices()), list(load_presets()))"
```

Expected:
```
['excited', 'aggressive', 'resolute'] ['fb-reel-hype', 'tutorial-explainer', 'direct-pitch', 'story-emotional', 'ad-aggressive-cta']
```

- [ ] **One-shot end-to-end check (optional but recommended):**

```bash
echo "Hello [excited] world." | .venv-mlx/bin/python vo/render.py \
    --script - --out /tmp/vo_smoke.wav --voice excited
```

Expected: JSON line on stdout with `quality_passed: true` and three output paths.

---

## Spec coverage map

Quick mapping from spec sections to tasks for review:

| Spec section | Task(s) |
|---|---|
| File layout | 1, 16, 17 |
| Skill workflow | 17 |
| Renderer public Python API | 11, 12 |
| Renderer CLI surface | 13, 14, 15 |
| Output sidecars | 11, 12 |
| `voices.json` registry | 2, 3, 16 |
| `presets.json` registry | 4, 16 |
| Flexibility principles | 5 (tag modes), 10 (voice resolver), 13–15 (preset+save flags), 17 (skill prompt) |
| Feature mapping (all Fish features) | 11 (model.generate kwargs), 13 (CLI flags), 5 (tag modes) |
| Quality gate | 6, 7, 8, 12 |
| Error handling | 1 (version pin), 2–4 (registry errors), 10 (voice resolution errors), 13 (CLI conflicts) |
| Testing | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18 |
