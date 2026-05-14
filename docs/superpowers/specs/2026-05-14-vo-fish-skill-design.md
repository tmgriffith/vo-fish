# VO Fish Skill — Design

> Date: 2026-05-14
> Status: Approved (pending user review of this spec file)

## Goal

Build a Claude Code skill (`vo-fish`) that produces expressive voiceovers for
Content Machine using Fish Audio S2 Pro locally on Apple Silicon. The skill
must let Claude make creative choices (voice, emotion tags, sampling) based on
the script and the stated goal of the content, while keeping every aspect
overridable. The tool must not lock users into preset goals or registered
voices.

## Non-goals

- Not a reel/video renderer. Outputs are audio + sidecars; existing reel
  scripts (`make-reel-synced.py`, etc.) consume them.
- Not a multi-take A/B picker. A single best take is produced per run (with
  auto-retry under the quality gate).
- Not a cloud-hosted service. Local Apple Silicon only, via the pinned
  `.venv-mlx` (mlx-audio 0.4.2).

## Architecture

Two-layer split:

1. **Skill markdown** (`.claude/skills/vo-fish/SKILL.md`) — the creative brain.
   Claude reads the script, brand brief, voices.json, and presets.json, picks a
   voice, injects inline emotion tags, sets sampling params, and shells out to
   the Python renderer.

2. **Python renderer** (`vo/render.py`) — the inference engine. Loads the Fish
   Speech model, runs generation with the chosen params, applies the quality
   gate with auto-retry, writes the WAV + sidecars. Importable as a function
   from other scripts; also exposes a CLI.

The two layers are decoupled: the renderer can be called by any script (not
just the skill), and the skill never imports MLX directly.

### File layout

```
Content Machine/
├── .claude/
│   └── skills/
│       └── vo-fish/
│           └── SKILL.md
├── vo/
│   ├── render.py            # renderer (CLI + importable)
│   ├── voices.json          # voice registry
│   ├── presets.json         # goal presets
│   └── README.md            # human-readable notes
└── .venv-mlx/               # existing 3.13 venv pinned to mlx-audio==0.4.2
```

## Skill workflow

When invoked (e.g., user says "make a hype VO for this script: …"):

1. Skill reads `brand-brief.md` for voice rules, audience, primary CTA.
2. Skill reads `vo/voices.json` and `vo/presets.json` to know what's available.
3. Claude analyzes the user-provided script + the stated goal, then:
   - Picks a voice (registered ID, or `--ref-audio` path the user supplied).
   - Decides tag density and injects emotion tags clause-by-clause.
   - Optionally leans on a preset for sampling defaults; freely overrides.
4. Skill writes the tagged script to a temp file (preserving the user's
   original script untouched).
5. Skill shells out: `.venv-mlx/bin/python vo/render.py --script <tmp> ...`.
6. Renderer prints output paths to stdout (JSON line for easy parsing).
7. Skill reports the output triplet back to the user:
   `<out>.wav`, `<out>.words.json`, `<out>.tagged.txt`.

## Renderer interface

`vo/render.py` is both a CLI and a Python module.

### Public Python API

```python
@dataclass
class RenderResult:
    wav_path: Path
    words_path: Path | None      # None if --no-stt
    tagged_path: Path
    voice_id: str | None         # None if ad-hoc ref_audio used
    duration_s: float
    attempts_used: int
    quality_passed: bool

def render(
    script: str,
    out_path: Path,
    *,
    voice: str | None = None,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    preset: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 30,
    speed: float = 1.0,
    max_tokens: int = 4096,
    chunk_length: int = 300,
    max_retries: int = 4,
    max_silence_gap: float = 2.5,
    no_stt: bool = False,
    multi_speaker: bool = False,
    language: str = "en",
    seed: int | None = None,
    tag_mode: str = "auto",        # auto | explicit | none
    anchors: list[list[str]] | None = None,  # optional anchor-phrase list for quality gate (used by reel-sync callers)
) -> RenderResult: ...
```

### CLI surface

| Flag | Default | Purpose |
|---|---|---|
| `--script PATH` | required | Tagged script file. `-` for stdin. |
| `--out PATH` | required | Output WAV path. Sidecars written alongside. |
| `--voice ID` | – | Voice ID from `voices.json`. Mutually exclusive with `--ref-audio`. |
| `--ref-audio PATH` | – | Ad-hoc reference audio. No registry entry required. |
| `--ref-text TEXT` | – | Transcript of the reference. Auto-transcribed via Whisper if omitted. |
| `--preset NAME` | – | Optional starting-point defaults from `presets.json`. |
| `--no-preset` | – | Explicitly ignore any preset defaults. |
| `--temperature FLOAT` | 0.7 | Fish Speech sampling temperature. |
| `--top-p FLOAT` | 0.7 | Top-p sampling. |
| `--top-k INT` | 30 | Top-k sampling. |
| `--speed FLOAT` | 1.0 | Playback speed factor. |
| `--max-tokens INT` | 4096 | Semantic token cap. |
| `--chunk-length INT` | 300 | Long-form chunking. |
| `--max-retries INT` | 4 | Retries on collapse. |
| `--max-silence-gap FLOAT` | 2.5 | Inter-word gap threshold for collapse detection. |
| `--no-stt` | – | Skip Whisper alignment; no `.words.json`. |
| `--multi-speaker` | – | Pass `<\|speaker:i\|>` tokens through to the model. |
| `--language CODE` | `en` | Pass-through language hint. |
| `--seed INT` | – | Reproducibility. |
| `--tag-mode MODE` | `auto` | `auto` / `explicit` / `none`. |
| `--save-voice ID` | – | After a successful `--ref-audio` run, write to `voices.json`. |
| `--label TEXT` | – | Label for `--save-voice` / `--add-voice`. |
| `--notes TEXT` | – | Notes field. |
| `--save-preset NAME` | – | After a successful run, snapshot this run's config into `presets.json`. |
| `--preset-notes TEXT` | – | Notes for the saved preset. |
| `--add-voice ID --audio PATH` | – | Admin: add a voice without rendering. |
| `--add-preset NAME --json STR` | – | Admin: add a preset without rendering. |
| `--transcribe PATH` | – | Admin: print Whisper transcript for an audio file. |

CLI subcommand-style admin paths (`--add-voice`, `--add-preset`,
`--transcribe`) exit after their action; they don't try to generate audio.

### Output sidecars

For an `--out path/foo.wav`:

- `path/foo.wav` — the audio (44.1 kHz PCM).
- `path/foo.words.json` — Whisper word-level timing JSON (unless `--no-stt`):
  ```json
  {
    "duration_s": 28.42,
    "words": [
      {"start": 0.00, "end": 0.45, "text": "Most"},
      {"start": 0.45, "end": 0.92, "text": "content"},
      ...
    ],
    "segments": [{"start": 0.0, "end": 5.4, "text": "Most content ..."}]
  }
  ```
- `path/foo.tagged.txt` — the final tag-annotated script as sent to the model.

The skill reports all three paths to the user. The existing slide-sync reel
renderer can be updated to consume `foo.words.json` directly so it doesn't
re-transcribe.

## Registries

Both registries live alongside `render.py` (i.e. `vo/voices.json` and
`vo/presets.json`). The renderer resolves these paths relative to its own
`__file__`, not the caller's working directory, so it behaves predictably
when invoked from anywhere. Relative `audio` paths inside `voices.json` are
resolved relative to the project root (the directory containing `vo/`).

### `vo/voices.json`

Seeded with the three existing brand voices. Schema:

```json
{
  "version": 1,
  "default": "excited",
  "voices": {
    "<id>": {
      "label": "Human-readable label",
      "audio": "relative/or/absolute/path.m4a",
      "transcript": "Reference transcript.",
      "notes": "Optional notes.",
      "created_at": "2026-05-14T17:00:00Z",
      "created_by": "skill | cli | manual"
    }
  }
}
```

The `default` field is used only if no `--voice` / `--ref-audio` is supplied
and the caller hasn't explicitly opted out. Unknown fields are tolerated.

### `vo/presets.json`

Seeded with five starter presets covering common content goals (fb-reel-hype,
tutorial-explainer, direct-pitch, story-emotional, ad-aggressive-cta). Schema:

```json
{
  "version": 1,
  "presets": {
    "<name>": {
      "voice": "voice_id",                 // optional default voice
      "tag_hints": ["[emphasis]", ...],    // suggestions only
      "tag_density": "low|medium|high",
      "temperature": 0.7,
      "top_p": 0.7,
      "top_k": 30,
      "speed": 1.0,
      "language": "en",
      "notes": "Description of when to use.",
      "created_at": "...",
      "created_by": "skill | cli | manual"
    }
  }
}
```

`tag_hints` are hints to Claude, not enforced injections.

### Flexibility principles (load-bearing)

The skill prompt makes these explicit so Claude doesn't fall back into rigid
behavior:

- **Presets are starting points, not constraints.** Override anything when the
  script demands it. `--preset` + explicit flag combinations work.
- **The voice registry is a convenience, not a fence.** Users can pass any
  `--ref-audio PATH` without ever touching `voices.json`. If `--ref-text` is
  omitted, the renderer auto-transcribes with Whisper.
- **Tag injection is opt-out per-run.** `--tag-mode explicit` respects
  pre-tagged scripts; `--tag-mode none` strips/skips entirely.
- **Registries grow on demand.** Claude offers to save successful ad-hoc
  voices and successful preset configurations after a run (user confirms).
  Never silent writes.

## Feature mapping — Fish Speech S2 Pro features → renderer

| Fish Speech feature | Renderer surface |
|---|---|
| Inline emotion tags (`[whisper]`, `[excited]`, …) | Passed through in script text. |
| Free-form text tags (`[professional broadcast tone]`) | Passed through; Claude is free to invent. |
| Voice cloning | `--voice ID` (registry) or `--ref-audio` + `--ref-text` (ad-hoc). `ref_text` always supplied to the model. |
| Multi-speaker `<\|speaker:i\|>` | `--multi-speaker` flag; tokens passed through. |
| Multi-turn context | Implicit via `--chunk-length` (single-pass long-form). |
| Multilingual (80+ langs) | `--language CODE`. |
| Sampling: T/top_p/top_k | Direct flags. |
| Speed | `--speed`. |
| Long-form chunking | `--chunk-length`. |
| Quality gate / retry | `--max-retries`, `--max-silence-gap`. |
| Whisper word alignment | `.words.json` sidecar (default on). |

## Quality gate (from this session's findings)

After each generation attempt:

1. Run Whisper word-level transcription on the output.
2. Check largest inter-word silence gap. If > `max_silence_gap` (default
   2.5 s) → collapse detected.
3. Locate slide-equivalent "anchor" phrases (configurable, default: skip if
   no anchors supplied by caller). If anchors are supplied and any are
   unfindable → collapse detected.
4. On collapse, perturb temperature (+0.05 per retry) and retry up to
   `max_retries` (default 4).
5. On exhaustion, write the last attempt's WAV but exit non-zero with a clear
   "quality gate failed: <reason>" message naming the failing region.

The slide-sync renderer's existing anchor logic is moved into the renderer so
the same gate guards both ad-hoc VO and reel pipelines.

## Error handling

| Failure mode | Behavior |
|---|---|
| Unknown `--voice` ID | List available IDs, exit 2. |
| `--ref-audio` path missing | Clear path error, exit 2. |
| `--voice` and `--ref-audio` both given | Reject as conflicting, exit 2. |
| Model file missing at expected MLX cache path | Clear "model not installed" message with install hint, exit 3. |
| `mlx_audio.__version__ != "0.4.2"` | Hard refuse with "pin to 0.4.2" message, exit 4. |
| Quality gate exhausted retries | Write last WAV, exit 5 with diagnostic. |
| Whisper alignment failure | Warn, continue without sidecar, exit 0. |

## Testing

Light test suite suitable for the workflow. Tests live in `vo/tests/`.

Unit tests (fast, no model):

1. **Registry loaders** — reject unknown voice IDs / preset names; tolerate
   unknown fields; reject malformed JSON.
2. **Flag parser** — `--voice` and `--ref-audio` mutually exclusive;
   `--ref-text` optional only when `--ref-audio` is given; `--no-preset`
   beats `--preset`.
3. **Tag mode handling** — `explicit` leaves script untouched; `none` strips
   tags; `auto` is a no-op at the renderer level (Claude has already done the
   work in the skill).
4. **Quality gate** — feed canned Whisper output with a fake 5 s gap, assert
   retry triggers; feed clean output, assert single attempt.
5. **Registry mutations** — `--save-voice` after a successful render writes
   the expected entry; `--save-preset` snapshots the run config.

Smoke test (slow, opt-in, not in CI):

- End-to-end run on the `excited` voice with a short canned script;
  re-transcribe the output and assert WER < 0.1 vs the input script.

## Open questions

None at design approval time. Decisions deferred to implementation:

- Exact `presets.json` defaults — five starter entries listed above are
  examples; final wording captured during scaffold.
- Whether to ship a small `vo/voices/` directory containing the m4a sources
  vs. referencing them from `voice samples/`. Default: reference existing
  `voice samples/` to avoid duplication; renderer resolves relative paths
  from the project root.

## Risks

- **mlx-audio 0.4.3 regression** is mitigated by the import-time version
  pin (`feedback_mlx_audio_version_pin.md` memory captures the history).
- **Fish Speech occasional long-form collapse** is mitigated by the quality
  gate with auto-retry. Worst case: exit non-zero with a clear failure rather
  than silently shipping garbage audio.
- **Python 3.14 / mlx version drift** — the `.venv-mlx` pin guards the
  renderer; skill always invokes via the venv's interpreter.
