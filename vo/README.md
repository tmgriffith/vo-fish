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
