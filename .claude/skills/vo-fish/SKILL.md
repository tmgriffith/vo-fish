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
