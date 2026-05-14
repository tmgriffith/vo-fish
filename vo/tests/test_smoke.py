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
    from vo.render import render
    text = "Most content creators are sitting on hundreds of hours of footage they'll never use."
    out = tmp_path / "smoke.wav"
    result = render(
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
