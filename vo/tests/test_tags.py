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
