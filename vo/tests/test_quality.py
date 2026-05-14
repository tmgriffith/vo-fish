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
