"""Unit tests for whisperx.conjunctions language data lookups."""

from __future__ import annotations

from whisperx.conjunctions import (
    commas_by_language,
    conjunctions_by_language,
    get_comma,
    get_conjunctions,
)


class TestGetConjunctions:
    def test_known_language_returns_set(self):
        en = get_conjunctions("en")
        assert isinstance(en, set)
        assert "and" in en
        assert "but" in en

    def test_unknown_language_returns_empty_set(self):
        unknown = get_conjunctions("xx")
        assert unknown == set()

    def test_each_language_has_nonempty_set(self):
        for code, words in conjunctions_by_language.items():
            assert isinstance(code, str) and code
            assert len(words) > 0, f"empty conjunctions for {code}"
            assert all(isinstance(w, str) and w for w in words)

    def test_japanese_contains_specific(self):
        ja = get_conjunctions("ja")
        assert "そして" in ja


class TestGetComma:
    def test_default_comma_for_unlisted_language(self):
        assert get_comma("en") == ","

    def test_japanese_comma(self):
        assert get_comma("ja") == "、"

    def test_chinese_comma(self):
        assert get_comma("zh") == "，"

    def test_persian_comma(self):
        assert get_comma("fa") == "،"

    def test_urdu_comma(self):
        assert get_comma("ur") == "،"

    def test_all_commas_in_dict_are_strings(self):
        for code, comma in commas_by_language.items():
            assert isinstance(code, str) and code
            assert isinstance(comma, str) and comma
