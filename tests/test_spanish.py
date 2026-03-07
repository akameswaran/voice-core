# tests/test_spanish.py
"""Tests for Spanish acoustic analysis module."""

import numpy as np
import pytest


def test_classify_vowel_a():
    from voice_core.spanish import classify_vowel_spanish
    assert classify_vowel_spanish(740.0, 1260.0) == "A"


def test_classify_vowel_e():
    from voice_core.spanish import classify_vowel_spanish
    assert classify_vowel_spanish(460.0, 1880.0) == "E"


def test_classify_vowel_i():
    from voice_core.spanish import classify_vowel_spanish
    assert classify_vowel_spanish(310.0, 2280.0) == "I"


def test_classify_vowel_o():
    from voice_core.spanish import classify_vowel_spanish
    assert classify_vowel_spanish(510.0, 910.0) == "O"


def test_classify_vowel_u():
    from voice_core.spanish import classify_vowel_spanish
    assert classify_vowel_spanish(340.0, 790.0) == "U"


def test_classify_rejects_consonant():
    from voice_core.spanish import classify_vowel_spanish
    assert classify_vowel_spanish(100.0, 5000.0) is None


def test_get_spanish_vowel_norms_loads():
    from voice_core.spanish import get_spanish_vowel_norms
    norms = get_spanish_vowel_norms()
    assert set(norms.keys()) == {"A", "E", "I", "O", "U"}
    assert norms["A"]["f1_mean"] > 0
