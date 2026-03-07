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


def test_vowel_purity_stable_vowel():
    """A stable vowel (no drift) should score near 1.0."""
    from voice_core.spanish import score_vowel_purity
    rng = np.random.default_rng(42)
    f1_frames = 750.0 + rng.normal(0, 10, size=20)
    f2_frames = 1250.0 + rng.normal(0, 15, size=20)
    result = score_vowel_purity(f1_frames, f2_frames, expected_vowel="A")
    assert result["purity"] > 0.85
    assert result["diphthongized"] is False


def test_vowel_purity_diphthongized():
    """A drifting vowel (e->eI) should score low and flag diphthongization."""
    from voice_core.spanish import score_vowel_purity
    n = 20
    f1_frames = np.linspace(450, 400, n)
    f2_frames = np.linspace(1900, 2200, n)
    result = score_vowel_purity(f1_frames, f2_frames, expected_vowel="E")
    assert result["purity"] < 0.6
    assert result["diphthongized"] is True


def test_vowel_purity_too_few_frames():
    """Should return None if fewer than 4 frames."""
    from voice_core.spanish import score_vowel_purity
    result = score_vowel_purity(np.array([750.0, 745.0]), np.array([1250.0, 1260.0]), "A")
    assert result is None


def test_spanish_ipa_to_vowel_mapping():
    """Spanish IPA vowel labels should map to our 5-vowel keys."""
    from voice_core.spanish import SPANISH_IPA_TO_VOWEL
    assert SPANISH_IPA_TO_VOWEL["a"] == "A"
    assert SPANISH_IPA_TO_VOWEL["e"] == "E"
    assert SPANISH_IPA_TO_VOWEL["i"] == "I"
    assert SPANISH_IPA_TO_VOWEL["o"] == "O"
    assert SPANISH_IPA_TO_VOWEL["u"] == "U"
    # Stressed variants
    assert SPANISH_IPA_TO_VOWEL.get("a̠") == "A"


def test_spanish_ipa_consonant_targets():
    """Consonant targets should include sheísmo and tap-r labels."""
    from voice_core.spanish import SPANISH_CONSONANT_TARGETS
    assert "ʃ" in SPANISH_CONSONANT_TARGETS
    assert "ʒ" in SPANISH_CONSONANT_TARGETS
    assert "ɾ" in SPANISH_CONSONANT_TARGETS
    assert "j" in SPANISH_CONSONANT_TARGETS
