# tests/test_spanish_vowel_norms.py
"""Tests for Spanish vowel norms data file."""

import json
from pathlib import Path

NORMS_PATH = Path(__file__).resolve().parent.parent / "src" / "voice_core" / "data" / "spanish_vowel_norms.json"

EXPECTED_VOWELS = {"A", "E", "I", "O", "U"}


def test_file_exists_and_loads():
    assert NORMS_PATH.exists(), f"Missing {NORMS_PATH}"
    data = json.loads(NORMS_PATH.read_text())
    assert "vowels" in data


def test_has_all_five_vowels():
    data = json.loads(NORMS_PATH.read_text())
    assert set(data["vowels"].keys()) == EXPECTED_VOWELS


def test_each_vowel_has_required_fields():
    data = json.loads(NORMS_PATH.read_text())
    required = {"f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std", "label"}
    for vowel, vals in data["vowels"].items():
        missing = required - set(vals.keys())
        assert not missing, f"{vowel} missing: {missing}"


def test_formant_values_are_plausible():
    """F1 should be 250-900 Hz, F2 800-2500 Hz, F3 2000-3500 Hz."""
    data = json.loads(NORMS_PATH.read_text())
    for vowel, v in data["vowels"].items():
        assert 250 < v["f1_mean"] < 900, f"{vowel} F1={v['f1_mean']}"
        assert 800 < v["f2_mean"] < 2500, f"{vowel} F2={v['f2_mean']}"
        assert 2000 < v["f3_mean"] < 3500, f"{vowel} F3={v['f3_mean']}"
        assert v["f1_std"] > 0
        assert v["f2_std"] > 0
        assert v["f3_std"] > 0
