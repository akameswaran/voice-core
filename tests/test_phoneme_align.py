# tests/test_phoneme_align.py
"""Tests for phoneme alignment and TextGrid parsing."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest


# Use the real TextGrid from MFA test output
FIXTURE_TG = "/tmp/mfa_test/a1_test.TextGrid"
FIXTURE_WAV = "/tmp/mfa_test/test.wav"


@pytest.fixture
def textgrid_path():
    """Copy fixture TextGrid to a temp dir so tests don't mutate it."""
    if not Path(FIXTURE_TG).exists():
        pytest.skip("MFA test fixture not available")
    with tempfile.TemporaryDirectory() as tmpdir:
        dst = Path(tmpdir) / "test.TextGrid"
        shutil.copy2(FIXTURE_TG, dst)
        yield str(dst)


# --- parse_textgrid tests ---

class TestParseTextgrid:

    def test_returns_words_and_phones(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        assert "words" in result
        assert "phones" in result

    def test_words_have_required_fields(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        for w in result["words"]:
            assert "word" in w
            assert "start" in w
            assert "end" in w
            assert isinstance(w["start"], float)
            assert isinstance(w["end"], float)

    def test_filters_silence_from_words(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        word_texts = [w["word"] for w in result["words"]]
        assert "<eps>" not in word_texts
        assert "sil" not in word_texts

    def test_expected_words(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        word_texts = [w["word"] for w in result["words"]]
        assert word_texts == ["ella", "tiene", "un", "perro", "muy", "lindo"]

    def test_phones_have_required_fields(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        for p in result["phones"]:
            assert "phone" in p
            assert "start" in p
            assert "end" in p
            assert isinstance(p["start"], float)
            assert isinstance(p["end"], float)

    def test_filters_silence_from_phones(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        phone_labels = [p["phone"] for p in result["phones"]]
        assert "sil" not in phone_labels

    def test_phones_have_parent_word(self, textgrid_path):
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        for p in result["phones"]:
            assert "word" in p
            # Every non-silence phone should belong to a word
            assert p["word"] is not None, f"Phone {p['phone']} at {p['start']} has no parent word"

    def test_ella_phones(self, textgrid_path):
        """'ella' should have phones e, ʃ, a — with ʃ showing sheísmo."""
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        ella_phones = [p["phone"] for p in result["phones"] if p["word"] == "ella"]
        assert ella_phones == ["e", "ʃ", "a"]

    def test_perro_phones(self, textgrid_path):
        """'perro' should contain a trill-r."""
        from voice_core.phoneme_align import parse_textgrid
        result = parse_textgrid(textgrid_path)
        perro_phones = [p["phone"] for p in result["phones"] if p["word"] == "perro"]
        assert "r" in perro_phones  # trill


# --- align_async tests ---

class TestAlignAsync:

    @pytest.mark.skipif(
        not Path("/home/ak/miniconda3/envs/mfa/bin/mfa").exists(),
        reason="MFA not installed",
    )
    def test_align_async_produces_textgrid(self):
        from voice_core.phoneme_align import align_async
        if not Path(FIXTURE_WAV).exists():
            pytest.skip("Test WAV not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            tg_path = str(Path(tmpdir) / "async_test.TextGrid")
            result = asyncio.run(
                align_async(FIXTURE_WAV, "ella tiene un perro muy lindo",
                            tg_path, language="es")
            )
            assert Path(result).exists()
            assert result == tg_path

    @pytest.mark.skipif(
        not Path("/home/ak/miniconda3/envs/mfa/bin/mfa").exists(),
        reason="MFA not installed",
    )
    def test_align_async_error_raises(self):
        from voice_core.phoneme_align import align_async
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_wav = str(Path(tmpdir) / "nonexistent.wav")
            tg_path = str(Path(tmpdir) / "bad.TextGrid")
            with pytest.raises(RuntimeError, match="MFA alignment failed"):
                asyncio.run(
                    align_async(bad_wav, "hello",
                                tg_path, language="en")
                )
