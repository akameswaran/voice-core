#!/usr/bin/env python3
"""Persistent MFA alignment server — keeps aligner warm for fast repeated calls.

Run from the MFA conda env:
  /home/ak/miniconda3/envs/mfa/bin/python voice-core/scripts/mfa-server.py

Listens on localhost:9010 (not exposed externally).
POST /align  {wav_path, transcript, language}  →  {words, phones}

Startup loads acoustic model + lexicon (~2.5s one-time cost).
Each alignment: ~100-200ms (vs ~5s for subprocess align_one).
"""

import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Ensure MFA env's packages are available
MFA_ENV = Path("/home/ak/miniconda3/envs/mfa")
os.environ["PATH"] = f"{MFA_ENV / 'bin'}:{os.environ.get('PATH', '')}"

from kalpy.aligner import KalpyAligner, Segment
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler
from kalpy.utterance import Utterance as KalpyUtterance
import pywrapfst

from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.online.alignment import tokenize_utterance_text
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer

PORT = int(os.environ.get("MFA_SERVER_PORT", "9010"))

# ── One-time setup ───────────────────────────────────────────────────────

MODELS_DIR = Path("/home/ak/Documents/MFA/pretrained_models/acoustic")
DICT_CACHE = Path("/home/ak/Documents/MFA/extracted_models/dictionary")

MFA_CONFIGS = {
    "es": {
        "acoustic": MODELS_DIR / "spanish_mfa.zip",
        "dict_dir": DICT_CACHE / "spanish_latin_america_mfa",
    },
    "en": {
        "acoustic": MODELS_DIR / "english_mfa.zip",
        "dict_dir": DICT_CACHE / "english_mfa",
    },
}

_aligners = {}  # language → {aligner, tokenizer, acoustic_model, cmvn_computer}


def _load_language(lang: str) -> dict:
    """Load acoustic model, lexicon, and aligner for a language."""
    t0 = time.perf_counter()
    cfg = MFA_CONFIGS[lang]

    am = AcousticModel(str(cfg["acoustic"]))
    lc = LexiconCompiler(
        disambiguation=False,
        silence_probability=am.parameters["silence_probability"],
        initial_silence_probability=am.parameters["initial_silence_probability"],
        final_silence_correction=am.parameters["final_silence_correction"],
        final_non_silence_correction=am.parameters["final_non_silence_correction"],
        silence_phone=am.parameters["optional_silence_phone"],
        oov_phone=am.parameters["oov_phone"],
        position_dependent_phones=am.parameters["position_dependent_phones"],
        phones=am.parameters["non_silence_phones"],
        ignore_case=True,
    )
    dict_dir = cfg["dict_dir"]
    lc.load_l_from_file(dict_dir / "L.fst")
    lc.load_l_align_from_file(dict_dir / "L_align.fst")
    lc.word_table = pywrapfst.SymbolTable.read_text(str(dict_dir / "words.txt"))
    lc.phone_table = pywrapfst.SymbolTable.read_text(str(dict_dir / "phones.txt"))

    aligner = KalpyAligner(am, lc)
    tokenizer = generate_language_tokenizer(am.language)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[mfa-server] loaded {lang} in {elapsed:.0f}ms")

    return {
        "aligner": aligner,
        "acoustic_model": am,
        "lexicon_compiler": lc,
        "tokenizer": tokenizer,
        "cmvn_computer": CmvnComputer(),
    }


def _get_aligner(lang: str) -> dict:
    if lang not in _aligners:
        _aligners[lang] = _load_language(lang)
    return _aligners[lang]


def _align(wav_path: str, transcript: str, language: str = "es") -> dict:
    """Align a single utterance and return TextGrid-like word/phone data."""
    t0 = time.perf_counter()
    ctx = _get_aligner(language)

    # Write transcript to temp .lab file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lab",
                                      dir=Path(wav_path).parent,
                                      delete=False) as f:
        f.write(transcript)
        txt_path = f.name

    try:
        file_name = Path(wav_path).stem
        file = FileData.parse_file(file_name, Path(wav_path), Path(txt_path), "", 0)

        utterances = []
        for utt_data in file.utterances:
            seg = Segment(Path(wav_path), utt_data.begin, utt_data.end, utt_data.channel)
            normalized_text = tokenize_utterance_text(
                utt_data.text,
                ctx["lexicon_compiler"],
                ctx["tokenizer"],
                None,  # no g2p model
                language=ctx["acoustic_model"].language,
            )
            utt = KalpyUtterance(seg, normalized_text)
            utt.generate_mfccs(ctx["acoustic_model"].mfcc_computer)
            utterances.append(utt)

        cmvn = ctx["cmvn_computer"].compute_cmvn_from_features(
            [u.mfccs for u in utterances]
        )

        file_ctm = HierarchicalCtm([])
        for utt in utterances:
            utt.apply_cmvn(cmvn)
            ctm = ctx["aligner"].align_utterance(utt)
            file_ctm.word_intervals.extend(ctm.word_intervals)

        # Export to temp TextGrid, then parse it
        with tempfile.NamedTemporaryFile(suffix=".TextGrid", delete=False) as tg_tmp:
            tg_path = tg_tmp.name

        file_ctm.export_textgrid(
            Path(tg_path),
            file_duration=file.wav_info.duration,
            output_format="short_textgrid",
        )

        # Parse TextGrid using voice-core's parser
        from praatio import textgrid as tg_mod
        tg = tg_mod.openTextgrid(tg_path, includeEmptyIntervals=False)

        words_tier = tg.getTier("words")
        words = []
        for start, end, label in words_tier.entries:
            if label in ("<eps>", "sil", ""):
                continue
            words.append({"word": label, "start": float(start), "end": float(end)})

        phones_tier = tg.getTier("phones")
        phones = []
        for start, end, label in phones_tier.entries:
            if label in ("sil", ""):
                continue
            mid = (float(start) + float(end)) / 2
            parent_word = None
            for w in words:
                if w["start"] <= mid <= w["end"]:
                    parent_word = w["word"]
                    break
            phones.append({
                "phone": label,
                "start": float(start),
                "end": float(end),
                "word": parent_word,
            })

        os.unlink(tg_path)

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[mfa-server] aligned {len(words)} words in {elapsed:.0f}ms")

        return {"words": words, "phones": phones, "align_ms": round(elapsed, 1)}

    finally:
        os.unlink(txt_path)


# ── HTTP server ──────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path != "/align":
            self.send_response(404)
            self.end_headers()
            return

        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}

        wav_path = body.get("wav_path", "")
        transcript = body.get("transcript", "")
        language = body.get("language", "es")

        if not wav_path or not transcript:
            self._json(400, {"error": "wav_path and transcript required"})
            return

        try:
            result = _align(wav_path, transcript, language)
            self._json(200, result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            langs = list(_aligners.keys())
            self._json(200, {"status": "ok", "languages_loaded": langs})
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    # Pre-load Spanish on startup
    print(f"[mfa-server] starting on :{PORT}")
    _get_aligner("es")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[mfa-server] ready on http://127.0.0.1:{PORT}")
    server.serve_forever()
