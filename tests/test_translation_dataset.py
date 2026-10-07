from __future__ import annotations

from pathlib import Path
import sys
import unittest

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.mt.slang_normalizer import (
    normalize_indonesian_slang,
    split_into_sentences,
)
from lumacaption.mt.conversational_data import (
    CONVERSATIONAL_MAP,
    CONVERSATIONAL_ALIASES,
)


class TranslationNormalizerTests(unittest.TestCase):
    def test_slang_normalization_pronouns_and_adverbs(self):
        text = "gue lagi otw ke rumah lu nih, santuy aja kawan gak usah buru-buru"
        norm = normalize_indonesian_slang(text)
        self.assertIn("saya", norm)
        self.assertIn("kamu", norm)
        self.assertIn("sedang di jalan", norm)
        self.assertIn("santai saja", norm)
        self.assertIn("tidak usah", norm)

    def test_slang_normalization_gaming_and_streamer(self):
        text = "mabar game bareng yuk, gaskeun bro kita ratain musuhnya"
        norm = normalize_indonesian_slang(text)
        self.assertIn("bermain game bersama", norm)
        self.assertIn("ayo kawan,", norm)
        self.assertIn("kalahkan musuhnya", norm)

    def test_slang_normalization_common_idioms(self):
        text = "mager banget gue hari ini sumpah, bikin pusing aja masalah ini gak kelar-kelar"
        norm = normalize_indonesian_slang(text)
        self.assertIn("saya sangat malas", norm)
        self.assertIn(", sumpah", norm)
        self.assertIn("sangat membingungkan", norm)
        self.assertIn("tidak pernah selesai", norm)

    def test_slang_normalization_kepo_and_baper(self):
        text = "jangan baper dong kali, kepo banget sih lu jadi orang"
        norm = normalize_indonesian_slang(text)
        self.assertIn("jangan tersinggung", norm)
        self.assertIn("kamu terlalu ingin tahu", norm)

    def test_split_into_sentences_preserves_clauses(self):
        text = "Udah makan belom kamu? Kalo belom ayo cari makan!"
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "Udah makan belom kamu?")
        self.assertEqual(sents[1], "Kalo belom ayo cari makan!")

    def test_conversational_map_has_essential_greetings(self):
        for greeting in ("halo", "hai", "apa kabar", "baik", "terima kasih", "sama-sama", "selamat pagi", "selamat malam", "maaf"):
            self.assertIn(greeting, CONVERSATIONAL_MAP)
            self.assertIn("English", CONVERSATIONAL_MAP[greeting])

    def test_conversational_aliases_resolution(self):
        self.assertEqual(CONVERSATIONAL_ALIASES["helo"], "halo")
        self.assertEqual(CONVERSATIONAL_ALIASES["makasih"], "terima kasih")
        self.assertEqual(CONVERSATIONAL_ALIASES["gapapa"], "tidak apa-apa")
        self.assertEqual(CONVERSATIONAL_ALIASES["sama2"], "sama-sama")


if __name__ == "__main__":
    unittest.main()
