from __future__ import annotations

import unittest
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.censor import censor_text


class CensorTests(unittest.TestCase):
    def test_censor_indonesian_profanity(self):
        self.assertEqual(censor_text("kamu anjing banget"), "kamu *** banget")
        self.assertEqual(censor_text("dasar goblok dan tolol"), "dasar *** dan ***")
        self.assertEqual(censor_text("jancok tenan"), "*** tenan")
        self.assertEqual(censor_text("babi lu"), "*** lu")

    def test_censor_english_profanity(self):
        self.assertEqual(censor_text("what the fuck is this shit"), "what the *** is this ***")
        self.assertEqual(censor_text("he is an asshole and bastard"), "he is an *** and ***")
        self.assertEqual(censor_text("stop bitching around"), "stop *** around")

    def test_preserves_benign_words(self):
        self.assertEqual(censor_text("ini kelas programming"), "ini kelas programming")
        self.assertEqual(censor_text("tolong kontak saya di pantai"), "tolong kontak saya di pantai")
        self.assertEqual(censor_text("assassin passes the ball"), "assassin passes the ball")
        self.assertEqual(censor_text("baju saya bersih"), "baju saya bersih")

    def test_disabled_censor(self):
        raw = "kamu goblok dan fuck this"
        self.assertEqual(censor_text(raw, enabled=False), raw)

    def test_empty_and_whitespace(self):
        self.assertEqual(censor_text(""), "")
        self.assertEqual(censor_text("   "), "   ")
        self.assertIsNone(censor_text(None))


if __name__ == "__main__":
    unittest.main()
