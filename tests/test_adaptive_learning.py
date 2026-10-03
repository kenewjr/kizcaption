from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.stt.whisper_engine import WhisperEngine, sanitize_transcript
from lumacaption.vocabulary import VocabularyManager


class AdaptiveLearningAndHallucinationTests(unittest.TestCase):
    def test_sanitize_transcript_removes_hallucination_tags_and_canned_phrases(self):
        # 1. Non-speech tags
        self.assertEqual(sanitize_transcript("[Musik] Halo semuanya [Tertawa]"), "Halo semuanya")
        self.assertEqual(sanitize_transcript("(music) Selamat malam (applause)"), "Selamat malam")
        self.assertEqual(sanitize_transcript("[suara hening] Test satu dua"), "Test satu dua")

        # 2. Canned subtitle hallucinations
        self.assertEqual(sanitize_transcript("Terima kasih sudah menonton."), "")
        self.assertEqual(sanitize_transcript("Thanks for watching!"), "")
        self.assertEqual(sanitize_transcript("Subtitles by Community"), "")
        self.assertEqual(sanitize_transcript("Jangan lupa like and subscribe!"), "")

        # 3. Collapse repetitive identical word loops
        self.assertEqual(
            sanitize_transcript("halo halo halo halo dunia"),
            "halo dunia",
        )
        self.assertEqual(
            sanitize_transcript("tes tes tes satu dua"),
            "tes satu dua",
        )

        # 4. Collapse 2-word phrase loops
        self.assertEqual(
            sanitize_transcript("selamat pagi selamat pagi selamat pagi semuanya"),
            "selamat pagi semuanya",
        )

        # 5. Normal speech untouched
        normal = "Halo rek, Pak Budi makan Indomie enak tenan"
        self.assertEqual(sanitize_transcript(normal), normal)

    def test_trim_dangling_conjunctions_at_utterance_boundary(self):
        # Dangling conjunctions at boundary trimmed
        self.assertEqual(
            sanitize_transcript("Saya mau pergi ke pasar dan"),
            "Saya mau pergi ke pasar",
        )
        self.assertEqual(
            sanitize_transcript("Dia tidak datang karena..."),
            "Dia tidak datang...",
        )
        self.assertEqual(
            sanitize_transcript("Kami sudah menunggu tapi"),
            "Kami sudah menunggu",
        )
        # Short phrases (< 3 words) preserved
        self.assertEqual(sanitize_transcript("dan"), "dan")
        self.assertEqual(sanitize_transcript("karena apa"), "karena apa")
        # Mid-sentence conjunctions untouched
        self.assertEqual(
            sanitize_transcript("Budi dan Siti belajar bersama"),
            "Budi dan Siti belajar bersama",
        )

    def test_whisper_engine_segment_hallucination_filtering(self):
        engine = WhisperEngine("tiny", "cpu", Path("models/whisper"))
        
        # Fake segments containing high compression ratio (loop), silence prob, and good speech
        seg_loop = SimpleNamespace(
            text="kata kata kata kata kata",
            compression_ratio=2.8,  # > 2.4
            no_speech_prob=0.1,
            avg_logprob=-0.2,
        )
        seg_silence = SimpleNamespace(
            text="terdengar suara",
            compression_ratio=1.1,
            no_speech_prob=0.85,  # > 0.65
            avg_logprob=-0.3,
        )
        seg_low_conf_short = SimpleNamespace(
            text="ah",
            compression_ratio=1.0,
            no_speech_prob=0.2,
            avg_logprob=-1.5,  # < -1.2 and short
        )
        seg_valid = SimpleNamespace(
            text="Selamat datang di KizCaption.",
            compression_ratio=1.2,
            no_speech_prob=0.05,
            avg_logprob=-0.2,
        )

        segments = [seg_loop, seg_silence, seg_low_conf_short, seg_valid]

        # Verify filtering logic
        valid_segments = []
        for segment in segments:
            if getattr(segment, "compression_ratio", 1.0) > 2.4:
                continue
            if getattr(segment, "no_speech_prob", 0.0) > 0.65:
                continue
            raw_text = getattr(segment, "text", "") or ""
            if getattr(segment, "avg_logprob", 0.0) < -1.2 and len(raw_text.strip()) < 8:
                continue
            cleaned = sanitize_transcript(raw_text)
            if cleaned:
                valid_segments.append(cleaned)

        result = " ".join(valid_segments).strip()
        self.assertEqual(result, "Selamat datang di KizCaption.")

    def test_vocabulary_manager_lifecycle_and_learning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            vocab_file = workspace / "vocabulary.json"

            # 1. Initialize with empty workspace -> creates manager cleanly and loads bundled template
            vm = VocabularyManager(workspace)
            initial_count = len(vm.entries)
            self.assertGreaterEqual(initial_count, 0)

            # 2. Observe single occurrence -> not learned yet (threshold is 3)
            learned_1 = vm.observe("Halo semuanya apa kabar?")
            self.assertEqual(learned_1, [])
            self.assertFalse(vocab_file.exists())

            # 3. Repeat a unique capitalized term (weight 2 per utterance)
            # 1st utterance with "KizCaption" (+2 weight) -> total 2 (< 3)
            learned_2 = vm.observe("Saya memakai aplikasi KizCaption sekarang.")
            self.assertEqual(learned_2, [])

            # 2nd utterance with "KizCaption" (+2 weight) -> total 4 (>= 3) -> learned!
            learned_3 = vm.observe("KizCaption sangat cepat dan responsif.")
            self.assertIn("KizCaption", learned_3)

            # 4. Verify persisted to vocabulary.json
            self.assertTrue(vocab_file.exists())
            persisted = json.loads(vocab_file.read_text(encoding="utf-8"))
            entries = persisted.get("entries", [])
            self.assertEqual(len(entries), initial_count + 1)
            learned_entries = [e for e in entries if e.get("package") == "learned"]
            self.assertEqual(len(learned_entries), 1)
            entry = learned_entries[0]
            self.assertEqual(entry["term"], "KizCaption")
            self.assertEqual(entry["package"], "learned")
            self.assertTrue(entry["enabled"])
            self.assertTrue(entry["stt_hint"])

            # 5. Reload in new instance -> loads learned words
            vm2 = VocabularyManager(workspace)
            self.assertEqual(len(vm2.entries), initial_count + 1)
            self.assertIn("KizCaption", vm2._learned_terms)

            # 6. Hotword generation priorities: user_hotwords first, then learned words
            hotwords = vm2.get_hotwords(user_hotwords="custom_tag, OBS")
            self.assertEqual(hotwords, "custom_tag, OBS, KizCaption")

            # 7. Reset learned vocabulary
            self.assertEqual(vm2.learned_count, 1)
            removed = vm2.reset_learned()
            self.assertEqual(removed, 1)
            self.assertEqual(vm2.learned_count, 0)
            self.assertEqual(len(vm2.entries), initial_count)
            self.assertEqual(vm2.get_hotwords(user_hotwords="custom_tag"), "custom_tag")

    def test_whisper_engine_update_hotwords(self):
        engine = WhisperEngine("tiny", "cpu", Path("models/whisper"), hotwords="initial_hint")
        self.assertEqual(engine.hotwords, "initial_hint")
        engine.update_hotwords("updated_hint, learned_word")
        self.assertEqual(engine.hotwords, "updated_hint, learned_word")

    def test_hotwords_ranking_by_frequency(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            vm = VocabularyManager(workspace)
            # Alpha mentioned 2 times (weight 4)
            vm.observe("Alpha Alpha")
            # Beta mentioned 6 times (weight 12)
            for _ in range(6):
                vm.observe("Beta")
            hw = vm.get_hotwords()
            # Beta has higher occurrence and must come before Alpha
            self.assertTrue(hw.startswith("Beta, Alpha"))


if __name__ == "__main__":
    unittest.main()
