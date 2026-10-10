from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.config import AppConfig, TargetConfig
from lumacaption.pipeline import CaptionPipeline, PipelineEvent
from lumacaption.stt.whisper_engine import Transcript


class PipelineLogReviewTests(unittest.TestCase):
    def test_default_config_is_on(self):
        config = AppConfig()
        self.assertTrue(config.translation_review_log)

    def test_append_review_log_creates_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            pipeline = CaptionPipeline(
                AppConfig(translation_review_log=True),
                app_dir,
                lambda _ev: None,
            )

            pipeline._append_review_log(
                detected_lang="id",
                source_code="ind_Latn",
                source_text="halo semuanya",
                translations={"English": "hello everyone", "Japanese": "皆さんこんにちは"},
                stt_ms=45.2,
                mt_ms=18.7,
            )

            jsonl_file = app_dir / "logs" / "translation_review.jsonl"
            self.assertTrue(jsonl_file.is_file())

            lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

            entry = json.loads(lines[0])
            self.assertEqual(entry["detected_language"], "id")
            self.assertEqual(entry["source_code"], "ind_Latn")
            self.assertEqual(entry["source_text"], "halo semuanya")
            self.assertEqual(entry["translations"], {"English": "hello everyone", "Japanese": "皆さんこんにちは"})
            self.assertEqual(entry["stt_ms"], 45.2)
            self.assertEqual(entry["mt_ms"], 18.7)
            self.assertIn("ts", entry)

    def test_rotation_over_5mb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            log_dir = app_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / "translation_review.jsonl"

            # Pre-populate with > 5MB file
            with open(path, "wb") as f:
                f.seek(5_242_880 + 10)
                f.write(b"x")

            pipeline = CaptionPipeline(
                AppConfig(translation_review_log=True),
                app_dir,
                lambda _ev: None,
            )

            pipeline._append_review_log(
                detected_lang="id",
                source_code="ind_Latn",
                source_text="test rotasi",
                translations={"English": "rotation test"},
                stt_ms=10.0,
                mt_ms=5.0,
            )

            backup = log_dir / "translation_review.jsonl.1"
            self.assertTrue(backup.is_file(), "Backup file .jsonl.1 must exist after rotation")
            self.assertGreater(backup.stat().st_size, 5_242_880)

            # Main file should only contain the new single record
            self.assertTrue(path.is_file())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["source_text"], "test rotasi")

    def test_write_failure_does_not_raise_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            pipeline = CaptionPipeline(
                AppConfig(translation_review_log=True),
                app_dir,
                lambda _ev: None,
            )

            with patch("builtins.open", side_effect=OSError("Disk write error")), \
                 patch("lumacaption.pipeline.logger.warning") as mock_warn:
                # Must not raise
                pipeline._append_review_log(
                    detected_lang="id",
                    source_code="ind_Latn",
                    source_text="halo",
                    translations={"English": "hello"},
                    stt_ms=1.0,
                    mt_ms=1.0,
                )
                mock_warn.assert_called_once()
                self.assertIn("Gagal menulis translation review log", mock_warn.call_args[0][0])

    def test_translations_summary_and_disabled_review_log(self):
        emitted_events = []

        def on_event(ev: PipelineEvent):
            emitted_events.append(ev)

        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            # Default config: translation_review_log is False
            config = AppConfig(
                source_language="id",
                targets=[TargetConfig(language="English")],
                translation_review_log=False,
            )
            pipeline = CaptionPipeline(
                config,
                app_dir,
                on_event,
            )

            # Emit translations directly to verify 21.2 format
            translations = {"English": "hello world", "Japanese": "こんにちは世界"}
            summary = " | ".join(f"{lang}={text}" for lang, text in translations.items())
            pipeline._emit("translations", summary, translations)

            self.assertEqual(len(emitted_events), 1)
            self.assertEqual(emitted_events[0].kind, "translations")
            self.assertEqual(emitted_events[0].message, "English=hello world | Japanese=こんにちは世界")
            self.assertEqual(emitted_events[0].data, translations)

            # Confirm no review log file was generated
            jsonl_file = app_dir / "logs" / "translation_review.jsonl"
            self.assertFalse(jsonl_file.exists())

    def test_export_and_clear_review_log(self):
        from lumacaption.ui.control_panel import ControlPanel
        import lumacaption.ui.control_panel as cp_module

        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            logs_dir = app_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / "translation_review.jsonl"
            backup_file = logs_dir / "translation_review.jsonl.1"
            dest_file = app_dir / "exported.jsonl"

            log_file.write_text('{"test": 1}\n', encoding="utf-8")
            backup_file.write_text('{"backup": 1}\n', encoding="utf-8")

            dummy = MagicMock()
            dummy.app_dir = app_dir
            dummy.config = AppConfig()
            dummy.root = MagicMock()
            dummy.t = lambda key, **kw: key

            with patch.object(cp_module.filedialog, "asksaveasfilename", return_value=str(dest_file)), \
                 patch.object(cp_module.messagebox, "showinfo") as mock_info:
                ControlPanel._export_review_log(dummy)
                self.assertTrue(dest_file.is_file())
                self.assertEqual(dest_file.read_text(encoding="utf-8"), '{"test": 1}\n')
                mock_info.assert_called_once()

            # Now test clearing
            ControlPanel._clear_review_log(dummy)
            self.assertFalse(log_file.exists())
            self.assertFalse(backup_file.exists())


if __name__ == "__main__":
    unittest.main()

