from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
src = root / "src"
for _p in (str(src), str(root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.main import setup_logging
from lumacaption.pipeline import CaptionPipeline
from lumacaption.config import AppConfig


class LoggingAndQueueTests(unittest.TestCase):
    def test_setup_logging_creates_files_and_handles_4_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            logger = setup_logging(app_dir)
            try:
                logger.debug("Debug test message")
                logger.info("Info test message")
                logger.warning("Warning test message")
                logger.error("Error test message")

                for handler in list(logger.handlers):
                    handler.flush()

                # Verify 4 distinct level log files matching {level}log-{datetime}.log
                debug_logs = list((app_dir / "logs").glob("debuglog-*.log"))
                info_logs = list((app_dir / "logs").glob("infolog-*.log"))
                warning_logs = list((app_dir / "logs").glob("warninglog-*.log"))
                error_logs = list((app_dir / "logs").glob("errorlog-*.log"))

                self.assertEqual(len(debug_logs), 1, "Exactly one debuglog file must exist")
                self.assertEqual(len(info_logs), 1, "Exactly one infolog file must exist")
                self.assertEqual(len(warning_logs), 1, "Exactly one warninglog file must exist")
                self.assertEqual(len(error_logs), 1, "Exactly one errorlog file must exist")

                debug_content = debug_logs[0].read_text(encoding="utf-8")
                info_content = info_logs[0].read_text(encoding="utf-8")
                warning_content = warning_logs[0].read_text(encoding="utf-8")
                error_content = error_logs[0].read_text(encoding="utf-8")

                self.assertIn("SESSION STARTED", debug_content)
                self.assertIn("Debug test message", debug_content)
                self.assertNotIn("Error test message", debug_content)

                self.assertIn("SESSION STARTED", info_content)
                self.assertIn("Info test message", info_content)
                self.assertNotIn("Debug test message", info_content)

                self.assertIn("SESSION STARTED", warning_content)
                self.assertIn("Warning test message", warning_content)
                self.assertNotIn("Debug test message", warning_content)

                self.assertIn("SESSION STARTED", error_content)
                self.assertIn("Error test message", error_content)
                self.assertNotIn("Debug test message", error_content)
            finally:
                for handler in list(logger.handlers):
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)

    def test_utterances_queue_capacity(self):
        pipeline = CaptionPipeline(
            AppConfig(),
            ".",
            lambda _ev: None,
        )
        self.assertFalse(pipeline.running)

    def test_pipeline_telemetry_not_spammed_to_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            logger = setup_logging(app_dir)
            try:
                pipeline = CaptionPipeline(
                    AppConfig(),
                    app_dir,
                    lambda _ev: None,
                )
                pipeline._emit("audio_level", "Audio level", {"rms": 0.05, "peak": 0.1})
                pipeline._emit("vad_probability", "VAD probability", {"probability": 0.8})
                pipeline._emit("speech", "Suara terdeteksi")

                for handler in list(logger.handlers):
                    handler.flush()

                debug_log = list((app_dir / "logs").glob("debuglog-*.log"))[0]
                debug_content = debug_log.read_text(encoding="utf-8")
                self.assertNotIn("audio_level", debug_content)
                self.assertNotIn("vad_probability", debug_content)
                self.assertIn("Suara terdeteksi", debug_content)
            finally:
                for handler in list(logger.handlers):
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
