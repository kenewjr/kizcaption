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
    def test_setup_logging_creates_single_file_all_levels(self):
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

                # Verify single consolidated log file matching kizcaption-{datetime}.log
                all_logs = list((app_dir / "logs").glob("kizcaption-*.log"))
                self.assertEqual(len(all_logs), 1, "Exactly one kizcaption log file must exist")

                content = all_logs[0].read_text(encoding="utf-8")
                self.assertIn("SESSION STARTED", content)
                self.assertIn("[DEBUG] [lumacaption] Debug test message", content)
                self.assertIn("[INFO] [lumacaption] Info test message", content)
                self.assertIn("[WARNING] [lumacaption] Warning test message", content)
                self.assertIn("[ERROR] [lumacaption] Error test message", content)
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

                log_file = list((app_dir / "logs").glob("kizcaption-*.log"))[0]
                content = log_file.read_text(encoding="utf-8")
                self.assertNotIn("audio_level", content)
                self.assertNotIn("vad_probability", content)
                self.assertIn("Suara terdeteksi", content)
            finally:
                for handler in list(logger.handlers):
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
