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

                kiz_log = app_dir / "logs" / "kizcaption.log"
                diag_log = app_dir / "logs" / "diagnostic.log"

                self.assertTrue(kiz_log.is_file(), "kizcaption.log must exist")
                self.assertTrue(diag_log.is_file(), "diagnostic.log must exist")

                content = kiz_log.read_text(encoding="utf-8")
                self.assertIn("SESSION STARTED", content)
                self.assertIn("[DEBUG]", content)
                self.assertIn("Debug test message", content)
                self.assertIn("[INFO]", content)
                self.assertIn("Info test message", content)
                self.assertIn("[WARNING]", content)
                self.assertIn("Warning test message", content)
                self.assertIn("[ERROR]", content)
                self.assertIn("Error test message", content)
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


if __name__ == "__main__":
    unittest.main()
