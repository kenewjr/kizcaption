from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
src = root / "src"
for _p in (str(src), str(root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.main import ensure_workspace, app_directory, bundled_directory
from lumacaption.config import ConfigStore


class FirstLaunchWorkspaceTest(unittest.TestCase):
    def test_first_launch_creates_required_folders_and_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "app"
            test_dir.mkdir()

            # Ensure workspace creates models, output, logs, config.json
            ensure_workspace(test_dir)

            self.assertTrue((test_dir / "models").is_dir(), "models directory must be created")
            self.assertTrue((test_dir / "models" / "silero_vad.onnx").is_file(), "silero_vad.onnx must be copied")
            self.assertTrue((test_dir / "output").is_dir(), "output directory must be created")
            self.assertTrue((test_dir / "output" / "overlay.html").is_file(), "overlay.html must be copied")
            self.assertTrue((test_dir / "logs").is_dir(), "logs directory must be created")
            self.assertTrue((test_dir / "config.json").is_file(), "config.json must be created")
            self.assertTrue((test_dir / "vocabulary.json").is_file(), "vocabulary.json must be created")

            # Validate generated config.json can be loaded cleanly
            store = ConfigStore(test_dir / "config.json")
            cfg, warning = store.load()
            self.assertIsNotNone(cfg)
            self.assertIsNone(warning)
            self.assertEqual(cfg.overlay.theme, "vtuber")

    def test_app_and_bundled_directory_paths(self):
        app_dir = app_directory()
        self.assertTrue(app_dir.is_dir())
        bundled = bundled_directory()
        self.assertTrue(bundled.is_dir())

    def test_ensure_workspace_replaces_truncated_zero_byte_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "app"
            test_dir.mkdir()
            (test_dir / "models").mkdir(parents=True)
            corrupt_file = test_dir / "models" / "silero_vad.onnx"
            corrupt_file.write_bytes(b"")
            self.assertEqual(corrupt_file.stat().st_size, 0)

            ensure_workspace(test_dir)
            self.assertGreater(corrupt_file.stat().st_size, 0, "Corrupt 0-byte file must be restored")


if __name__ == "__main__":
    unittest.main()

