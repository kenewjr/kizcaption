from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.model_manager import (
    CATALOG,
    get_persistent_models_dir,
    get_candidate_model_dirs,
    find_model_locally,
    adopt_model,
    detect_and_link_models,
    inspect_model,
)


class ModelDetectionTests(unittest.TestCase):
    def test_persistent_and_candidate_dirs(self):
        pdir = get_persistent_models_dir()
        self.assertIsInstance(pdir, Path)
        self.assertTrue(pdir.exists())

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp)
            candidates = get_candidate_model_dirs(app_dir)
            self.assertTrue(len(candidates) >= 2)
            # app_dir models should be in candidates
            app_models_resolved = (app_dir / "models").resolve()
            # If app_dir/models is created, it will be added
            (app_dir / "models").mkdir(parents=True, exist_ok=True)
            candidates_with_models = get_candidate_model_dirs(app_dir)
            self.assertIn(app_models_resolved, candidates_with_models)

    def test_find_and_adopt_silero(self):
        with tempfile.TemporaryDirectory() as tmp_source, tempfile.TemporaryDirectory() as tmp_target:
            src_dir = Path(tmp_source)
            tgt_dir = Path(tmp_target)

            # Create dummy silero in source
            silero_file = src_dir / "silero_vad.onnx"
            silero_file.write_bytes(b"x" * 1_500_000)

            # Find model locally
            state, found_path, loc = find_model_locally("silero", search_dirs=[src_dir])
            self.assertEqual(state, "Tersedia lokal")
            self.assertIsNotNone(found_path)

            # Adopt into target
            ok = adopt_model("silero", found_path, tgt_dir / "models")
            self.assertTrue(ok)
            self.assertTrue((tgt_dir / "models" / "silero_vad.onnx").is_file())

            # Verify inspect_model on target
            insp_state, insp_path = inspect_model("silero", tgt_dir / "models")
            self.assertEqual(insp_state, "Tersedia lokal")

    def test_detect_and_link_whisper_repo(self):
        with tempfile.TemporaryDirectory() as tmp_old, tempfile.TemporaryDirectory() as tmp_new:
            old_app = Path(tmp_old)
            new_app = Path(tmp_new)

            # Mock a completed whisper model in old_app/models/whisper/models--Systran--faster-whisper-tiny
            repo_name = "models--Systran--faster-whisper-tiny"
            snapshot_dir = old_app / "models" / "whisper" / repo_name / "snapshots" / "mock_hash"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (snapshot_dir / "model.bin").write_bytes(b"model_data")
            (snapshot_dir / "config.json").write_bytes(b"config_data")
            (snapshot_dir / "vocabulary.txt").write_bytes(b"vocab_data")

            # Check new_app doesn't have it yet
            state_before, _ = inspect_model("tiny", new_app / "models" / "whisper")
            self.assertEqual(state_before, "Belum ada")

            # Run detect_and_link_models with old_app as custom source
            adopted = detect_and_link_models(new_app, custom_source_dir=old_app / "models")
            self.assertTrue(any(a["key"] == "tiny" for a in adopted))

            # Now verify inspect_model on new_app returns Tersedia lokal
            state_after, path_after = inspect_model("tiny", new_app / "models" / "whisper")
            self.assertEqual(state_after, "Tersedia lokal")
            self.assertIsNotNone(path_after)
            self.assertTrue((path_after / "model.bin").is_file())


if __name__ == "__main__":
    unittest.main()
