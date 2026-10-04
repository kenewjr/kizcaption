from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lumacaption.model_manager import (
    CATALOG,
    complete,
    inspect_model,
    resource_details,
    hardware_info,
    evaluate_vram_safety,
    model_resource_table_rows,
)


class ModelManagerTest(unittest.TestCase):
    def test_catalog_completeness(self):
        expected_keys = {"tiny", "base", "small", "medium", "large-v3-turbo", "nllb", "silero"}
        self.assertTrue(expected_keys.issubset(set(CATALOG.keys())))
        for key, info in CATALOG.items():
            self.assertEqual(info.key, key)
            self.assertGreater(info.disk_mib, 0)
            self.assertTrue(len(info.files) > 0)

    def test_inspect_missing_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            # Missing model
            state, path = inspect_model("tiny", cache)
            self.assertEqual(state, "Belum ada")
            self.assertIsNone(path)

            # Silero available when file is present
            silero_file = cache / "silero_vad.onnx"
            silero_file.write_bytes(b"dummy")
            state_silero, path_silero = inspect_model("silero", cache)
            self.assertEqual(state_silero, "Tersedia lokal")
            self.assertEqual(path_silero, cache)

    def test_resource_details_formatting(self):
        for key in ("small", "large-v3-turbo", "nllb", "silero"):
            res_cuda = resource_details(key, "cuda", threads=4, beam=3, targets=2)
            self.assertIn("Disk:", res_cuda)
            self.assertIn("Konfigurasi:", res_cuda)
            res_cpu = resource_details(key, "cpu", threads=4, beam=1, targets=1)
            self.assertIn("VRAM", res_cpu)

    def test_hardware_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = hardware_info(Path(tmp))
            self.assertIn("Disk bebas", info)
            self.assertIn("CPU", info)

    def test_evaluate_vram_safety(self):
        # CPU mode returns cpu status
        status, badge, detail = evaluate_vram_safety("small", "mijuanlo/nllb-200-distilled-600M-ct2-int8", "cpu", "cpu")
        self.assertEqual(status, "cpu")
        self.assertIn("Mode CPU", badge)

        # Abundant VRAM (8 GB)
        with patch("lumacaption.model_manager.get_vram_free_mb", return_value=8192.0):
            st, bdg, dtl = evaluate_vram_safety("small", "mijuanlo/nllb-200-distilled-600M-ct2-int8", "cuda", "cuda")
            self.assertEqual(st, "safe")
            self.assertIn("VRAM Aman", bdg)

        # Insufficient VRAM (1.2 GB for large models)
        with patch("lumacaption.model_manager.get_vram_free_mb", return_value=1200.0):
            st, bdg, dtl = evaluate_vram_safety("large-v3", "mijuanlo/nllb-200-distilled-1.3B-int8-ct2", "cuda", "cuda")
            self.assertEqual(st, "danger")
            self.assertIn("Bahaya OOM", bdg)

        # Compute type awareness: small model with 2.6 GB VRAM free
        # float32 needs more memory than int8_float16
        with patch("lumacaption.model_manager.get_vram_free_mb", return_value=2600.0):
            st_int8, bdg_int8, dtl_int8 = evaluate_vram_safety(
                "small", "mijuanlo/nllb-200-distilled-600M-ct2-int8", "cuda", "cuda",
                stt_compute_type="int8_float16", mt_compute_type="int8_float16"
            )
            st_f32, bdg_f32, dtl_f32 = evaluate_vram_safety(
                "small", "mijuanlo/nllb-200-distilled-600M-ct2-int8", "cuda", "cuda",
                stt_compute_type="float32", mt_compute_type="float32"
            )
            # int8 should be safer than float32
            self.assertIn(st_int8, ("safe", "caution"))
            self.assertEqual(st_f32, "danger")

    def test_model_resource_table_rows(self):
        rows = model_resource_table_rows()
        self.assertEqual(len(rows), 13)
        keys = [r["key"] for r in rows]
        self.assertEqual(
            keys,
            [
                "silero",
                "dtln",
                "tiny",
                "base",
                "small",
                "whisper-small-id",
                "medium",
                "whisper-medium-id",
                "large-v3-turbo",
                "distil-large-v3",
                "large-v3",
                "nllb",
                "nllb-1.3b",
            ],
        )
        for r in rows:
            for field in ("model", "disk", "ram", "vram", "threads", "note"):
                self.assertIn(field, r)
                self.assertTrue(len(r[field]) > 0)
        # Verify small has recommendation label
        small_row = next(r for r in rows if r["key"] == "small")
        self.assertIn("[Rekomendasi]", small_row["model"])


if __name__ == "__main__":
    unittest.main()
