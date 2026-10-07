from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.hardware_benchmark import (
    BenchmarkResult,
    compute_recommendation,
    run_hardware_benchmark,
)


class HardwareBenchmarkTests(unittest.TestCase):
    def test_compute_recommendation_logic(self):
        # 1. High spec with fast CUDA and low MT latency
        rec_high = compute_recommendation(rtf_cpu=2.5, rtf_cuda=5.2, nllb_ms=65.0)
        self.assertEqual(rec_high, "high")

        # 2. Medium spec on CPU with adequate real-time factor
        rec_med = compute_recommendation(rtf_cpu=2.1, rtf_cuda=None, nllb_ms=110.0)
        self.assertEqual(rec_med, "medium")

        # 3. Low spec when real-time factor is slow (< 1.8x)
        rec_low = compute_recommendation(rtf_cpu=1.1, rtf_cuda=None, nllb_ms=220.0)
        self.assertEqual(rec_low, "low")

    @patch("lumacaption.stt.whisper_engine.WhisperEngine")
    @patch("lumacaption.mt.nllb_engine.NllbEngine")
    @patch("lumacaption.hardware_benchmark.is_cuda_available", return_value=False)
    def test_run_hardware_benchmark_cpu_flow(self, mock_cuda, mock_nllb_cls, mock_whisper_cls):
        mock_whisper = MagicMock()
        mock_whisper_cls.return_value = mock_whisper

        mock_nllb = MagicMock()
        mock_nllb.translate.return_value = {"English": "Hello world"}
        mock_nllb_cls.return_value = mock_nllb

        progress_msgs = []
        result = run_hardware_benchmark(
            app_dir=_root,
            on_progress=progress_msgs.append,
        )

        self.assertIsInstance(result, BenchmarkResult)
        self.assertGreater(result.whisper_rtf_cpu, 0.0)
        self.assertIsNone(result.whisper_rtf_cuda)
        self.assertIn(result.recommended_preset, ("low", "medium", "high"))
        self.assertTrue(len(progress_msgs) >= 2)
        mock_whisper.close.assert_called_once()
        mock_nllb.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
