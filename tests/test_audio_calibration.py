from __future__ import annotations

from pathlib import Path
import queue
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from lumacaption.audio.calibration import (
    AudioCalibrationResult,
    evaluate_calibration,
    run_audio_calibration,
)


class AudioCalibrationTests(unittest.TestCase):
    def test_evaluate_calibration_quiet_audio(self):
        peaks = [0.05, 0.08, 0.12, 0.06] * 10
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.1, 0.2, 0.4, 0.1] * 10

        res = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            current_normalize=False,
            lang="id",
        )

        self.assertEqual(res.volume_status, "quiet")
        self.assertIsNotNone(res.recommended_gain_db)
        self.assertGreater(res.recommended_gain_db, 0.0)
        self.assertTrue(res.recommended_auto_normalize)
        self.assertTrue(any("terlalu rendah" in r.lower() for r in (res.recommendations or [])))

    def test_evaluate_calibration_clipping_audio(self):
        peaks = [0.98, 0.99, 0.96, 0.50] * 10
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.9, 0.9, 0.8, 0.1] * 10

        res = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=6.0,
            current_vad_threshold=0.5,
            current_normalize=True,
            lang="id",
        )

        self.assertEqual(res.volume_status, "loud")
        self.assertIsNotNone(res.recommended_gain_db)
        self.assertLess(res.recommended_gain_db, 6.0)
        self.assertTrue(any("terlalu keras" in r.lower() for r in (res.recommendations or [])))

    def test_evaluate_calibration_optimal_speech(self):
        peaks = [0.55, 0.60, 0.05, 0.02] * 20
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.92, 0.95, 0.05, 0.02] * 20

        res = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=3.0,
            current_vad_threshold=0.5,
            current_normalize=True,
            utterances_count=2,
            lang="en",
        )

        self.assertEqual(res.volume_status, "optimal")
        self.assertEqual(res.vad_status, "optimal")
        self.assertIsNone(res.recommended_gain_db)
        self.assertIsNone(res.recommended_vad_threshold)
        self.assertEqual(len(res.recommendations or []), 0)
        self.assertIn("Speech detected", res.summary_text)

    def test_evaluate_calibration_unresponsive_vad(self):
        # Audio is present (peak 0.40), but VAD prob is zero
        peaks = [0.35, 0.40, 0.38, 0.30] * 20
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.10, 0.15, 0.12, 0.10] * 20

        res = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            current_normalize=True,
            lang="id",
        )

        self.assertEqual(res.volume_status, "optimal")
        self.assertEqual(res.vad_status, "unresponsive")
        self.assertIsNotNone(res.recommended_vad_threshold)
        self.assertLess(res.recommended_vad_threshold, 0.5)

    def test_evaluate_calibration_noisy_background(self):
        # Noise prob is high when quiet
        peaks = [0.05, 0.08, 0.04, 0.03] * 20
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.55, 0.52, 0.48, 0.53] * 20

        res = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            current_normalize=True,
            lang="id",
        )

        self.assertEqual(res.vad_status, "noisy")
        self.assertIsNotNone(res.recommended_vad_threshold)
        self.assertGreater(res.recommended_vad_threshold, 0.5)

    def test_run_audio_calibration_with_mock_capture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            (app_dir / "models").mkdir(parents=True, exist_ok=True)

            mock_frames = queue.Queue()
            # Push 5 mock audio frames (512 samples each @ 16kHz)
            for _ in range(5):
                frame = (np.sin(np.linspace(0, 10, 512)) * 16000).astype(np.int16)
                mock_frames.put(frame)

            mock_capture = MagicMock()
            mock_capture._frames = mock_frames

            mock_silero = MagicMock()
            mock_silero.return_value = 0.85

            progress_calls = []

            def on_progress(remaining, peak, prob):
                progress_calls.append((remaining, peak, prob))

            res = run_audio_calibration(
                app_dir=app_dir,
                device_key=None,
                duration_sec=0.2,
                capture_factory=lambda **_: mock_capture,
                silero_factory=lambda _: mock_silero,
                on_progress=on_progress,
            )

            mock_capture.start.assert_called_once()
            mock_capture.stop.assert_called_once()
            self.assertIsInstance(res, AudioCalibrationResult)
            self.assertGreater(len(progress_calls), 0)

    def test_silero_receives_normalized_audio_not_raw_int16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            (app_dir / "models").mkdir(parents=True, exist_ok=True)

            mock_frames = queue.Queue()
            loud_frame = np.full(512, 30000, dtype=np.int16)
            mock_frames.put(loud_frame)

            mock_capture = MagicMock()
            mock_capture._frames = mock_frames

            received_args = []

            def fake_model(x):
                received_args.append(x)
                return 0.5

            mock_silero = MagicMock(side_effect=fake_model)

            run_audio_calibration(
                app_dir=app_dir,
                device_key=None,
                duration_sec=0.05,
                capture_factory=lambda **_: mock_capture,
                silero_factory=lambda _: mock_silero,
            )

            self.assertGreater(len(received_args), 0)
            for arr in received_args:
                self.assertLessEqual(float(np.max(np.abs(arr))), 1.5)
            self.assertEqual(mock_silero.call_count, 1)

    def test_evaluate_calibration_recommends_hybrid_denoiser_when_noisy(self):
        peaks = [0.05, 0.08, 0.04, 0.03] * 20
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.55, 0.52, 0.48, 0.53] * 20

        # When current is clarity -> recommend hybrid
        res1 = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            current_normalize=True,
            current_denoise_engine="clarity",
            lang="id",
        )
        self.assertEqual(res1.vad_status, "noisy")
        self.assertEqual(res1.recommended_denoise_engine, "hybrid")

        # When current is already hybrid -> do not recommend hybrid
        res2 = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            current_normalize=True,
            current_denoise_engine="hybrid",
            lang="id",
        )
        self.assertEqual(res2.vad_status, "noisy")
        self.assertIsNone(res2.recommended_denoise_engine)

    def test_evaluate_calibration_fragmented_speech_hint(self):
        peaks = [0.60] * 60 + [0.05] * 40
        rms_list = [p * 0.7 for p in peaks]
        vad_probs = [0.90] * 60 + [0.05] * 40

        # With utterances_count >= 2 -> should add hint
        res_hint = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            utterances_count=3,
            lang="id",
        )
        self.assertTrue(any("Silence Gap" in r for r in (res_hint.recommendations or [])))

        # With utterances_count <= 1 -> should NOT add hint
        res_no_hint = evaluate_calibration(
            peaks=peaks,
            rms_list=rms_list,
            vad_probs=vad_probs,
            current_gain_db=0.0,
            current_vad_threshold=0.5,
            utterances_count=1,
            lang="id",
        )
        self.assertFalse(any("Silence Gap" in r for r in (res_no_hint.recommendations or [])))


if __name__ == "__main__":
    unittest.main()
