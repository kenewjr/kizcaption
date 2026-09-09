from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from audio.capture import AudioFrameConverter, float_audio_to_mono_pcm16
from audio.vad import FRAME_SAMPLES, UtteranceTooLongError, VadSegmenter
from config import AppConfig, CAPTION_FONTS, ConfigStore, MODEL_SIZES, OverlayConfig, TargetConfig
from gpu_runtime import configure_cuda_runtime
from languages import source_nllb_code, target_nllb_code
from output.overlay_server import OverlayServer, OverlayService, overlay_url
from ui.control_panel import UNUSED_TARGET, selected_targets


class FakeProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def reset(self):
        pass

    def __call__(self, _frame):
        return next(self.probabilities)


class CoreTests(unittest.TestCase):
    def test_config_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "config.json")
            expected = AppConfig(
                targets=[TargetConfig("English")],
                overlay=OverlayConfig(font_family="Consolas", font_size=64),
            )
            store.save(expected)
            actual, warning = store.load()
            self.assertIsNone(warning)
            self.assertEqual(actual, expected)
            with self.assertRaises(ValueError):
                AppConfig(
                    targets=[TargetConfig("English"), TargetConfig("English")]
                ).validate()
            with self.assertRaises(ValueError):
                AppConfig(targets=[TargetConfig("Unknown")]).validate()

    def test_legacy_config_migrates_without_obs_fields(self):
        legacy = {
            "obs": {"host": "127.0.0.1", "port": 4455, "password": "secret"},
            "output_mode": "obs",
            "targets": [
                {"language": "English", "obs_source": "caption_en"},
                {"language": "Japanese", "obs_source": "caption_ja"},
            ],
            "overlay": {"port": 9001, "unknown": True},
        }
        config = AppConfig.from_dict(legacy)
        self.assertEqual(config.targets, [TargetConfig("English"), TargetConfig("Japanese")])
        self.assertEqual(config.overlay, OverlayConfig(port=9001))
        self.assertFalse(hasattr(config, "obs"))

    def test_corrupt_config_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{bad", encoding="utf-8")
            config, warning = ConfigStore(path).load()
            self.assertIsInstance(config, AppConfig)
            self.assertIn("corrupt", warning.lower())
            self.assertTrue(path.with_suffix(".json.corrupt").exists())

    def test_vad_emits_only_after_pause(self):
        probabilities = [0.0, 0.8, 0.9, 0.9, 0.1, 0.1]
        vad = VadSegmenter(
            FakeProbabilityModel(probabilities),
            threshold=0.5,
            min_silence_ms=64,
            min_speech_ms=64,
            speech_pad_ms=32,
        )
        frame = np.ones(FRAME_SAMPLES, dtype=np.int16)
        results = [vad.process(frame) for _ in probabilities]
        self.assertTrue(all(result is None for result in results[:-1]))
        self.assertGreater(results[-1].size, FRAME_SAMPLES * 2)

    def test_vad_probability_resets(self):
        vad = VadSegmenter(FakeProbabilityModel([0.75]))
        self.assertEqual(vad.last_probability, 0)
        vad.process(np.ones(FRAME_SAMPLES, dtype=np.int16))
        self.assertEqual(vad.last_probability, 0.75)
        vad.reset()
        self.assertEqual(vad.last_probability, 0)
        self.assertFalse(vad.speaking)

    def test_optional_targets_are_explicit(self):
        for choices, expected in (
            (["English", UNUSED_TARGET, UNUSED_TARGET], ["English"]),
            (["English", UNUSED_TARGET, "Japanese"], ["English", "Japanese"]),
            (["English", "Japanese", "Korean"], ["English", "Japanese", "Korean"]),
        ):
            targets = selected_targets(choices)
            self.assertEqual([target.language for target in targets], expected)
            AppConfig(targets=targets).validate()
        for choices in ([], [UNUSED_TARGET, "English"], ["", "English"], ["English", ""], ["English", "Unknown"]):
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                selected_targets(choices)
        with self.assertRaises(ValueError):
            AppConfig(targets=selected_targets(["English", "English"])).validate()
        self.assertEqual(AppConfig().targets, [TargetConfig("English")])

    def test_caption_style_validation(self):
        for font in CAPTION_FONTS:
            for size in (16, 48, 96):
                AppConfig(overlay=OverlayConfig(font_family=font, font_size=size)).validate()
        for overlay in (
            OverlayConfig(font_family=""), OverlayConfig(font_family='Arial; color:red'),
            OverlayConfig(font_size=15), OverlayConfig(font_size=97),
        ):
            with self.subTest(overlay=overlay), self.assertRaises(ValueError):
                AppConfig(overlay=overlay).validate()

    def test_float_downmix_preserves_second_channel_and_meter(self):
        samples = np.array([[0, 0.5], [0, -0.5], [0, 0.5], [0, -0.5]], dtype=np.float32)
        pcm, rms, peak = float_audio_to_mono_pcm16(samples)
        np.testing.assert_array_equal(pcm, [8192, -8192, 8192, -8192])
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(rms, 0.25)
        self.assertEqual(peak, 0.25)
        pcm, rms, peak = float_audio_to_mono_pcm16(np.array([2, -2], dtype=np.float32))
        np.testing.assert_array_equal(pcm, [32767, -32767])
        self.assertEqual((rms, peak), (2, 2))
        pcm, rms, peak = float_audio_to_mono_pcm16(np.empty((0, 2), dtype=np.float32))
        self.assertEqual(pcm.size, 0)
        self.assertEqual((rms, peak), (0, 0))

    def test_resampler_frames_are_continuous(self):
        for rate in (16_000, 44_100, 48_000):
            with self.subTest(rate=rate):
                samples = np.rint(np.sin(np.arange(rate) * 2 * np.pi * 440 / rate) * 8000).astype(np.int16)
                converter = AudioFrameConverter(rate)
                frames = []
                for offset in range(0, len(samples), 137):
                    frames.extend(converter.process(samples[offset:offset + 137]))
                expected = AudioFrameConverter(rate).process(samples)
                self.assertEqual(len(frames), 31)
                self.assertTrue(all(frame.shape == (512,) and frame.dtype == np.int16 for frame in frames))
                np.testing.assert_allclose(np.concatenate(frames), np.concatenate(expected), atol=1)

    def test_language_routing(self):
        self.assertEqual(source_nllb_code("id", "en"), "ind_Latn")
        self.assertEqual(source_nllb_code("auto", "id"), "ind_Latn")
        self.assertEqual(target_nllb_code("Japanese"), "jpn_Jpan")

    def test_overlay_uses_text_content_and_encoded_preview_url(self):
        app_dir = Path(__file__).parents[1]
        html = (app_dir / "output" / "overlay.html").read_text(encoding="utf-8")
        self.assertIn("textContent", html)
        self.assertNotIn("innerHTML", html)
        service = OverlayService(
            OverlayConfig(),
            [TargetConfig("Chinese (Simplified)")],
            app_dir / "output" / "overlay.html",
            8,
        )
        parsed = urlparse(service.preview_url("Chinese (Simplified)"))
        self.assertEqual(parse_qs(parsed.query)["lang"], ["Chinese (Simplified)"])
        self.assertEqual(parse_qs(parsed.query)["preview"], ["1"])
        service.config.font_family = "Trebuchet MS"
        service.config.font_size = 72
        server = OverlayServer(service.config, service.targets, app_dir / "output" / "overlay.html")
        language = "Chinese (Simplified)"
        self.assertEqual(server.url(language), service.url(language))
        self.assertEqual(server.preview_url(language), service.preview_url(language))
        self.assertEqual(service.url(language), overlay_url(service.config, language))
        query = parse_qs(urlparse(service.preview_url(language)).query)
        self.assertEqual(query["font"], ["Trebuchet MS"])
        self.assertEqual(query["size"], ["72"])
        self.assertNotIn("preview", parse_qs(urlparse(service.url(language)).query))
        ipv6 = urlparse(overlay_url(OverlayConfig(host="::1"), "English"))
        self.assertEqual(ipv6.hostname, "::1")
        self.assertEqual(ipv6.port, 8765)

    def test_named_microphone_beats_default_with_same_index(self):
        from unittest.mock import patch
        from audio.capture import InputDevice, MicrophoneCapture

        devices = [
            InputDevice("__default__", "Default Windows", 40, 48_000, "WASAPI", True),
            InputDevice("device:sonar", "Sonar Microphone", 40, 48_000, "WASAPI"),
        ]
        with patch.object(MicrophoneCapture, "devices", return_value=devices):
            self.assertEqual(MicrophoneCapture.resolve_device("device:sonar").name, "Sonar Microphone")
            self.assertEqual(MicrophoneCapture.resolve_device(None).name, "Default Windows")

    def test_failed_capture_start_closes_stream_and_can_retry(self):
        from unittest.mock import Mock, patch
        from audio.capture import InputDevice, MicrophoneCapture

        selected = InputDevice("device:test", "Test Mic", 7, 48_000, "Windows WASAPI")
        for fail_on_create in (False, True):
            with self.subTest(fail_on_create=fail_on_create):
                failed, good, sd = Mock(), Mock(), Mock()
                error = RuntimeError("start failed")
                failed.start.side_effect = error
                sd.InputStream.side_effect = [error if fail_on_create else failed, good]
                with patch.dict("sys.modules", {"sounddevice": sd}), \
                        patch.object(MicrophoneCapture, "resolve_device", return_value=selected):
                    capture = MicrophoneCapture(selected.key)
                    with self.assertRaisesRegex(RuntimeError, "start failed"):
                        capture.start()
                    self.assertIsNone(capture._stream)
                    self.assertTrue(capture._closed.is_set())
                    if fail_on_create:
                        failed.close.assert_not_called()
                    else:
                        failed.close.assert_called_once_with(ignore_errors=True)
                    capture.stop()
                    try:
                        self.assertEqual(capture.start(), selected)
                        self.assertFalse(capture._closed.is_set())
                        self.assertIs(capture._stream, good)
                        good.start.assert_called_once_with()
                    finally:
                        capture.stop()
                    good.abort.assert_called_once_with(ignore_errors=True)
                    good.close.assert_called_once_with(ignore_errors=True)

    def test_decoder_config_validation(self):
        self.assertIn("large-v3-turbo", MODEL_SIZES)
        for beam in (1, 3, 5):
            AppConfig(whisper_beam_size=beam).validate()
        for bad_beam in (0, 2, 4, 6, "3", 3.0):
            with self.subTest(beam=bad_beam), self.assertRaises(ValueError):
                AppConfig(whisper_beam_size=bad_beam).validate()
        AppConfig(whisper_hotwords="teknologi, LumaCaption").validate()
        with self.assertRaises(ValueError):
            AppConfig(whisper_hotwords="x" * 301).validate()
        with self.assertRaises(ValueError):
            AppConfig(whisper_hotwords="baris1\nbaris2").validate()
        for duration in (3, 20, 60):
            AppConfig(max_utterance_seconds=duration).validate()
        for bad_duration in (2, 61, 0):
            with self.subTest(duration=bad_duration), self.assertRaises(ValueError):
                AppConfig(max_utterance_seconds=bad_duration).validate()

    def test_vad_raises_on_continuous_speech_without_pause(self):
        # 3 frames max utterance limit
        vad = VadSegmenter(
            FakeProbabilityModel([0.9] * 10),
            threshold=0.5,
            min_silence_ms=64,
            min_speech_ms=32,
            max_utterance_seconds=0.096,  # ~3 frames (3 * 512 / 16000 = 0.096s)
            speech_pad_ms=32,
        )
        frame = np.ones(FRAME_SAMPLES, dtype=np.int16)
        # 1st frame starts speech, 2nd frame adds, 3rd frame exceeds max_frames limit without pause
        vad.process(frame)
        vad.process(frame)
        with self.assertRaises(UtteranceTooLongError):
            vad.process(frame)
        self.assertFalse(vad.speaking)

    def test_vad_preserves_preroll_and_postroll(self):
        # 2 frames silence, 2 frames speech, 2 frames silence (pause)
        probs = [0.1, 0.1, 0.9, 0.9, 0.1, 0.1]
        vad = VadSegmenter(
            FakeProbabilityModel(probs),
            threshold=0.5,
            min_silence_ms=64,   # 2 frames
            min_speech_ms=32,    # 1 frame
            speech_pad_ms=64,    # 2 frames pre-roll
        )
        frame = np.ones(FRAME_SAMPLES, dtype=np.int16)
        result = None
        for prob in probs:
            res = vad.process(frame)
            if res is not None:
                result = res
        self.assertIsNotNone(result)
        # Should include pre-roll + speech + post-roll frames: >= 4 frames
        self.assertGreaterEqual(result.size, FRAME_SAMPLES * 4)

    def test_gpu_runtime_idempotent(self):
        configure_cuda_runtime()
        configure_cuda_runtime()


if __name__ == "__main__":
    unittest.main()
