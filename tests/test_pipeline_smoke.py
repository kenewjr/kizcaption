from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from config import AppConfig, TargetConfig
from pipeline import CaptionPipeline
from stt.whisper_engine import Transcript


class FakeCapture:
    def __init__(self, _device, _warning, on_level=None):
        self.stopped = False
        self.on_level = on_level or (lambda _rms, _peak: None)

    def start(self):
        self.on_level(0.1, 0.2)

    def stop(self):
        self.stopped = True

    async def frames(self):
        yield np.ones(512, dtype=np.int16)
        while not self.stopped:
            await asyncio.sleep(0.01)


class FakeVad:
    def __init__(self):
        self.done = False
        self.last_probability = 0.85
        self.speaking = False

    def process(self, _frame):
        if self.done:
            return None
        self.done = True
        return np.ones(16_000, dtype=np.int16)


class FakeStt:
    active_model = "medium"
    active_device = "cpu"

    def __init__(self, *_args, **_kwargs):
        pass

    def prepare(self, _language):
        pass

    def transcribe(self, _pcm, _language):
        return Transcript("selamat pagi", "id")

    def close(self):
        pass


class FakeMt:
    active_device = "cpu"

    def __init__(self, *_args):
        pass

    def prepare(self, _targets):
        pass

    def translate(self, text, source, targets):
        assert text == "selamat pagi"
        assert source == "ind_Latn"
        return {target: f"translated-{target}" for target in targets}

    def close(self):
        pass


class FakeOutput:
    def __init__(self):
        self.published = []
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def publish(self, translations):
        self.published.append(translations)

    async def stop(self):
        self.stopped = True


class PipelineSmokeTests(unittest.TestCase):
    def test_worker_com_lifetime(self):
        for platform, result, run_error in (
            ("win32", 0, False), ("win32", 1, False),
            ("win32", 0, True), ("win32", -2147417850, False),
            ("linux", 0, False),
        ):
            with self.subTest(platform=platform, result=result, run_error=run_error):
                calls, events = [], []
                ole32 = Mock()

                def initialize(reserved, mode):
                    self.assertIsNone(reserved)
                    self.assertEqual(mode, 0)
                    calls.append(("init", threading.get_ident()))
                    return result

                def uninitialize():
                    calls.append(("uninit", threading.get_ident()))

                async def run():
                    calls.append(("run", threading.get_ident()))
                    if run_error:
                        raise RuntimeError("capture failed")

                ole32.CoInitializeEx.side_effect = initialize
                ole32.CoUninitialize.side_effect = uninitialize
                pipeline = CaptionPipeline(AppConfig(), ".", events.append)
                with patch("pipeline.sys.platform", platform), \
                        patch("pipeline.ctypes.windll", Mock(ole32=ole32), create=True), \
                        patch.object(pipeline, "_run", run):
                    pipeline.start()
                    pipeline._thread.join(3)
                self.assertFalse(pipeline.running)
                expected = ["run"] if platform != "win32" else (
                    ["init"] if result < 0 else ["init", "run", "uninit"]
                )
                self.assertEqual([name for name, _ in calls], expected)
                self.assertEqual({thread for _, thread in calls}, {pipeline._thread.ident})
                self.assertNotEqual(pipeline._thread.ident, threading.get_ident())
                errors = [event.message for event in events if event.kind == "error"]
                self.assertEqual(len(errors), int(run_error or result < 0))
                if result < 0:
                    self.assertIn("0x80010106", errors[0])
                self.assertEqual(events[-1].kind, "stopped")

    def test_one_and_three_targets_and_clean_stop(self):
        for config in (
            AppConfig(),
            AppConfig(targets=[TargetConfig(name) for name in ("English", "Japanese", "Korean")]),
        ):
            with self.subTest(targets=config.targets):
                events = []
                published = threading.Event()

                def on_event(event):
                    events.append(event)
                    if event.kind in {"published", "error"}:
                        published.set()

                output = FakeOutput()
                pipeline = CaptionPipeline(
                    config,
                    ".",
                    on_event,
                    capture_factory=FakeCapture,
                    vad_factory=FakeVad,
                    stt_factory=FakeStt,
                    mt_factory=FakeMt,
                    output_factory=lambda: output,
                )
                pipeline.start()
                try:
                    self.assertTrue(published.wait(3), events)
                finally:
                    pipeline.request_stop()
                    pipeline._thread.join(3)
                self.assertFalse(pipeline.running)
                self.assertFalse(pipeline.inference_busy)
                self.assertTrue(output.started)
                self.assertTrue(output.stopped)
                self.assertEqual(output.published, [
                    {target.language: f"translated-{target.language}" for target in config.targets}
                ])
                kinds = [event.kind for event in events]
                self.assertNotIn("error", kinds)
                self.assertIn("audio_level", kinds)
                self.assertIn("translations", kinds)
                self.assertIn("published", kinds)
                self.assertLess(kinds.index("model_ready"), kinds.index("started"))
                self.assertLess(kinds.index("started"), kinds.index("transcript"))
                metrics = next(event.data for event in events if event.kind == "metrics")
                self.assertGreaterEqual(metrics["after_vad_ms"], metrics["stt_ms"] + metrics["mt_ms"])
                vad = next(event for event in events if event.kind == "vad_probability")
                self.assertEqual(vad.data, {"probability": 0.85, "speaking": False, "threshold": 0.5})

if __name__ == "__main__":
    unittest.main()
