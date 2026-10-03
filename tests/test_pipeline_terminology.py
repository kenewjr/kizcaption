from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
import unittest

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from lumacaption.config import AppConfig, TargetConfig
from lumacaption.pipeline import CaptionPipeline, PipelineEvent
from lumacaption.stt.whisper_engine import Transcript


class DirectFakeCapture:
    def __init__(self, _device, _warning, on_level=None):
        self.stopped = False

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    async def frames(self):
        yield np.ones(512, dtype=np.int16)
        while not self.stopped:
            await asyncio.sleep(0.01)


class DirectFakeVad:
    def __init__(self):
        self.emitted = False
        self.reset_called = False
        self.last_probability = 0.9
        self.speaking = False

    def reset(self):
        self.reset_called = True

    def process(self, _frame):
        if self.emitted:
            return None
        self.emitted = True
        return np.ones(16_000, dtype=np.int16)


class DirectFakeStt:
    active_model = "medium"
    active_device = "cpu"
    received_hotwords = ""

    def __init__(self, *_args, hotwords="", **_kwargs):
        DirectFakeStt.received_hotwords = hotwords

    def prepare(self, _language):
        pass

    def transcribe(self, _pcm, _language):
        return Transcript("halo rek, Pak Budi makan Indomie enak tenan", "id")

    def close(self):
        pass


class DirectFakeMt:
    active_device = "cpu"
    received_source_text = ""

    def __init__(self, *_args, **_kwargs):
        pass

    def prepare(self, _targets):
        pass

    def translate(self, text, _source_code, _targets):
        DirectFakeMt.received_source_text = text
        translated = text.replace("halo", "hello").replace("makan", "ate")
        return {"English": translated}

    def close(self):
        pass


class DirectFakeOutput:
    async def start(self):
        pass

    async def stop(self):
        pass

    async def publish(self, _translations):
        pass


class PipelineDirectTranslationTests(unittest.TestCase):
    def test_pipeline_direct_translation_and_vad_reset(self):
        events: list[PipelineEvent] = []
        cfg = AppConfig(
            targets=[TargetConfig("English", profile=1)],
            whisper_hotwords="custom_word",
        )

        fake_vad = DirectFakeVad()
        pipeline = CaptionPipeline(
            cfg,
            Path("."),
            events.append,
            capture_factory=DirectFakeCapture,
            vad_factory=lambda **_kwargs: fake_vad,
            stt_factory=DirectFakeStt,
            mt_factory=DirectFakeMt,
            output_factory=DirectFakeOutput,
        )

        pipeline.start()
        # Wait for translation event
        for _ in range(30):
            time.sleep(0.05)
            if any(e.kind == "translations" for e in events):
                break
        pipeline.request_stop()
        time.sleep(0.1)

        # 1. STT hotwords receives user hotwords directly without artificial dictionary pollution
        self.assertEqual(DirectFakeStt.received_hotwords, "custom_word")

        # 2. MT receives EXACT transcript text (no placeholder tokens «LC_P*», no slang replacement)
        self.assertEqual(DirectFakeMt.received_source_text, "halo rek, Pak Budi makan Indomie enak tenan")
        self.assertNotIn("«LC_P", DirectFakeMt.received_source_text)

        # 3. Translations delivered cleanly
        trans_events = [e for e in events if e.kind == "translations"]
        self.assertTrue(trans_events)
        en_trans = trans_events[0].data.get("English", "")
        self.assertIn("hello", en_trans)
        self.assertIn("ate", en_trans)

        # 4. vad.reset() called upon capture stop
        self.assertTrue(fake_vad.reset_called)

    def test_pipeline_backward_compatible_kwargs(self):
        events: list[PipelineEvent] = []
        cfg = AppConfig(targets=[TargetConfig("English", profile=1)])
        # Must not raise TypeError when legacy callers pass terminology_store
        pipeline = CaptionPipeline(
            cfg,
            Path("."),
            events.append,
            capture_factory=DirectFakeCapture,
            vad_factory=lambda **_kwargs: DirectFakeVad(),
            stt_factory=DirectFakeStt,
            mt_factory=DirectFakeMt,
            output_factory=DirectFakeOutput,
            terminology_store="legacy_store_object",
        )
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()
