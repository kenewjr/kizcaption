from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from faster_whisper.audio import decode_audio

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import AppConfig, OverlayConfig, TargetConfig
from output.overlay_server import OverlayService
from pipeline import CaptionPipeline, PipelineEvent

class ReplayCapture:
    def __init__(self, pcm_frames: list[np.ndarray], on_warning, on_level):
        self.pcm_frames = pcm_frames
        self.on_warning = on_warning
        self.on_level = on_level
        self.name = "Synthetic Replay Mic"
        self.sample_rate = 16000
        self.stopped = False
        self.completed = False

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    async def frames(self) -> AsyncIterator[np.ndarray]:
        for frame in self.pcm_frames:
            if self.stopped:
                break
            await asyncio.sleep(0.005)
            self.on_level(0.05, 0.1)
            yield frame
        # Feed 1 second of silence so VAD triggers final pause
        silence_frame = np.zeros(512, dtype=np.int16)
        for _ in range(35):
            if self.stopped:
                break
            await asyncio.sleep(0.005)
            self.on_level(0.001, 0.002)
            yield silence_frame
        self.completed = True

class AudioReplayPipelineTest(unittest.TestCase):
    def test_pipeline_real_inference_replay(self):
        app_dir = Path(__file__).parents[1]
        manifest_path = Path(r"C:\Users\Kenewjr\.gemini\antigravity-ide\brain\4d99ab57-0e07-467c-9ff4-fa109ed8455b\scratch\fleurs_id\manifest.json")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        # Pick first sample: 8.5s speech
        sample = manifest["samples"][0]
        audio_file = manifest_path.parent / sample["path"]
        raw_audio = decode_audio(str(audio_file), sampling_rate=16000)
        int_pcm = np.clip(raw_audio * 32767.0, -32768, 32767).astype(np.int16)

        # Slice into 512-sample frames
        frames = [int_pcm[i:i+512] for i in range(0, len(int_pcm) - 512 + 1, 512)]

        config = AppConfig(
            source_language="id",
            targets=[TargetConfig("English"), TargetConfig("Japanese")],
            whisper_model="large-v3-turbo",
            whisper_beam_size=3,
            stt_device="cuda",
            mt_device="cuda",
            overlay=OverlayConfig(host="127.0.0.1", port=8768),
            vad_threshold=0.3,
            min_silence_ms=500,
        )

        events: list[PipelineEvent] = []
        event_cond = asyncio.Event()

        overlay_service = OverlayService(
            config.overlay,
            config.targets,
            app_dir / "output" / "overlay.html",
            8,
            lambda kind, msg: None,
        )
        overlay_service.start()
        self.addCleanup(lambda: overlay_service.stop(timeout=2.0))

        def on_event(ev: PipelineEvent):
            events.append(ev)
            if ev.kind in {"published", "error"}:
                event_cond.set()

        capture_inst = None

        def make_capture(dev, warn, lvl):
            nonlocal capture_inst
            capture_inst = ReplayCapture(frames, warn, lvl)
            return capture_inst

        pipeline = CaptionPipeline(
            config,
            app_dir,
            on_event,
            capture_factory=make_capture,
            overlay_service=overlay_service,
        )

        pipeline.start()
        try:
            start_t = time.monotonic()
            while time.monotonic() - start_t < 15:
                time.sleep(0.1)
                if capture_inst and capture_inst.completed and any(ev.kind == "published" for ev in events):
                    time.sleep(0.5)
                    break
        finally:
            pipeline.request_stop()
            if pipeline._thread:
                pipeline._thread.join(timeout=3.0)

        kinds = [ev.kind for ev in events]
        print(f"Replay pipeline events: {kinds}", flush=True)

        self.assertIn("model_ready", kinds)
        self.assertIn("started", kinds)
        self.assertIn("speech", kinds)
        self.assertIn("transcript", kinds)
        self.assertIn("translations", kinds)
        self.assertIn("published", kinds)

        sys.stdout.reconfigure(encoding="utf-8")
        all_transcripts = [ev.data["text"] for ev in events if ev.kind == "transcript"]
        print(f"Ref: {sample['text']}")
        print(f"Hyp: {' '.join(all_transcripts)}")

        trans_events = [ev.data for ev in events if ev.kind == "translations"]
        print(f"Translations count: {len(trans_events)}")
        for t in trans_events:
            print(f"  EN: {t.get('English')}")

        metrics_ev = next(ev for ev in events if ev.kind == "metrics")
        print(f"Metrics: {metrics_ev.data}")

if __name__ == "__main__":
    unittest.main()
