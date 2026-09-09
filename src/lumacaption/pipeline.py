from __future__ import annotations

import asyncio
from collections.abc import Callable
import ctypes
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time

from audio.capture import MicrophoneCapture
from audio.vad import SileroOnnx, UtteranceTooLongError, VadSegmenter
from config import AppConfig
from languages import source_nllb_code, source_whisper_code
from mt.nllb_engine import NllbEngine
from output.overlay_server import OverlayPublisher, OverlayServer
from stt.whisper_engine import WhisperEngine


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    kind: str
    message: str
    data: dict | None = None


_INFERENCE_STOP = object()


class CaptionPipeline:
    def __init__(
        self,
        config: AppConfig,
        app_dir: str | Path,
        on_event: Callable[[PipelineEvent], None],
        capture_factory=MicrophoneCapture,
        vad_factory=None,
        stt_factory=WhisperEngine,
        mt_factory=NllbEngine,
        output_factory=None,
        overlay_service=None,
    ) -> None:
        self.config = config
        self.app_dir = Path(app_dir)
        self.on_event = on_event
        self.capture_factory = capture_factory
        self.vad_factory = vad_factory
        self.stt_factory = stt_factory
        self.mt_factory = mt_factory
        self.output_factory = output_factory
        self.overlay_service = overlay_service
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._capture = None
        self._utterances: queue.Queue | None = None
        self._stop_requested = threading.Event()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def inference_busy(self) -> bool:
        if self._worker and not self._worker.is_alive():
            self._worker = None
        return bool(self._worker and self._worker.is_alive())

    def _emit(self, kind: str, message: str, data: dict | None = None) -> None:
        try:
            self.on_event(PipelineEvent(kind, message, data))
        except Exception:
            pass

    def start(self) -> None:
        if self.running or self.inference_busy:
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="caption-pipeline",
            daemon=True,
        )
        self._thread.start()

    def _thread_main(self) -> None:
        com = None
        try:
            if sys.platform == "win32":
                # WASAPI needs COM on this worker, not only on Tk's main thread.
                ole32 = ctypes.windll.ole32
                result = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
                if result < 0:
                    raise OSError(f"Windows COM initialization failed: 0x{result & 0xFFFFFFFF:08X}")
                com = ole32
            asyncio.run(self._run())
        except Exception as exc:
            if not self._stop_requested.is_set():
                self._emit("error", f"Pipeline stopped: {exc}")
        finally:
            if com is not None:
                com.CoUninitialize()
            self._loop = None
            self._capture = None
            self._emit("stopped", "Caption dihentikan")

    def _signal_worker_stop(self) -> None:
        utterances = self._utterances
        if utterances is None:
            return
        while True:
            try:
                utterances.put_nowait(_INFERENCE_STOP)
                return
            except queue.Full:
                try:
                    utterances.get_nowait()
                    utterances.task_done()
                except queue.Empty:
                    return

    def request_stop(self) -> None:
        self._stop_requested.set()
        # Capture shutdown belongs to the pipeline thread, not the Tk button callback.
        self._signal_worker_stop()

    async def stop(self, timeout: float = 3.0) -> None:
        self.request_stop()
        if self._thread and self._thread is not threading.current_thread():
            await asyncio.to_thread(self._thread.join, timeout)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()

        def warning(message: str) -> None:
            if not self._stop_requested.is_set():
                self._emit("warning", message)

        def audio_level(rms: float, peak: float) -> None:
            if not self._stop_requested.is_set():
                self._emit("audio_level", "Audio level", {"rms": rms, "peak": peak})

        capture = self.capture_factory(self.config.microphone_device, warning, audio_level)
        self._capture = capture
        if self.vad_factory:
            vad = self.vad_factory()
        else:
            probability_model = SileroOnnx(self.app_dir / "models" / "silero_vad.onnx")
            vad = VadSegmenter(
                probability_model,
                threshold=self.config.vad_threshold,
                min_silence_ms=self.config.min_silence_ms,
                min_speech_ms=self.config.min_speech_ms,
                max_utterance_seconds=self.config.max_utterance_seconds,
            )

        status = lambda level, message: self._emit(level, message)
        if self.output_factory:
            output = self.output_factory()
        elif self.overlay_service is not None:
            output = OverlayPublisher(self.overlay_service)
        else:
            output = OverlayServer(
                self.config.overlay,
                self.config.targets,
                self.app_dir / "output" / "overlay.html",
                self.config.caption_timeout_seconds,
                status,
            )

        utterances: queue.Queue = queue.Queue(maxsize=2)
        self._utterances = utterances
        target_names = [target.language for target in self.config.targets]
        publish_tasks: set[asyncio.Task] = set()
        ready = asyncio.Event()
        publish_lock = asyncio.Lock()

        async def deliver(translations: dict[str, str], metrics: dict) -> None:
            if self._stop_requested.is_set():
                return
            try:
                async with publish_lock:
                    if self._stop_requested.is_set():
                        return
                    await output.publish(translations)
                if not self._stop_requested.is_set():
                    self._emit("metrics", "Waktu pemrosesan", {
                        **metrics,
                        "after_vad_ms": (time.monotonic() - metrics["queued_at"]) * 1000,
                    })
                    self._emit(
                        "published",
                        "Caption tampil di Browser Source",
                        dict(translations),
                    )
            except Exception as exc:
                if not self._stop_requested.is_set():
                    warning(f"Overlay output failed: {exc}")

        def schedule_publish(translations: dict[str, str], metrics: dict) -> None:
            if self._stop_requested.is_set() or not self._loop or self._loop.is_closed():
                return
            task = asyncio.create_task(deliver(translations, metrics), name="overlay-publish")
            publish_tasks.add(task)
            task.add_done_callback(publish_tasks.discard)

        def inference_worker() -> None:
            stt = None
            mt = None
            try:
                if self._stop_requested.is_set():
                    return
                self._emit("preparing", "Menyiapkan Whisper dan memeriksa inference…")
                stt = self.stt_factory(
                    self.config.whisper_model,
                    self.config.stt_device,
                    self.app_dir / "models" / "whisper",
                    warning,
                    beam_size=self.config.whisper_beam_size,
                    hotwords=self.config.whisper_hotwords,
                )
                stt.prepare(source_whisper_code(self.config.source_language))
                if self._stop_requested.is_set():
                    return
                self._emit("preparing", "Menyiapkan NLLB dan memeriksa terjemahan…")
                mt = self.mt_factory(
                    self.config.nllb_model,
                    self.config.mt_device,
                    self.app_dir / "models" / "nllb-cache",
                    warning,
                )
                mt.prepare(target_names)
                if self._stop_requested.is_set():
                    return
                self._emit("model_ready", "Model siap", {
                    "model": stt.active_model,
                    "stt_device": stt.active_device,
                    "mt_device": mt.active_device,
                })
                self._loop.call_soon_threadsafe(ready.set)
                while not self._stop_requested.is_set():
                    try:
                        item = utterances.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    try:
                        if item is _INFERENCE_STOP or self._stop_requested.is_set():
                            return
                        utterance, queued_at = item
                        self._emit("processing", "Mentranskripsi ucapan selesai…")
                        stt_start = time.monotonic()
                        transcript = stt.transcribe(
                            utterance,
                            source_whisper_code(self.config.source_language),
                        )
                        stt_ms = (time.monotonic() - stt_start) * 1000
                        if self._stop_requested.is_set():
                            return
                        if not transcript.text:
                            self._emit("listening", "Tidak ada ucapan yang dikenali")
                            continue
                        self._emit(
                            "transcript",
                            transcript.text,
                            {"language": transcript.language, "text": transcript.text},
                        )
                        source_code = source_nllb_code(
                            self.config.source_language,
                            transcript.language,
                        )
                        self._emit("processing", "Menerjemahkan caption…")
                        mt_start = time.monotonic()
                        translations = mt.translate(transcript.text, source_code, target_names)
                        mt_ms = (time.monotonic() - mt_start) * 1000
                        if self._stop_requested.is_set():
                            return
                        self._emit("translations", "Translations ready", translations)
                        loop = self._loop
                        if loop and loop.is_running():
                            loop.call_soon_threadsafe(schedule_publish, translations, {
                                "stt_ms": stt_ms, "mt_ms": mt_ms, "queued_at": queued_at,
                                "stt_device": stt.active_device, "mt_device": mt.active_device,
                            })
                    except ValueError as exc:
                        if not self._stop_requested.is_set():
                            self._emit("error", str(exc))
                        self.request_stop()
                    finally:
                        utterances.task_done()
            except Exception as exc:
                if not self._stop_requested.is_set():
                    self._emit("error", f"Inference failed: {exc}")
                    self.request_stop()
            finally:
                for engine in (mt, stt):
                    if engine is not None:
                        try:
                            engine.close()
                        except Exception as exc:
                            self._emit("error", f"Model cleanup failed: {exc}")
                self._emit("inference_idle", "Mesin caption berhenti")

        async def capture_stage() -> None:
            await ready.wait()
            delay = 1.0
            last_vad_event_at = 0.0
            while not self._stop_requested.is_set():
                try:
                    selected = capture.start()
                    name = getattr(selected, "name", "microphone")
                    rate = getattr(selected, "sample_rate", 16_000)
                    self._emit(
                        "listening",
                        f"Mendengarkan {name}",
                        {"device": name, "sample_rate": rate},
                    )
                    self._emit("started", "LIVE • hasil tampil setelah jeda bicara")
                    delay = 1.0
                    async for frame in capture.frames():
                        if self._stop_requested.is_set():
                            break
                        was_speaking = bool(getattr(vad, "speaking", False))
                        utterance = vad.process(frame)
                        now = time.monotonic()
                        if now - last_vad_event_at >= 0.2:
                            self._emit(
                                "vad_probability",
                                "VAD probability",
                                {
                                    "probability": float(getattr(vad, "last_probability", 0.0)),
                                    "speaking": bool(getattr(vad, "speaking", False)),
                                    "threshold": self.config.vad_threshold,
                                },
                            )
                            last_vad_event_at = now
                        if not was_speaking and bool(getattr(vad, "speaking", False)):
                            self._emit("speech", "Suara terdeteksi")
                        if utterance is None:
                            continue
                        seconds = len(utterance) / 16_000
                        self._emit(
                            "queued",
                            f"Ucapan {seconds:.1f} d siap diproses",
                            {"seconds": seconds},
                        )
                        try:
                            utterances.put_nowait((utterance, time.monotonic()))
                        except queue.Full:
                            self._emit("error", "Mesin tertinggal: antrean penuh. Hentikan lalu gunakan model lebih cepat.")
                            self.request_stop()
                            return
                except UtteranceTooLongError as exc:
                    self._emit("error", str(exc))
                    self.request_stop()
                    return
                except Exception as exc:
                    if self._stop_requested.is_set():
                        break
                    warning(f"Microphone unavailable: {exc}; retrying")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8.0)
                finally:
                    capture.stop()

        await output.start()
        self._worker = threading.Thread(
            target=inference_worker,
            name="caption-inference",
            daemon=True,
        )
        self._worker.start()
        capture_task = asyncio.create_task(capture_stage(), name="audio-vad")
        try:
            while not self._stop_requested.is_set():
                done, _pending = await asyncio.wait(
                    [capture_task],
                    timeout=0.1,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                if done:
                    error = capture_task.exception()
                    if error:
                        raise error
                    break
        finally:
            self._stop_requested.set()
            capture.stop()
            self._signal_worker_stop()
            capture_task.cancel()
            await asyncio.gather(capture_task, return_exceptions=True)
            for task in list(publish_tasks):
                task.cancel()
            await asyncio.gather(*publish_tasks, return_exceptions=True)
            await output.stop()
            self._utterances = None
            worker = self._worker
            if worker and worker.is_alive():
                worker.join(0.05)
                if worker.is_alive():
                    self._emit(
                        "background",
                        "Capture berhenti; panggilan model lama belum selesai. Browser Source tetap tersedia jika memakai panel.",
                    )
            if worker is None or not worker.is_alive():
                self._worker = None
