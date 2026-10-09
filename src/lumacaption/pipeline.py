from __future__ import annotations

import asyncio
from collections.abc import Callable
import ctypes
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
import queue
import sys
import threading
import time

logger = logging.getLogger("lumacaption.pipeline")

from lumacaption.audio.capture import MicrophoneCapture
from lumacaption.audio.vad import SileroOnnx, UtteranceTooLongError, VadSegmenter
from lumacaption.config import AppConfig
from lumacaption.languages import source_nllb_code, source_whisper_code, target_nllb_code
from lumacaption.censor import censor_text
from lumacaption.mt.nllb_engine import NllbEngine
from lumacaption.output.overlay_server import OverlayPublisher, OverlayServer
from lumacaption.stt.whisper_engine import WhisperEngine
from lumacaption.vocabulary import VocabularyManager


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
        terminology_store=None,
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
        self.vocab_manager = VocabularyManager(self.app_dir)
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
            if kind in {"audio_level", "vad_probability"}:
                pass
            elif kind == "error":
                logger.error(f"[{kind}] {message}")
            elif kind == "warning":
                logger.warning(f"[{kind}] {message}")
            elif kind in {"model_ready", "started", "published", "transcript", "translations", "learning"}:
                logger.info(f"[{kind}] {message}")
            else:
                logger.debug(f"[{kind}] {message}")
        except Exception:
            pass
        try:
            self.on_event(PipelineEvent(kind, message, data))
        except Exception:
            pass

    def _append_review_log(self, detected_lang, source_code, source_text, translations, stt_ms, mt_ms) -> None:
        try:
            log_dir = self.app_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / "translation_review.jsonl"
            # Rotasi manual sederhana: kalau file sudah > 5MB, geser ke .jsonl.1 (timpa yang lama)
            if path.exists() and path.stat().st_size > 5_242_880:
                backup = log_dir / "translation_review.jsonl.1"
                path.replace(backup)
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "detected_language": detected_lang,
                "source_code": source_code,
                "source_text": source_text,
                "translations": translations,
                "stt_ms": round(stt_ms, 1),
                "mt_ms": round(mt_ms, 1),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Gagal menulis translation review log: {exc}")

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
        # Drain any backlog first so worker does not process stale audio
        while not utterances.empty():
            try:
                utterances.get_nowait()
                utterances.task_done()
            except Exception:
                break
        try:
            utterances.put_nowait(_INFERENCE_STOP)
        except Exception:
            pass

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

        capture_kwargs = {}
        if hasattr(self.config, "audio_channel"):
            capture_kwargs["channel"] = self.config.audio_channel
        if hasattr(self.config, "audio_gain_db"):
            capture_kwargs["gain_db"] = self.config.audio_gain_db
        if hasattr(self.config, "audio_clarity"):
            capture_kwargs["clarity"] = self.config.audio_clarity
        if hasattr(self.config, "denoise_engine"):
            capture_kwargs["denoise_engine"] = self.config.denoise_engine
        if hasattr(self.config, "normalize_audio"):
            capture_kwargs["normalize_audio"] = self.config.normalize_audio
        capture_kwargs["model_dir"] = self.app_dir / "models" / "dtln"
        try:
            capture = self.capture_factory(
                self.config.microphone_device,
                warning,
                audio_level,
                **capture_kwargs,
            )
        except TypeError:
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

        utterances: queue.Queue = queue.Queue(maxsize=8)
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
                        "Caption terkirim ke server overlay",
                        dict(translations),
                    )
            except Exception as exc:
                if not self._stop_requested.is_set():
                    warning(f"Overlay output failed: {exc}")

        def schedule_publish(translations: dict[str, str], metrics: dict) -> None:
            if self._stop_requested.is_set() or not self._loop or self._loop.is_closed():
                return
            if len(publish_tasks) >= 5:
                warning("Antrean output overlay penuh, membuang pembaruan lama")
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

                combined_hotwords = self.vocab_manager.get_hotwords(
                    user_hotwords=self.config.whisper_hotwords or "",
                )

                total_threads = getattr(self.config, "cpu_threads", 4)
                both_cpu = (self.config.stt_device == "cpu" and self.config.mt_device == "cpu" and bool(target_names))
                if both_cpu:
                    stt_threads = max(1, round(total_threads * 0.60))
                    mt_threads = max(1, total_threads - stt_threads)
                else:
                    stt_threads = total_threads
                    mt_threads = total_threads

                self._emit("preparing", f"Memuat Whisper [{self.config.whisper_model}]…")
                stt_kwargs = {
                    "beam_size": self.config.whisper_beam_size,
                    "hotwords": combined_hotwords,
                }
                if hasattr(self.config, "stt_compute_type"):
                    stt_kwargs["compute_type"] = self.config.stt_compute_type
                if hasattr(self.config, "cpu_threads"):
                    stt_kwargs["cpu_threads"] = stt_threads
                try:
                    stt = self.stt_factory(
                        self.config.whisper_model,
                        self.config.stt_device,
                        self.app_dir / "models" / "whisper",
                        warning,
                        **stt_kwargs,
                    )
                except TypeError:
                    # Fallback for mock/test factories accepting fewer kwargs
                    stt = self.stt_factory(
                        self.config.whisper_model,
                        self.config.stt_device,
                        self.app_dir / "models" / "whisper",
                        warning,
                        beam_size=self.config.whisper_beam_size,
                        hotwords=combined_hotwords,
                    )
                self._emit("preparing", f"Memeriksa inferensi Whisper [{self.config.whisper_model}]…")
                stt.prepare(source_whisper_code(self.config.source_language))
                if self._stop_requested.is_set():
                    return
                active_targets_str = ", ".join(target_names) if target_names else "tanpa translasi"
                self._emit("preparing", f"Menyiapkan translasi NLLB-200 [{active_targets_str}]…")
                mt_kwargs = {}
                if hasattr(self.config, "mt_compute_type"):
                    mt_kwargs["compute_type"] = self.config.mt_compute_type
                if hasattr(self.config, "mt_beam_size"):
                    mt_kwargs["beam_size"] = self.config.mt_beam_size
                if hasattr(self.config, "cpu_threads"):
                    mt_kwargs["cpu_threads"] = mt_threads
                if hasattr(self.config, "slang_normalization"):
                    mt_kwargs["normalize_slang"] = self.config.slang_normalization
                try:
                    mt = self.mt_factory(
                        self.config.nllb_model,
                        self.config.mt_device,
                        self.app_dir / "models" / "nllb-cache",
                        warning,
                        **mt_kwargs,
                    )
                except TypeError:
                    mt = self.mt_factory(
                        self.config.nllb_model,
                        self.config.mt_device,
                        self.app_dir / "models" / "nllb-cache",
                        warning,
                    )
                if target_names:
                    self._emit("preparing", f"Memeriksa inferensi translasi NLLB-200 [{active_targets_str}]…")
                mt.prepare(target_names)
                if self._stop_requested.is_set():
                    return
                self._emit("model_ready", "Model siap", {
                    "model": stt.active_model,
                    "stt_device": stt.active_device,
                    "mt_device": mt.active_device,
                })
                self._loop.call_soon_threadsafe(ready.set)
                processed_count = 0
                last_gc_count = 0
                last_gc_time = time.monotonic()
                while not self._stop_requested.is_set():
                    try:
                        item = utterances.get(timeout=0.25)
                    except queue.Empty:
                        now = time.monotonic()
                        if (processed_count - last_gc_count >= 50) and (now - last_gc_time >= 120.0):
                            gc.collect()
                            last_gc_count = processed_count
                            last_gc_time = now
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
                        newly_learned = self.vocab_manager.observe(transcript.text)
                        if newly_learned and hasattr(stt, "update_hotwords"):
                            updated_hotwords = self.vocab_manager.get_hotwords(
                                user_hotwords=self.config.whisper_hotwords or "",
                            )
                            stt.update_hotwords(updated_hotwords)
                            self._emit("learning", f"Kosakata baru dipelajari: {', '.join(newly_learned)}", {
                                "terms": newly_learned,
                            })
                        do_censor = getattr(self.config, "profanity_filter", False)
                        display_text = censor_text(transcript.text, do_censor)
                        self._emit(
                            "transcript",
                            display_text,
                            {"language": transcript.language, "text": display_text},
                        )
                        source_code = source_nllb_code(
                            self.config.source_language,
                            transcript.language,
                        )

                        # Early passthrough delivery for same-language slots (e.g. native spoken transcript)
                        # Eliminates waiting for other target languages to complete MT
                        early_deliveries: dict[str, str] = {}
                        for t_name in target_names:
                            try:
                                if target_nllb_code(t_name) == source_code:
                                    early_deliveries[t_name] = display_text
                            except Exception:
                                pass
                        if early_deliveries and len(target_names) > len(early_deliveries):
                            loop = self._loop
                            if loop and loop.is_running():
                                loop.call_soon_threadsafe(schedule_publish, early_deliveries, {
                                    "stt_ms": stt_ms, "mt_ms": 0.0, "queued_at": queued_at,
                                    "stt_device": stt.active_device, "mt_device": "early_passthrough",
                                })

                        self._emit("processing", "Menerjemahkan caption…")
                        mt_start = time.monotonic()
                        translations = mt.translate(transcript.text, source_code, target_names)
                        mt_ms = (time.monotonic() - mt_start) * 1000
                        if self._stop_requested.is_set():
                            return

                        if do_censor:
                            translations = {k: censor_text(v, True) for k, v in translations.items()}

                        translations_summary = " | ".join(f"{lang}={text}" for lang, text in translations.items())
                        self._emit("translations", translations_summary, translations)
                        if getattr(self.config, "translation_review_log", False):
                            self._append_review_log(transcript.language, source_code, display_text, translations, stt_ms, mt_ms)

                        loop = self._loop
                        if loop and loop.is_running():
                            loop.call_soon_threadsafe(schedule_publish, translations, {
                                "stt_ms": stt_ms, "mt_ms": mt_ms, "queued_at": queued_at,
                                "stt_device": stt.active_device, "mt_device": mt.active_device,
                            })
                        processed_count += 1
                    except ValueError as exc:
                        if self._stop_requested.is_set():
                            return
                        warning(f"Gagal memproses kalimat ({exc}); melanjutkan sesi streaming")
                    except Exception as exc:
                        if self._stop_requested.is_set():
                            return
                        err_msg = str(exc)
                        if "All Whisper fallbacks failed" in err_msg or "All NLLB fallbacks failed" in err_msg:
                            self._emit("error", f"Fatal inference error: {err_msg}")
                            self.request_stop()
                        else:
                            warning(f"Gagal memproses kalimat ({err_msg}); melanjutkan sesi streaming")
                    finally:
                        utterances.task_done()
            except Exception as exc:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(ready.set)
                if not self._stop_requested.is_set():
                    self._emit("error", f"Inference failed: {exc}")
                    self.request_stop()
            finally:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(ready.set)
                for engine in (mt, stt):
                    if engine is not None:
                        try:
                            engine.close()
                        except Exception as exc:
                            self._emit("error", f"Model cleanup failed: {exc}")
                try:
                    import importlib
                    torch_mod = importlib.import_module("torch")
                    if getattr(getattr(torch_mod, "cuda", None), "is_available", lambda: False)():
                        torch_mod.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                self._emit("inference_idle", "Mesin caption berhenti")

        async def capture_stage() -> None:
            await ready.wait()
            if self._stop_requested.is_set():
                return
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
                        if self._worker and not self._worker.is_alive():
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
                        if self._worker and not self._worker.is_alive():
                            break
                        seconds = len(utterance) / 16_000
                        self._emit(
                            "queued",
                            f"Ucapan {seconds:.1f} d siap diproses",
                            {"seconds": seconds},
                        )
                        try:
                            utterances.put_nowait((utterance, time.monotonic()))
                        except queue.Full:
                            # Drop oldest pending utterance to keep pipeline real-time without stopping
                            try:
                                dropped = utterances.get_nowait()
                                utterances.task_done()
                            except queue.Empty:
                                dropped = None
                            try:
                                utterances.put_nowait((utterance, time.monotonic()))
                                self._emit("warning", "Antrean penuh: ucapan tertua dilewati untuk menjaga latensi real-time")
                            except queue.Full:
                                pass
                except UtteranceTooLongError as exc:
                    self._emit("warning", f"{exc}; memotong segmen dan melanjutkan ucapan baru")
                    if hasattr(vad, "reset"):
                        vad.reset()
                    continue
                except Exception as exc:
                    if self._stop_requested.is_set():
                        break
                    warning(f"Microphone unavailable: {exc}; retrying")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8.0)
                finally:
                    capture.stop()
                    if hasattr(vad, "reset"):
                        vad.reset()

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
                await asyncio.to_thread(worker.join, 1.0)
                if worker.is_alive():
                    self._emit(
                        "background",
                        "Capture berhenti; panggilan model lama belum selesai. Browser Source tetap tersedia jika memakai panel.",
                    )
            if worker is None or not worker.is_alive():
                self._worker = None
