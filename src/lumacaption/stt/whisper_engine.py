from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import os
from pathlib import Path

import numpy as np

from gpu_runtime import configure_cuda_runtime


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str


class WhisperEngine:
    def __init__(
        self,
        model_size: str,
        device: str,
        model_cache: str | Path,
        on_warning: Callable[[str], None] | None = None,
        *,
        beam_size: int = 3,
        hotwords: str = "",
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.model_cache = str(model_cache)
        self.on_warning = on_warning or (lambda _message: None)
        self._model = None
        self._candidate_index = 0
        self._candidates: list[tuple[str, str]] | None = None
        self.active_model = ""
        self.active_device = ""
        self.beam_size = beam_size
        self.hotwords = hotwords
        self._model_path: str | None = None

    def _build_candidates(self) -> list[tuple[str, str]]:
        configure_cuda_runtime()
        import ctranslate2

        wants_cuda = self.requested_device == "cuda" or (
            self.requested_device == "auto" and ctranslate2.get_cuda_device_count() > 0
        )
        candidates: list[tuple[str, str]] = []
        if wants_cuda:
            candidates.append((self.model_size, "cuda"))
        candidates.append((self.model_size, "cpu"))
        return candidates

    def _release(self) -> None:
        if self._model is not None:
            try:
                self._model.model.unload_model()
            except (AttributeError, RuntimeError):
                pass
        self._model = None
        gc.collect()

    def _resolve_model(self) -> str:
        from faster_whisper.utils import download_model
        from huggingface_hub.errors import LocalEntryNotFoundError

        def complete(path: str) -> bool:
            root = Path(path)
            required = ("model.bin", "config.json", "tokenizer.json")
            return all((root / name).is_file() for name in required) and any(root.glob("vocabulary.*"))

        # A cached snapshot directory can exist while model.bin is still downloading.
        try:
            path = download_model(self.model_size, cache_dir=self.model_cache, local_files_only=True)
        except LocalEntryNotFoundError:
            path = ""
        if not path or not complete(path):
            self.on_warning(f"Mengunduh Whisper {self.model_size}; progres ada di terminal")
            path = download_model(self.model_size, cache_dir=self.model_cache)
        if not complete(path):
            raise RuntimeError(f"Cache Whisper {self.model_size} belum lengkap")
        return path

    def _load_next(self) -> None:
        configure_cuda_runtime()
        from faster_whisper import WhisperModel

        if self._model_path is None:
            self._model_path = self._resolve_model()
        if self._candidates is None:
            self._candidates = self._build_candidates()
        self._release()
        last_error: Exception | None = None
        while self._candidate_index < len(self._candidates):
            model_size, device = self._candidates[self._candidate_index]
            self._candidate_index += 1
            compute_type = "int8_float16" if device == "cuda" else "int8"
            try:
                self._model = WhisperModel(
                    self._model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                    cpu_threads=max(1, min(4, os.cpu_count() or 1)),
                    num_workers=1,
                )
                self.active_model = model_size
                self.active_device = device
                return
            except Exception as exc:
                last_error = exc
                self.on_warning(f"Whisper {model_size}/{device} unavailable: {exc}")
        raise RuntimeError("No Whisper fallback could load") from last_error

    def prepare(self, language: str | None) -> None:
        """Warm the actual encoder/decoder; loading weights alone misses DLL failures."""
        self.transcribe(np.zeros(16_000, dtype=np.int16), language)

    def transcribe(self, pcm: np.ndarray, language: str | None) -> Transcript:
        if pcm.dtype != np.int16 or pcm.ndim != 1 or not pcm.size:
            raise ValueError("Whisper expects nonempty mono int16 PCM at 16000 Hz")
        audio = pcm.astype(np.float32, copy=False) / 32768.0
        peak = float(np.max(np.abs(audio)))
        if 0.001 < peak < 0.65:
            audio = audio * min(0.8 / peak, 20.0)

        prompt = None
        if language in ("id", None):
            prompt = (
                "Transkripsi percakapan bahasa Indonesia santai, tidak baku, gaul, "
                "dan campuran bahasa Jawa sehari-hari (seperti ya opo, piye, ora, wes, gak, nggih, monggo, rek)."
            )
        elif language == "jw":
            prompt = "Transkripsi pacelathon basa Jawa ngoko lan krama sehari-hari."
        if self.hotwords:
            prompt = f"{self.hotwords}. {prompt}" if prompt else self.hotwords

        last_error: Exception | None = None
        while True:
            if self._model is None:
                self._load_next()
                self.on_warning(f"Whisper active: {self.active_model}/{self.active_device}")
            try:
                segments, info = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=self.beam_size,
                    best_of=self.beam_size,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    vad_filter=False,
                    initial_prompt=prompt,
                    hotwords=self.hotwords or None,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                return Transcript(text=text, language=info.language)
            except Exception as exc:
                last_error = exc
                failed = f"{self.active_model}/{self.active_device}"
                self.on_warning(f"Whisper {failed} inference failed: {exc}; falling back")
                self._release()
                if self._candidates is not None and self._candidate_index >= len(self._candidates):
                    raise RuntimeError("All Whisper fallbacks failed") from last_error

    def close(self) -> None:
        self._release()
