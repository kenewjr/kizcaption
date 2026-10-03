from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import re

import numpy as np

from lumacaption.gpu_runtime import configure_cuda_runtime


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str



_HALLUCINATION_PATTERNS = [
    re.compile(r"\[(?:musik|music|tertawa|applause|laughter|suara|hening|batuk|nafas|desah).*?\]", re.IGNORECASE),
    re.compile(r"\((?:musik|music|tertawa|applause|laughter|suara|hening|batuk|nafas|desah).*?\)", re.IGNORECASE),
    re.compile(r"terima\s+kasih\s+(?:sudah|telah)\s+menonton[.!?,]*", re.IGNORECASE),
    re.compile(r"thanks\s+for\s+watching[.!?,]*", re.IGNORECASE),
    re.compile(r"subtitles\s+by\s+[^\n.,]+", re.IGNORECASE),
    re.compile(r"(?:jangan\s+lupa\s+)?(?:like|subscribe|share)(?:\s+(?:and|dan|atau|or|&)?\s+(?:like|subscribe|share|comment|komentar))*[.!?,]*", re.IGNORECASE),
]


DANGLING_CONJUNCTIONS = {
    "dan", "atau", "tapi", "tetapi", "karena", "soalnya", "terus", "lalu",
    "jadi", "kalau", "bahwa", "sedangkan", "namun",
    "and", "or", "but", "because", "so", "then", "that", "if",
}


def trim_dangling_conjunctions(text: str) -> str:
    """Trims trailing dangling conjunctions caused by abrupt VAD cuts."""
    if not text:
        return ""
    words = text.split()
    while len(words) >= 3:
        clean_last = re.sub(r"[.!?…,:;\"'“”]+$", "", words[-1]).lower()
        if clean_last in DANGLING_CONJUNCTIONS:
            m = re.search(r"([.!?…]+)$", words[-1])
            punct = m.group(1) if m else ""
            words.pop()
            if punct and not re.search(r"[.!?…]$", words[-1]):
                words[-1] += punct
        else:
            break
    return " ".join(words).strip()


def sanitize_transcript(text: str) -> str:
    cleaned = text
    for pat in _HALLUCINATION_PATTERNS:
        cleaned = pat.sub("", cleaned)
    # Collapse repetitive word loops: >= 3 repetitions -> single
    cleaned = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", cleaned, flags=re.IGNORECASE)
    # Collapse 2-word phrase loops: repeated phrases -> single
    cleaned = re.sub(r"\b(\w+\s+\w+)(?:\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return trim_dangling_conjunctions(cleaned)


WHISPER_CUSTOM_MODELS: dict[str, str] = {
    "whisper-small-id": "ammaraldirawi/faster-whisper-small-id-int8",
    "whisper-medium-id": "cahya/faster-whisper-medium-id",
}


def resolve_whisper_repo(model_key: str) -> str:
    return WHISPER_CUSTOM_MODELS.get(model_key, model_key)


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
        compute_type: str = "auto",
        cpu_threads: int = 4,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.model_cache = str(model_cache)
        self.on_warning = on_warning or (lambda _message: None)
        self._model = None
        self._candidate_index = 0
        self._candidates: list[tuple[str, str]] | None = None
        self.active_model = ""
        self.active_device = ""
        self.active_compute_type = ""
        self.beam_size = beam_size
        self.hotwords = hotwords
        self._model_path: str | None = None

    def _resolve_compute_type(self, device: str) -> str:
        if device == "cpu":
            return "int8"
        if self.requested_compute_type != "auto":
            return self.requested_compute_type
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda", 0)
            if "int8_float16" in supported:
                return "int8_float16"
            if "float16" in supported:
                return "float16"
            if "int8_float32" in supported:
                return "int8_float32"
            return "float32"
        except Exception:
            return "int8_float16"

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

        def complete(path: str) -> bool:
            root = Path(path)
            required = ("model.bin", "config.json", "tokenizer.json")
            return all((root / name).is_file() for name in required) and any(root.glob("vocabulary.*"))

        # 1. Direct path check if user passed a local folder
        direct_path = Path(self.model_size).expanduser()
        if direct_path.is_dir() and complete(str(direct_path)):
            return str(direct_path.resolve())

        # 2. Check local cache via model_manager inspect_model
        try:
            from lumacaption.model_manager import inspect_model
            state, local_path = inspect_model(self.model_size, Path(self.model_cache))
            if state == "Tersedia lokal" and local_path is not None and complete(str(local_path)):
                return str(local_path.resolve())
        except Exception:
            pass

        target = resolve_whisper_repo(self.model_size)

        # 3. Try local cache only first
        path = ""
        try:
            path = download_model(target, cache_dir=self.model_cache, local_files_only=True)
        except Exception:
            path = ""

        # 4. If not available locally or incomplete, download from Hugging Face
        if not path or not complete(path):
            self.on_warning(f"Mengunduh Whisper {self.model_size}; progres ada di terminal")
            path = download_model(target, cache_dir=self.model_cache)

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
            compute_type = self._resolve_compute_type(device)
            try:
                self._model = WhisperModel(
                    self._model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                    cpu_threads=max(1, min(self.cpu_threads, os.cpu_count() or 1)),
                    num_workers=1,
                )
                self.active_model = model_size
                self.active_device = device
                self.active_compute_type = compute_type
                return
            except Exception as exc:
                last_error = exc
                self.on_warning(f"Whisper {model_size}/{device} unavailable: {exc}")
        raise RuntimeError("No Whisper fallback could load") from last_error

    def prepare(self, language: str | None) -> None:
        """Warm the actual encoder/decoder; loading weights alone misses DLL failures."""
        if self._model is None:
            self._load_next()
        try:
            silence = np.zeros(4000, dtype=np.float32)
            segments, _ = self._model.transcribe(
                silence,
                language=language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
            )
            for _ in segments:
                pass
        except Exception as exc:
            self.on_warning(f"Whisper warmup note: {exc}")

    def transcribe(self, pcm: np.ndarray, language: str | None) -> Transcript:
        if pcm.dtype != np.int16 or pcm.ndim != 1 or not pcm.size:
            raise ValueError("Whisper expects nonempty mono int16 PCM at 16000 Hz")
        audio = pcm.astype(np.float32, copy=False) / 32768.0
        peak = float(np.max(np.abs(audio)))
        if 0.02 < peak < 0.65:
            audio = audio * min(0.8 / peak, 3.5)
        elif peak > 0.92:
            audio = audio * (0.85 / peak)

        prompt = None
        if language in ("id", None):
            if self.hotwords:
                prompt = f"Halo, siaran langsung berbahasa Indonesia santai. Kosakata: {self.hotwords}."
            else:
                prompt = "Halo, selamat datang di siaran langsung. Pembicara berbicara dalam bahasa Indonesia dengan jelas dan artikulasi setiap kata yang runtut."
        elif language == "jw":
            if self.hotwords:
                prompt = f"Sugeng rawuh ing siaran langsung iki. Tembung: {self.hotwords}."
            else:
                prompt = "Sugeng rawuh ing siaran langsung iki."
        elif self.hotwords:
            prompt = self.hotwords

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
                    repetition_penalty=1.15,
                    no_speech_threshold=0.5,
                    hallucination_silence_threshold=2.0,
                )
                valid_segments: list[str] = []
                for segment in segments:
                    if getattr(segment, "compression_ratio", 1.0) > 2.4:
                        continue
                    if getattr(segment, "no_speech_prob", 0.0) > 0.65:
                        continue
                    raw_text = getattr(segment, "text", "") or ""
                    if getattr(segment, "avg_logprob", 0.0) < -1.2 and len(raw_text.strip()) < 8:
                        continue
                    cleaned = sanitize_transcript(raw_text)
                    if cleaned:
                        valid_segments.append(cleaned)
                text = " ".join(valid_segments).strip()
                return Transcript(text=text, language=info.language)
            except Exception as exc:
                last_error = exc
                failed = f"{self.active_model}/{self.active_device}"
                self.on_warning(f"Whisper {failed} inference failed: {exc}; falling back")
                self._release()
                if self._candidates is not None and self._candidate_index >= len(self._candidates):
                    raise RuntimeError("All Whisper fallbacks failed") from last_error

    def update_hotwords(self, hotwords: str) -> None:
        self.hotwords = hotwords

    def close(self) -> None:
        self._release()
