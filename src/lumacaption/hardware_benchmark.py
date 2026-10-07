"""Hardware benchmark module for empirical preset recommendations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable
import numpy as np

from lumacaption.gpu_runtime import is_cuda_available


@dataclass
class BenchmarkResult:
    whisper_rtf_cpu: float
    whisper_rtf_cuda: float | None
    nllb_ms_per_sentence: float
    recommended_preset: str  # "ultra_low", "low", "medium", "high", "ultra"
    details: str


def compute_recommendation(
    rtf_cpu: float,
    rtf_cuda: float | None,
    nllb_ms: float,
) -> str:
    """Determine recommended preset from real-time factors and translation latency.
    
    rtf (Real-Time Factor) represents processed-audio-seconds / wall-clock-seconds.
    Higher is faster (e.g. 4.0x means 5s audio is transcribed in 1.25s).
    """
    best_rtf = max(rtf_cpu, rtf_cuda if rtf_cuda is not None else 0.0)
    has_cuda = rtf_cuda is not None and rtf_cuda > 0.0

    if has_cuda and rtf_cuda >= 6.5 and nllb_ms <= 50.0:
        return "ultra"
    if has_cuda and rtf_cuda >= 4.0 and nllb_ms <= 150.0:
        return "high"
    if best_rtf >= 1.8:
        return "medium"
    if best_rtf >= 1.0:
        return "low"
    return "ultra_low"


def run_hardware_benchmark(
    app_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """Run empirical benchmark measuring STT real-time factor and MT latency."""
    from lumacaption.stt.whisper_engine import WhisperEngine
    from lumacaption.mt.nllb_engine import NllbEngine
    from lumacaption.model_manager import cache_for

    progress = on_progress or (lambda _msg: None)

    sr = 16000
    duration_sec = 5.0
    t_arr = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    # Synthetic speech-like audio converted to int16 mono PCM at 16000 Hz
    float_audio = 0.25 * np.sin(2 * np.pi * 220 * t_arr) + np.random.normal(0, 0.05, len(t_arr))
    synthetic_audio = np.clip(float_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)

    whisper_cache = cache_for(app_dir, "base")

    # 1. Benchmark STT on CPU
    progress("Mengukur kecepatan Speech-to-Text pada CPU...")
    cpu_engine = WhisperEngine(
        model_size="base",
        device="cpu",
        model_cache=whisper_cache,
        compute_type="int8",
        beam_size=1,
    )
    try:
        cpu_engine.transcribe(synthetic_audio[:16000], language="id")
        t0 = time.perf_counter()
        runs = 3
        for _ in range(runs):
            cpu_engine.transcribe(synthetic_audio, language="id")
        elapsed = time.perf_counter() - t0
        avg_sec = elapsed / runs
        rtf_cpu = round(duration_sec / max(0.001, avg_sec), 2)
    finally:
        cpu_engine.close()

    # 2. Benchmark STT on CUDA (if available)
    rtf_cuda: float | None = None
    if is_cuda_available():
        progress("Mengukur kecepatan Speech-to-Text pada NVIDIA CUDA...")
        cuda_engine = WhisperEngine(
            model_size="base",
            device="cuda",
            model_cache=whisper_cache,
            compute_type="auto",
            beam_size=1,
        )
        try:
            cuda_engine.transcribe(synthetic_audio[:16000], language="id")
            t0 = time.perf_counter()
            runs = 3
            for _ in range(runs):
                cuda_engine.transcribe(synthetic_audio, language="id")
            elapsed = time.perf_counter() - t0
            avg_sec = elapsed / runs
            rtf_cuda = round(duration_sec / max(0.001, avg_sec), 2)
        except Exception:
            rtf_cuda = None
        finally:
            cuda_engine.close()

    # 3. Benchmark MT on NLLB
    progress("Mengukur kecepatan Machine Translation NLLB-200...")
    nllb_cache = cache_for(app_dir, "nllb")
    nllb_device = "cuda" if is_cuda_available() else "cpu"
    nllb_engine = NllbEngine(
        model_id_or_path="nllb",
        device=nllb_device,
        cache_dir=nllb_cache,
    )
    try:
        sentences = [
            "Halo semuanya, selamat datang di siaran langsung hari ini.",
            "Semoga hari Anda menyenangkan dan penuh kebahagiaan.",
            "Ini adalah pengujian kecepatan translasi mesin.",
        ]
        nllb_engine.translate("Halo dunia.", "ind_Latn", ["English"])
        t0 = time.perf_counter()
        for s in sentences:
            nllb_engine.translate(s, "ind_Latn", ["English"])
        nllb_elapsed = time.perf_counter() - t0
        nllb_ms = round((nllb_elapsed / len(sentences)) * 1000.0, 1)
    finally:
        nllb_engine.close()

    recommended = compute_recommendation(rtf_cpu, rtf_cuda, nllb_ms)

    parts = [f"STT CPU: {rtf_cpu:.1f}x RT"]
    if rtf_cuda is not None:
        parts.append(f"STT CUDA: {rtf_cuda:.1f}x RT")
    parts.append(f"MT: {nllb_ms:.0f} ms/kalimat")
    details = " | ".join(parts)

    return BenchmarkResult(
        whisper_rtf_cpu=rtf_cpu,
        whisper_rtf_cuda=rtf_cuda,
        nllb_ms_per_sentence=nllb_ms,
        recommended_preset=recommended,
        details=details,
    )
