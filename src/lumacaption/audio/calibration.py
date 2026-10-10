from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

from lumacaption.audio.capture import MicrophoneCapture
from lumacaption.audio.vad import SileroOnnx, VadSegmenter


@dataclass
class AudioCalibrationResult:
    max_peak: float
    peak_dbfs: float
    rms_dbfs: float
    speech_detected_sec: float
    speech_ratio: float
    avg_speech_prob: float
    max_noise_prob: float
    utterances_count: int
    volume_status: str  # "quiet", "optimal", "loud"
    vad_status: str     # "unresponsive", "optimal", "noisy"
    recommended_gain_db: float | None = None
    recommended_vad_threshold: float | None = None
    recommended_auto_normalize: bool | None = None
    recommended_denoise_engine: str | None = None
    summary_text: str = ""
    recommendations: list[str] | None = None


def evaluate_calibration(
    peaks: list[float],
    rms_list: list[float],
    vad_probs: list[float],
    current_gain_db: float,
    current_vad_threshold: float,
    current_normalize: bool = True,
    utterances_count: int = 0,
    lang: str = "id",
    current_denoise_engine: str = "clarity",
) -> AudioCalibrationResult:
    """Analyze calibration metrics and generate actionable tuning recommendations."""
    is_en = (lang == "en")
    total_frames = max(1, len(peaks))
    max_peak = max(peaks) if peaks else 0.0
    mean_rms = float(np.mean(rms_list)) if rms_list else 0.0

    peak_dbfs = 20.0 * math.log10(max(1e-5, max_peak))
    rms_dbfs = 20.0 * math.log10(max(1e-5, mean_rms))

    speech_indices = [i for i, prob in enumerate(vad_probs) if prob >= current_vad_threshold]
    noise_indices = [i for i, prob in enumerate(vad_probs) if prob < current_vad_threshold]

    speech_frames = len(speech_indices)
    speech_detected_sec = speech_frames * 0.032
    speech_ratio = speech_frames / total_frames

    avg_speech_prob = float(np.mean([vad_probs[i] for i in speech_indices])) if speech_indices else 0.0
    # Peak noise prob evaluated when audio is quiet
    quiet_noise_probs = [vad_probs[i] for i in noise_indices if peaks[i] < 0.15]
    max_noise_prob = max(quiet_noise_probs) if quiet_noise_probs else 0.0

    # 1. Volume evaluation
    rec_gain: float | None = None
    rec_norm: bool | None = None
    rec_denoise: str | None = None
    recs: list[str] = []

    if max_peak < 0.20:
        volume_status = "quiet"
        diff = max(3.0, min(12.0, round((0.40 - max_peak) * 20.0)))
        rec_gain = min(18.0, current_gain_db + diff)
        if not current_normalize:
            rec_norm = True
        recs.append(
            f"Gain mic terlalu rendah ({peak_dbfs:.1f} dBFS). Sarankan naikkan Gain mic ke +{rec_gain:.1f} dB."
            if not is_en else
            f"Mic level is too quiet ({peak_dbfs:.1f} dBFS). Recommend raising Mic Gain to +{rec_gain:.1f} dB."
        )
    elif max_peak > 0.95:
        volume_status = "loud"
        rec_gain = max(-24.0, current_gain_db - 4.0)
        recs.append(
            f"Sinyal mic terlalu keras ({peak_dbfs:.1f} dBFS, potensi clipping). Sarankan turunkan Gain ke {rec_gain:.1f} dB."
            if not is_en else
            f"Mic level is clipping ({peak_dbfs:.1f} dBFS). Recommend reducing Mic Gain to {rec_gain:.1f} dB."
        )
    else:
        volume_status = "optimal"

    # 2. VAD sensitivity evaluation
    rec_vad: float | None = None
    if speech_detected_sec < 0.25 and max_peak >= 0.15:
        vad_status = "unresponsive"
        rec_vad = max(0.20, round(current_vad_threshold - 0.15, 2))
        recs.append(
            f"VAD tidak menangkap ucapan meskipun ada sinyal audio. Sarankan turunkan Ambang VAD ke {rec_vad:.2f}."
            if not is_en else
            f"VAD missed speech despite audio input. Recommend lowering VAD Threshold to {rec_vad:.2f}."
        )
    elif max_noise_prob >= (current_vad_threshold - 0.05) and max_noise_prob > 0.40:
        vad_status = "noisy"
        rec_vad = min(0.85, round(max_noise_prob + 0.15, 2))
        if current_denoise_engine != "hybrid":
            rec_denoise = "hybrid"
        recs.append(
            f"Noise lantai/derau latar cukup tinggi ({max_noise_prob*100:.0f}%). Sarankan naikkan Ambang VAD ke {rec_vad:.2f} atau gunakan Noise Suppressor 'hybrid'."
            if not is_en else
            f"Background noise is high ({max_noise_prob*100:.0f}%). Recommend increasing VAD Threshold to {rec_vad:.2f} or using 'hybrid' Noise Suppressor."
        )
    else:
        vad_status = "optimal"

    # 3. Fragmented speech / silence gap indication
    if utterances_count >= 2 and speech_ratio > 0.55:
        recs.append(
            f"Kalimat Anda kemungkinan terpotong {utterances_count}x selama tes — coba naikkan Silence Gap (ms) sedikit kalau ucapan Anda sering terjeda napas pendek. (Indikasi awal, uji beberapa kali untuk memastikan.)"
            if not is_en else
            f"Your speech was possibly split into {utterances_count} pieces during the test — try raising Silence Gap (ms) slightly if you often pause briefly to breathe. (Early indication only, retest to confirm.)"
        )

    summary_parts = []
    if is_en:
        summary_parts.append(f"Peak: {max_peak:.2f} ({peak_dbfs:.1f} dBFS) • RMS: {rms_dbfs:.1f} dBFS")
        summary_parts.append(f"Speech detected: {speech_detected_sec:.1f}s ({speech_ratio*100:.0f}%)")
        if utterances_count > 0:
            summary_parts.append(f"Utterances segmented: {utterances_count}")
    else:
        summary_parts.append(f"Puncak: {max_peak:.2f} ({peak_dbfs:.1f} dBFS) • Rata-rata: {rms_dbfs:.1f} dBFS")
        summary_parts.append(f"Ucapan terdeteksi: {speech_detected_sec:.1f}s ({speech_ratio*100:.0f}%)")
        if utterances_count > 0:
            summary_parts.append(f"Kalimat terpotong: {utterances_count}")

    summary_text = " • ".join(summary_parts)

    return AudioCalibrationResult(
        max_peak=round(max_peak, 3),
        peak_dbfs=round(peak_dbfs, 1),
        rms_dbfs=round(rms_dbfs, 1),
        speech_detected_sec=round(speech_detected_sec, 2),
        speech_ratio=round(speech_ratio, 3),
        avg_speech_prob=round(avg_speech_prob, 3),
        max_noise_prob=round(max_noise_prob, 3),
        utterances_count=utterances_count,
        volume_status=volume_status,
        vad_status=vad_status,
        recommended_gain_db=rec_gain,
        recommended_vad_threshold=rec_vad,
        recommended_auto_normalize=rec_norm,
        recommended_denoise_engine=rec_denoise,
        summary_text=summary_text,
        recommendations=recs,
    )


def run_audio_calibration(
    app_dir: Path,
    device_key: str | int | None,
    channel: str = "mix",
    gain_db: float = 0.0,
    clarity: bool = True,
    denoise_engine: str = "clarity",
    normalize_audio: bool = True,
    vad_threshold: float = 0.5,
    silence_gap_ms: int = 300,
    max_duration_sec: int = 20,
    duration_sec: float = 5.0,
    on_progress: Callable[[float, float, float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    capture_factory: Any = None,
    silero_factory: Any = None,
    lang: str = "id",
) -> AudioCalibrationResult:
    """Capture live microphone audio for duration_sec, analyze with Silero VAD, and return evaluation."""
    com = None
    if sys.platform == "win32":
        try:
            import ctypes
            res = ctypes.windll.ole32.CoInitializeEx(None, 0)
            if res in (0, 1):
                com = ctypes.windll.ole32
        except Exception:
            pass

    capture_cls = capture_factory or MicrophoneCapture
    silero_cls = silero_factory or SileroOnnx

    silero_model_path = app_dir / "models" / "silero_vad.onnx"
    silero = silero_cls(silero_model_path)

    vad = VadSegmenter(
        model=silero,
        threshold=vad_threshold,
        min_silence_ms=silence_gap_ms,
        max_utterance_seconds=max_duration_sec,
    )

    capture = capture_cls(
        device=device_key,
        channel=channel,
        gain_db=gain_db,
        clarity=clarity,
        denoise_engine=denoise_engine,
        model_dir=app_dir,
        normalize_audio=normalize_audio,
    )

    peaks: list[float] = []
    rms_list: list[float] = []
    vad_probs: list[float] = []
    utterances_count = 0

    capture.start()
    start_time = time.monotonic()
    try:
        while time.monotonic() - start_time < duration_sec:
            if is_cancelled and is_cancelled():
                break

            try:
                frame = capture._frames.get(timeout=0.1)
            except Exception:
                continue

            if not isinstance(frame, np.ndarray) or frame.size == 0:
                continue

            float_frame = frame.astype(np.float32) / 32768.0
            peak = float(np.max(np.abs(float_frame)))
            rms = float(np.sqrt(np.mean(float_frame * float_frame)))
            peaks.append(peak)
            rms_list.append(rms)

            try:
                utt = vad.process(frame)
                if utt is not None:
                    utterances_count += 1
            except Exception:
                pass

            prob = float(getattr(vad, "last_probability", 0.0))
            vad_probs.append(prob)

            elapsed = time.monotonic() - start_time
            remaining = max(0.0, duration_sec - elapsed)
            if on_progress:
                on_progress(remaining, peak, prob)
    finally:
        capture.stop()
        if com:
            try:
                com.CoUninitialize()
            except Exception:
                pass

    return evaluate_calibration(
        peaks=peaks,
        rms_list=rms_list,
        vad_probs=vad_probs,
        current_gain_db=gain_db,
        current_vad_threshold=vad_threshold,
        current_normalize=normalize_audio,
        utterances_count=utterances_count,
        lang=lang,
        current_denoise_engine=denoise_engine,
    )
