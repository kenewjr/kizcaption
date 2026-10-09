from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
import queue
import re
import sys
import threading
import time
from typing import Any

import numpy as np

from lumacaption.audio.denoiser import DtlnDenoiser

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
_STOP = object()


@dataclass(frozen=True, slots=True)
class InputDevice:
    key: str
    name: str
    index: int
    sample_rate: int
    host_api: str
    is_default: bool = False
    input_channels: int = 1


def soft_limit(x: np.ndarray, threshold: float = 0.70, ceiling: float = 0.98) -> np.ndarray:
    """Smooth C1-continuous soft-knee saturation curve.
    Linear and bit-exact below threshold.
    Gracefully and smoothly compresses above threshold without any harsh clipping harmonics."""
    over = np.abs(x) > threshold
    if not np.any(over):
        return x
    y = x.copy()
    k = ceiling - threshold
    xo = x[over]
    y[over] = np.sign(xo) * (threshold + k * np.tanh((np.abs(xo) - threshold) / k))
    return y


class VocalClarityProcessor:
    """Real-time vocal enhancement: 80Hz rumble cut + 3.2kHz presence boost for speech intelligibility."""

    def __init__(self, sample_rate: float = 16000.0) -> None:
        fs = float(sample_rate) if sample_rate > 0 else 16000.0
        # 80 Hz 2nd-order Butterworth High-pass (cleans rumble, breath pops & DC)
        w0_hp = 2.0 * math.pi * 80.0 / fs
        alpha_hp = math.sin(w0_hp) / (2.0 * 0.70710678)
        cos_hp = math.cos(w0_hp)
        a0_hp = 1.0 + alpha_hp
        self.hb0 = (1.0 + cos_hp) / (2.0 * a0_hp)
        self.hb1 = -(1.0 + cos_hp) / a0_hp
        self.hb2 = (1.0 + cos_hp) / (2.0 * a0_hp)
        self.ha1 = (-2.0 * cos_hp) / a0_hp
        self.ha2 = (1.0 - alpha_hp) / a0_hp
        self.hx1 = self.hx2 = self.hy1 = self.hy2 = 0.0

        # 3.2 kHz Speech Presence Boost (+3.0 dB, Q=1.0) (sharpens consonant clarity & word boundaries)
        A = 10.0 ** (3.0 / 40.0)
        w0_eq = 2.0 * math.pi * 3200.0 / fs
        alpha_eq = math.sin(w0_eq) / (2.0 * 1.0)
        cos_eq = math.cos(w0_eq)
        a0_eq = 1.0 + alpha_eq / A
        self.eb0 = (1.0 + alpha_eq * A) / a0_eq
        self.eb1 = (-2.0 * cos_eq) / a0_eq
        self.eb2 = (1.0 - alpha_eq * A) / a0_eq
        self.ea1 = (-2.0 * cos_eq) / a0_eq
        self.ea2 = (1.0 - alpha_eq / A) / a0_eq
        self.ex1 = self.ex2 = self.ey1 = self.ey2 = 0.0

        # Vectorized 4th-order combined biquad coefficients (HP 80Hz * Presence 3.2kHz)
        bh = np.array([self.hb0, self.hb1, self.hb2], dtype=np.float64)
        ah = np.array([1.0, self.ha1, self.ha2], dtype=np.float64)
        be = np.array([self.eb0, self.eb1, self.eb2], dtype=np.float64)
        ae = np.array([1.0, self.ea1, self.ea2], dtype=np.float64)
        self._B = np.convolve(bh, be)
        self._A = np.convolve(ah, ae)
        self._a1 = float(self._A[1])
        self._a2 = float(self._A[2])
        self._a3 = float(self._A[3])
        self._a4 = float(self._A[4])
        self._prev_x = np.zeros(4, dtype=np.float64)
        self._prev_y = [0.0, 0.0, 0.0, 0.0]

        self.gate_threshold = 0.002  # -54 dBFS quiet floor threshold
        self.expander_gain = 1.0

    def reset(self) -> None:
        self.hx1 = self.hx2 = self.hy1 = self.hy2 = 0.0
        self.ex1 = self.ex2 = self.ey1 = self.ey2 = 0.0
        self._prev_x.fill(0.0)
        self._prev_y = [0.0, 0.0, 0.0, 0.0]
        self.expander_gain = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if not x.size:
            return x
        # Vectorized FIR numerator across entire chunk via compiled NumPy convolution
        x_pad = np.concatenate((self._prev_x, x.astype(np.float64, copy=False)))
        self._prev_x = x_pad[-4:]
        v = np.convolve(x_pad, self._B, mode="valid")

        # Recursive 4th-order IIR denominator
        y = np.empty_like(v)
        y0, y1, y2, y3 = self._prev_y[3], self._prev_y[2], self._prev_y[1], self._prev_y[0]
        a1, a2, a3, a4 = self._a1, self._a2, self._a3, self._a4
        for i in range(len(v)):
            yi = v[i] - (a1 * y0 + a2 * y1 + a3 * y2 + a4 * y3)
            y3, y2, y1, y0 = y2, y1, y0, yi
            y[i] = yi
        self._prev_y = [y3, y2, y1, y0]

        # Downward expander: smooth soft noise gate during silence
        rms = float(np.sqrt(np.mean(y * y)))
        if rms < self.gate_threshold and rms > 1e-6:
            target = max(0.50, (rms / self.gate_threshold) ** 0.8)
        elif rms <= 1e-6:
            target = 0.50
        else:
            target = 1.0
        alpha = 0.4 if target > self.expander_gain else 0.85
        self.expander_gain = alpha * self.expander_gain + (1.0 - alpha) * target
        if self.expander_gain < 0.999:
            y = y * self.expander_gain
        return y.astype(x.dtype, copy=False)


def float_audio_to_mono_pcm16(
    audio: Any,
    channel: str = "mix",
    gain_db: float = 0.0,
    soft_clip: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Downmix native float audio and return PCM16 plus unclipped meter values."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 0:
        samples = samples.reshape(1)
    elif samples.ndim > 1:
        if channel == "left":
            samples = samples[:, 0]
        elif channel == "right" and samples.shape[1] > 1:
            samples = samples[:, 1]
        else:
            samples = np.mean(samples, axis=1, dtype=np.float32)
    mono = samples.reshape(-1)
    if gain_db != 0.0:
        mono = mono * (10.0 ** (gain_db / 20.0))
    if mono.size:
        rms = float(np.sqrt(np.mean(mono * mono)))
        peak = float(np.max(np.abs(mono)))
    else:
        rms = peak = 0.0
    if soft_clip:
        limited = soft_limit(mono)
    else:
        limited = np.clip(mono, -1.0, 1.0)
    pcm = np.rint(limited * 32767.0).astype(np.int16)
    return pcm, rms, peak


class AudioFrameConverter:
    """Streaming linear resampler plus exact 512-sample frame packer."""

    def __init__(self, input_rate: int) -> None:
        if input_rate <= 0:
            raise ValueError("Input sample rate must be positive")
        self.input_rate = input_rate
        self._step = input_rate / SAMPLE_RATE
        self._source = np.empty(0, dtype=np.float32)
        self._position = 0.0
        self._pending = np.empty(0, dtype=np.int16)

    def process(self, pcm: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if self.input_rate == SAMPLE_RATE:
            converted = samples.copy()
        else:
            self._source = np.concatenate((self._source, samples.astype(np.float32)))
            available = len(self._source) - 1 - self._position
            count = math.floor(available / self._step) + 1 if available >= 0 else 0
            if count <= 0:
                return []
            positions = self._position + self._step * np.arange(count)
            converted = np.rint(
                np.interp(positions, np.arange(len(self._source)), self._source)
            ).clip(-32768, 32767).astype(np.int16)
            next_position = self._position + count * self._step
            consumed = min(math.floor(next_position), len(self._source))
            self._source = self._source[consumed:]
            self._position = next_position - consumed

        if self._pending.size:
            converted = np.concatenate((self._pending, converted))
        frame_count = len(converted) // FRAME_SAMPLES
        frames = [
            converted[index * FRAME_SAMPLES:(index + 1) * FRAME_SAMPLES].copy()
            for index in range(frame_count)
        ]
        self._pending = converted[frame_count * FRAME_SAMPLES:].copy()
        return frames


class MicrophoneCapture:
    def __init__(
        self,
        device: str | int | None,
        on_warning: Callable[[str], None] | None = None,
        on_level: Callable[[float, float], None] | None = None,
        queue_frames: int = 128,
        channel: str = "mix",
        gain_db: float = 0.0,
        clarity: bool = True,
        denoise_engine: str = "clarity",
        model_dir: Any = None,
        normalize_audio: bool = True,
    ) -> None:
        self.device = device
        self.on_warning = on_warning or (lambda _message: None)
        self.on_level = on_level or (lambda _rms, _peak: None)
        self.channel = channel
        self.gain_db = gain_db
        self.clarity = clarity
        self.denoise_engine = denoise_engine
        self.model_dir = model_dir
        self.normalize_audio = normalize_audio
        self._auto_gain = 1.0
        self._frames: queue.Queue[Any] = queue.Queue(maxsize=queue_frames)
        self._stream: Any | None = None
        self._closed = threading.Event()
        self._last_frame_at = 0.0
        self._last_level_at = 0.0
        self._overflow_reported = False
        self._converter: AudioFrameConverter | None = None
        self._dsp: VocalClarityProcessor | None = VocalClarityProcessor() if (clarity and denoise_engine in ("clarity", "hybrid")) else None
        self._denoiser: DtlnDenoiser | None = DtlnDenoiser(model_dir) if denoise_engine in ("dtln", "hybrid") else None
        self.active_device: InputDevice | None = None

    @staticmethod
    def _key(name: str) -> str:
        normalized = re.sub(r"\s+", " ", name).strip().casefold()
        return f"device:{normalized}"

    @classmethod
    def devices(cls) -> list[InputDevice]:
        """Return OBS-like Windows endpoints instead of every PortAudio host duplicate."""
        import sounddevice as sd

        raw_devices = list(sd.query_devices())
        host_apis = list(sd.query_hostapis())
        wasapi_hosts = {
            index for index, host in enumerate(host_apis)
            if "wasapi" in str(host["name"]).casefold()
        }
        candidates: list[InputDevice] = []
        for index, info in enumerate(raw_devices):
            if int(info["max_input_channels"]) <= 0:
                continue
            host_index = int(info["hostapi"])
            if wasapi_hosts and host_index not in wasapi_hosts:
                continue
            name = re.sub(r"\s+", " ", str(info["name"])).strip()
            if name.casefold() in {
                "microsoft sound mapper - input",
                "primary sound capture driver",
            }:
                continue
            candidates.append(InputDevice(
                cls._key(name),
                name,
                index,
                max(1, round(float(info["default_samplerate"]))),
                str(host_apis[host_index]["name"]),
                input_channels=max(1, int(info["max_input_channels"])),
            ))

        # Some PortAudio builds expose a WASAPI host with no usable input.
        if not candidates and wasapi_hosts:
            for index, info in enumerate(raw_devices):
                if int(info["max_input_channels"]) <= 0:
                    continue
                host_index = int(info["hostapi"])
                host_name = str(host_apis[host_index]["name"])
                if "directsound" not in host_name.casefold() and host_index != 0:
                    continue
                name = re.sub(r"\s+", " ", str(info["name"])).strip()
                if name.casefold() in {
                    "microsoft sound mapper - input",
                    "primary sound capture driver",
                }:
                    continue
                candidates.append(InputDevice(
                    cls._key(name), name, index,
                    max(1, round(float(info["default_samplerate"]))), host_name,
                    input_channels=max(1, int(info["max_input_channels"])),
                ))

        unique: dict[str, InputDevice] = {}
        for item in candidates:
            unique.setdefault(item.key, item)
        endpoints = sorted(unique.values(), key=lambda item: item.name.casefold())

        default_index = -1
        if wasapi_hosts:
            default_index = int(host_apis[next(iter(wasapi_hosts))]["default_input_device"])
        if default_index < 0:
            default_index = int(sd.default.device[0])
        if 0 <= default_index < len(raw_devices):
            info = raw_devices[default_index]
            host_name = str(host_apis[int(info["hostapi"])]["name"])
            sample_rate = max(1, round(float(info["default_samplerate"])))
            # Show one friendly default plus concrete endpoints. Some PortAudio builds
            # alias default_index to one endpoint; stable key still keeps selection exact.
            default_entry = InputDevice(
                "__default__",
                "Default Windows",
                default_index,
                sample_rate,
                host_name,
                True,
                max(1, int(info["max_input_channels"])),
            )
            return [default_entry, *endpoints]
        return endpoints

    @classmethod
    def resolve_device(cls, selection: str | int | None) -> InputDevice:
        devices = cls.devices()
        if not devices:
            raise RuntimeError("No microphone input found")
        default = next((item for item in devices if item.is_default), devices[0])
        if selection is None or selection == "__default__":
            return default
        if isinstance(selection, int):
            return next((item for item in devices if item.index == selection), default)
        wanted = selection.strip().casefold()
        exact = next(
            (
                item
                for item in devices
                if item.key.casefold() == wanted or item.name.casefold() == wanted
            ),
            None,
        )
        if exact:
            return exact
        fuzzy = [
            item
            for item in devices
            if not item.is_default
            and (wanted in item.name.casefold() or item.key.casefold() in wanted)
        ]
        return fuzzy[0] if len(fuzzy) == 1 else default

    def _enqueue(self, frame: np.ndarray) -> None:
        try:
            self._frames.put_nowait(frame)
            self._overflow_reported = False
        except queue.Full:
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(frame)
            except queue.Empty:
                pass
            if not self._overflow_reported:
                self.on_warning("Audio backlog full; dropped oldest frame")
                self._overflow_reported = True

    def _callback(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        if self._closed.is_set():
            return
        try:
            if status:
                self.on_warning(f"Microphone status: {status}")
            samples = np.asarray(indata, dtype=np.float32)
            if samples.ndim == 0:
                samples = samples.reshape(1)
            elif samples.ndim > 1:
                if self.channel == "left":
                    samples = samples[:, 0]
                elif self.channel == "right" and samples.shape[1] > 1:
                    samples = samples[:, 1]
                else:
                    samples = np.mean(samples, axis=1, dtype=np.float32)
            mono = samples.reshape(-1)
            # Apply user gain first so quiet inputs are boosted before DSP/VAD
            if self.gain_db != 0.0:
                mono = mono * (10.0 ** (self.gain_db / 20.0))

            # Auto-normalize signal level so quiet mics (e.g. SteelSeries Sonar)
            # reach healthy dynamic range (-18 to -9 dBFS) instead of being buried in noise floor
            if self.normalize_audio and mono.size:
                raw_peak = float(np.max(np.abs(mono)))
                if raw_peak > 0.0003:  # Above silence/room hiss
                    target_peak = 0.35  # ~ -9 dBFS optimal working level
                    desired_gain = min(target_peak / raw_peak, 20.0)  # up to +26 dB boost
                    self._auto_gain = 0.85 * self._auto_gain + 0.15 * desired_gain
                else:
                    self._auto_gain = 0.98 * self._auto_gain + 0.02 * 1.0
                mono = mono * self._auto_gain

            # Process denoise and presence filter on properly leveled audio
            if self._denoiser is not None and self._denoiser.is_ready:
                try:
                    mono = self._denoiser.process(mono)
                except Exception as exc:
                    self.on_warning(f"Denoiser error: {exc}")
            if self._dsp is not None:
                try:
                    mono = self._dsp.process(mono)
                except Exception as exc:
                    self.on_warning(f"Vocal clarity DSP error: {exc}")

            if mono.size:
                rms = float(np.sqrt(np.mean(mono * mono)))
                peak = float(np.max(np.abs(mono)))
            else:
                rms = peak = 0.0
            limited = soft_limit(mono)
            pcm_samples = np.rint(limited * 32767.0).astype(np.int16)
            self._last_frame_at = time.monotonic()
            now = self._last_frame_at
            if now - self._last_level_at >= 0.08 and pcm_samples.size:
                self.on_level(rms, peak)
                self._last_level_at = now
            if self._converter is not None:
                for frame in self._converter.process(pcm_samples):
                    self._enqueue(frame)
        except Exception as exc:
            self.on_warning(f"Audio callback error: {exc}")

    def start(self) -> InputDevice:
        import sounddevice as sd

        if self._stream is not None:
            assert self.active_device is not None
            return self.active_device
        selected = self.resolve_device(self.device)
        self.active_device = selected
        self._converter = AudioFrameConverter(selected.sample_rate)
        if self.clarity and self.denoise_engine in ("clarity", "hybrid"):
            self._dsp = VocalClarityProcessor(selected.sample_rate)
        else:
            self._dsp = None
        if self.denoise_engine in ("dtln", "hybrid"):
            self._denoiser = DtlnDenoiser(self.model_dir)
        else:
            self._denoiser = None
        self._closed.clear()
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.ole32.CoInitializeEx(None, 0)
            except Exception:
                pass

        kwargs: dict[str, Any] = {
            "samplerate": selected.sample_rate,
            "blocksize": max(128, round(selected.sample_rate * FRAME_SAMPLES / SAMPLE_RATE)),
            "device": selected.index,
            "channels": min(2, selected.input_channels),
            "dtype": "float32",
            "latency": "low",
            "callback": self._callback,
        }
        if "wasapi" in selected.host_api.casefold():
            kwargs["extra_settings"] = sd.WasapiSettings(auto_convert=True)
        stream = None
        try:
            stream = sd.InputStream(**kwargs)
            stream.start()
        except Exception:
            self._closed.set()
            if stream is not None:
                stream.close(ignore_errors=True)
            self._stream = None
            raise
        self._stream = stream
        self._last_frame_at = time.monotonic()
        return selected



    def stop(self) -> None:
        if self._closed.is_set() and self._stream is None:
            return
        self._closed.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
            finally:
                stream.close(ignore_errors=True)
        try:
            self._frames.put_nowait(_STOP)
        except queue.Full:
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(_STOP)
            except queue.Empty:
                pass

    async def frames(self):
        while True:
            try:
                frame = await asyncio.to_thread(self._frames.get, True, 0.25)
            except queue.Empty:
                stream = self._stream
                if self._closed.is_set():
                    return
                if stream is not None and (
                    not stream.active or time.monotonic() - self._last_frame_at > 2.0
                ):
                    raise RuntimeError("Microphone disconnected or stopped")
                continue
            if frame is _STOP:
                return
            yield frame
