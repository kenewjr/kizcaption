from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
import queue
import re
import threading
import time
from typing import Any

import numpy as np

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


def float_audio_to_mono_pcm16(audio: Any) -> tuple[np.ndarray, float, float]:
    """Downmix native float audio and return PCM16 plus unclipped meter values."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 0:
        samples = samples.reshape(1)
    elif samples.ndim > 1:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    mono = samples.reshape(-1)
    if mono.size:
        rms = float(np.sqrt(np.mean(mono * mono)))
        peak = float(np.max(np.abs(mono)))
    else:
        rms = peak = 0.0
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
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
    ) -> None:
        self.device = device
        self.on_warning = on_warning or (lambda _message: None)
        self.on_level = on_level or (lambda _rms, _peak: None)
        self._frames: queue.Queue[Any] = queue.Queue(maxsize=queue_frames)
        self._stream: Any | None = None
        self._closed = threading.Event()
        self._last_frame_at = 0.0
        self._last_level_at = 0.0
        self._overflow_reported = False
        self._converter: AudioFrameConverter | None = None
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
        if status:
            self.on_warning(f"Microphone status: {status}")
        samples, rms, peak = float_audio_to_mono_pcm16(indata)
        self._last_frame_at = time.monotonic()
        now = self._last_frame_at
        if now - self._last_level_at >= 0.08 and samples.size:
            self.on_level(rms, peak)
            self._last_level_at = now
        assert self._converter is not None
        for frame in self._converter.process(samples):
            self._enqueue(frame)

    def start(self) -> InputDevice:
        import sounddevice as sd

        if self._stream is not None:
            assert self.active_device is not None
            return self.active_device
        selected = self.resolve_device(self.device)
        self.active_device = selected
        self._converter = AudioFrameConverter(selected.sample_rate)
        self._closed.clear()
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
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
