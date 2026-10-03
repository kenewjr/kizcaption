from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512


class UtteranceTooLongError(ValueError):
    pass


class ProbabilityModel(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroOnnx:
    """Minimal Silero v5 ONNX runner; avoids Torch/TorchAudio dependencies."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"Silero requires {FRAME_SAMPLES} samples, got {frame.shape}")
        audio = frame.astype(np.float32, copy=False).reshape(1, -1)
        audio = np.concatenate((self._context, audio), axis=1)
        output, self._state = self._session.run(
            None,
            {"input": audio, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)},
        )
        self._context = audio[:, -64:]
        return float(output.reshape(-1)[0])


class VadSegmenter:
    def __init__(
        self,
        model: ProbabilityModel,
        threshold: float = 0.5,
        min_silence_ms: int = 650,
        min_speech_ms: int = 250,
        max_utterance_seconds: int = 20,
        speech_pad_ms: int = 320,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.min_silence_frames = max(1, math.ceil(min_silence_ms / 32))
        self.min_speech_frames = max(1, math.ceil(min_speech_ms / 32))
        self.max_frames = max(1, round(max_utterance_seconds * SAMPLE_RATE / FRAME_SAMPLES))
        self.pre_roll = deque(maxlen=max(1, round(speech_pad_ms / 32)))
        self.post_roll_frames = min(8, self.min_silence_frames)
        self.reset()

    def reset(self) -> None:
        self.model.reset()
        self.pre_roll.clear()
        self._speech: list[np.ndarray] = []
        self._speaking = False
        self._silence_frames = 0
        self._voiced_frames = 0
        self._last_probability = 0.0
        self._ambient_prob = 0.0
        self._leading_frames = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def last_probability(self) -> float:
        return self._last_probability

    def process(self, pcm: np.ndarray) -> np.ndarray | None:
        if pcm.dtype != np.int16 or pcm.shape != (FRAME_SAMPLES,):
            raise ValueError("VAD input must be one int16 frame of 512 samples")
        probability = self.model(pcm.astype(np.float32) / 32768.0)
        self._last_probability = min(1.0, max(0.0, float(probability)))

        if not self._speaking:
            self._ambient_prob = 0.95 * self._ambient_prob + 0.05 * self._last_probability
            onset_threshold = min(0.70, max(self.threshold, self._ambient_prob + 0.12))
            if probability < onset_threshold:
                self.pre_roll.append(pcm.copy())
                return None
            self._speaking = True
            self._leading_frames = len(self.pre_roll)
            self._speech = [*self.pre_roll, pcm.copy()]
            self.pre_roll.clear()
            self._voiced_frames = 1
            self._silence_frames = 0
            return None

        self._speech.append(pcm.copy())
        # Hysteresis: keep speaking on softer consonant/vowel word endings
        exit_threshold = max(0.30, self.threshold - 0.08)
        if probability >= exit_threshold:
            self._voiced_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        hit_pause = self._silence_frames >= self.min_silence_frames
        hit_limit = len(self._speech) - self._leading_frames - self._silence_frames >= self.max_frames
        if hit_limit and not hit_pause:
            # ponytail: bounded final-only buffer; add internal chunking for longer monologues.
            seconds = self.max_frames * FRAME_SAMPLES / SAMPLE_RATE
            self.reset()
            raise UtteranceTooLongError(
                f"Ucapan melewati batas {seconds:.0f} detik tanpa jeda; "
                "hasil tidak dipotong. Beri jeda atau naikkan Batas ucapan."
            )
        if not hit_pause:
            return None

        trailing = max(0, self._silence_frames - self.post_roll_frames)
        kept = self._speech[:-trailing] if trailing else self._speech
        utterance = np.concatenate(kept) if self._voiced_frames >= self.min_speech_frames else None
        self._speech = []
        self._speaking = False
        self._silence_frames = 0
        self._voiced_frames = 0
        self.pre_roll.clear()
        return utterance

    def flush(self) -> np.ndarray | None:
        if not self._speech or self._voiced_frames < self.min_speech_frames:
            self.reset()
            return None
        utterance = np.concatenate(self._speech)
        self.reset()
        return utterance
