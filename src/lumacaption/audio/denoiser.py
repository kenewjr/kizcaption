"""DTLN Neural Network Noise Suppressor via ONNX Runtime."""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("lumacaption.audio.denoiser")

BLOCK_LEN = 512
BLOCK_SHIFT = 128


class DtlnDenoiser:
    """Real-time two-stage DTLN neural noise suppression at 16kHz."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.explicit_dir = model_dir is not None
        self.model_dir = Path(model_dir) if model_dir else Path("models/dtln")
        self.interpreter_1 = None
        self.interpreter_2 = None
        self.is_ready = False
        self._init_models()

    def _init_models(self) -> None:
        candidates = [
            (self.model_dir / "model_1.onnx", self.model_dir / "model_2.onnx"),
            (self.model_dir / "dtln_model_1.onnx", self.model_dir / "dtln_model_2.onnx"),
        ]
        if not self.explicit_dir:
            candidates.extend([
                (Path("models") / "dtln" / "model_1.onnx", Path("models") / "dtln" / "model_2.onnx"),
                (Path("models") / "model_1.onnx", Path("models") / "model_2.onnx"),
            ])
        m1_path, m2_path = None, None
        for p1, p2 in candidates:
            if p1.is_file() and p2.is_file():
                m1_path, m2_path = p1, p2
                break

        if not m1_path or not m2_path:
            return

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3
            self.interpreter_1 = ort.InferenceSession(str(m1_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self.interpreter_2 = ort.InferenceSession(str(m2_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self.in_names_1 = [inp.name for inp in self.interpreter_1.get_inputs()]
            self.in_names_2 = [inp.name for inp in self.interpreter_2.get_inputs()]
            self.is_ready = True
            self.reset()
        except Exception as exc:
            logger.warning("Gagal memuat model DTLN ONNX: %s", exc)
            self.is_ready = False

    def reset(self) -> None:
        if not self.is_ready or self.interpreter_1 is None or self.interpreter_2 is None:
            return
        self.in_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        self.out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        self.inputs_1 = {
            inp.name: np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)
            for inp in self.interpreter_1.get_inputs()
        }
        self.inputs_2 = {
            inp.name: np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)
            for inp in self.interpreter_2.get_inputs()
        }

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process 16kHz float32 audio signal through DTLN network."""
        if not self.is_ready or not audio.size:
            return audio

        samples = np.asarray(audio, dtype=np.float32)
        rem = len(samples) % BLOCK_SHIFT
        if rem != 0:
            pad_len = BLOCK_SHIFT - rem
            padded = np.pad(samples, (0, pad_len), mode="constant")
        else:
            padded = samples

        num_blocks = len(padded) // BLOCK_SHIFT
        output = np.zeros(len(padded), dtype=np.float32)

        for idx in range(num_blocks):
            self.in_buffer[:-BLOCK_SHIFT] = self.in_buffer[BLOCK_SHIFT:]
            self.in_buffer[-BLOCK_SHIFT:] = padded[idx * BLOCK_SHIFT : (idx + 1) * BLOCK_SHIFT]

            in_fft = np.fft.rfft(self.in_buffer)
            in_mag = np.abs(in_fft)
            in_phase = np.angle(in_fft)
            in_mag_tensor = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

            self.inputs_1[self.in_names_1[0]] = in_mag_tensor
            out_1 = self.interpreter_1.run(None, self.inputs_1)
            out_mask = out_1[0]
            self.inputs_1[self.in_names_1[1]] = out_1[1]

            est_complex = in_mag * out_mask * np.exp(1j * in_phase)
            est_block = np.fft.irfft(est_complex)
            est_tensor = np.reshape(est_block, (1, 1, -1)).astype(np.float32)

            self.inputs_2[self.in_names_2[0]] = est_tensor
            out_2 = self.interpreter_2.run(None, self.inputs_2)
            out_block = out_2[0]
            self.inputs_2[self.in_names_2[1]] = out_2[1]

            self.out_buffer[:-BLOCK_SHIFT] = self.out_buffer[BLOCK_SHIFT:]
            self.out_buffer[-BLOCK_SHIFT:] = 0.0
            self.out_buffer += np.squeeze(out_block)
            output[idx * BLOCK_SHIFT : (idx + 1) * BLOCK_SHIFT] = self.out_buffer[:BLOCK_SHIFT]

        return output[: len(samples)]
