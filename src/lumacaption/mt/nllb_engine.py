from __future__ import annotations

from collections.abc import Callable, Sequence
import gc
from pathlib import Path

from gpu_runtime import configure_cuda_runtime
from languages import target_nllb_code


class NllbEngine:
    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        cache_dir: str | Path,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.requested_device = device
        self.cache_dir = Path(cache_dir)
        self.on_warning = on_warning or (lambda _message: None)
        self._translator = None
        self._sentencepiece = None
        self._model_path: Path | None = None
        self.active_device = ""

    def _resolve_model(self) -> Path:
        path = Path(self.model_id_or_path).expanduser()
        if path.exists():
            return path.resolve()
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        required = ("model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model")
        kwargs = dict(
            repo_id=self.model_id_or_path,
            revision="16bc5ff0482f9f1c0d35bdef950721ce58640789",
            cache_dir=self.cache_dir,
            allow_patterns=required,
        )
        try:
            cached = Path(snapshot_download(**kwargs, local_files_only=True))
            if all((cached / name).is_file() for name in required):
                return cached
        except LocalEntryNotFoundError:
            pass
        self.on_warning("Mengunduh NLLB; progres ada di terminal")
        cached = Path(snapshot_download(**kwargs))
        if not all((cached / name).is_file() for name in required):
            raise RuntimeError("Cache NLLB belum lengkap")
        return cached

    def _load(self, device: str) -> None:
        configure_cuda_runtime()
        import ctranslate2
        import sentencepiece as spm

        if self._model_path is None:
            self._model_path = self._resolve_model()
        compute_type = "int8_float16" if device == "cuda" else "int8"
        self._translator = ctranslate2.Translator(
            str(self._model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=4 if device == "cpu" else 1,
        )
        self._sentencepiece = spm.SentencePieceProcessor(
            model_file=str(self._model_path / "sentencepiece.bpe.model")
        )
        self.active_device = device
        self.on_warning(f"NLLB active: {device}")

    def _preferred_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() else "cpu"

    def _release(self) -> None:
        if self._translator is not None:
            try:
                self._translator.unload_model()
            except RuntimeError:
                pass
        self._translator = None
        gc.collect()

    def prepare(self, targets: Sequence[str]) -> None:
        """Exercise translation kernels, but never publish the warm-up text."""
        self.translate("Hello.", "eng_Latn", targets)

    def translate(self, text: str, source_nllb: str, targets: Sequence[str]) -> dict[str, str]:
        if not text.strip() or not targets:
            return {}
        if self._translator is None:
            preferred = self._preferred_device()
            try:
                self._load(preferred)
            except Exception as exc:
                if preferred == "cpu":
                    raise
                self.on_warning(f"NLLB CUDA unavailable: {exc}; using CPU")
                self._release()
                self._load("cpu")

        assert self._sentencepiece is not None
        pieces = self._sentencepiece.encode(text, out_type=str)
        # NLLB accepts 512 tokens; reserve source language and EOS tokens.
        if len(pieces) > 510:
            raise ValueError("Ucapan terlalu panjang untuk NLLB; beri jeda lebih sering (batas 510 token)")
        source = [source_nllb, *pieces, "</s>"]
        target_codes = [target_nllb_code(name) for name in targets]

        try:
            results = self._translator.translate_batch(
                [source.copy() for _ in target_codes],
                target_prefix=[[code] for code in target_codes],
                beam_size=1,
                max_decoding_length=256,
                return_end_token=True,
                batch_type="tokens",
                max_batch_size=1024,
            )
        except Exception as exc:
            if self.active_device != "cuda":
                raise
            self.on_warning(f"NLLB CUDA inference failed: {exc}; retrying on CPU")
            self._release()
            self._load("cpu")
            return self.translate(text, source_nllb, targets)

        translations: dict[str, str] = {}
        for name, code, result in zip(targets, target_codes, results, strict=True):
            tokens = list(result.hypotheses[0])
            if not tokens or tokens[-1] != "</s>":
                raise ValueError("Terjemahan mencapai batas decoder; beri jeda lebih sering, hasil tidak dipotong")
            tokens.pop()
            if tokens and tokens[0] == code:
                tokens.pop(0)
            translations[name] = self._sentencepiece.decode(tokens).strip()
        return translations

    def close(self) -> None:
        self._release()
