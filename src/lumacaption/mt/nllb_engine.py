from __future__ import annotations

from collections.abc import Callable, Sequence
import gc
from pathlib import Path
import re

from lumacaption.gpu_runtime import configure_cuda_runtime
from lumacaption.languages import target_nllb_code


from lumacaption.mt.conversational_data import (
    CONVERSATIONAL_ALIASES,
    CONVERSATIONAL_MAP,
    INDONESIAN_SLANG_MAP,
    normalize_indonesian_slang,
)


class NllbEngine:
    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        cache_dir: str | Path,
        on_warning: Callable[[str], None] | None = None,
        *,
        compute_type: str = "auto",
        beam_size: int = 1,
        cpu_threads: int = 4,
        normalize_slang: bool = True,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.normalize_slang = normalize_slang
        self.cache_dir = Path(cache_dir)
        self.on_warning = on_warning or (lambda _message: None)
        self._translator = None
        self._sentencepiece = None
        self._model_path: Path | None = None
        self.active_device = ""
        self.active_compute_type = ""
        self._consecutive_load_failures = 0
        self._cache: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}
        self._item_cache: dict[tuple[str, str, str], str] = {}

    def _resolve_model(self) -> Path:
        path = Path(self.model_id_or_path).expanduser()
        if path.exists():
            return path.resolve()

        required = ("model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model")

        # 2. Check local cache via model_manager inspect_model & candidate discovery
        try:
            from lumacaption.model_manager import inspect_model, find_model_locally, adopt_model
            state, local_path = inspect_model(self.model_id_or_path, self.cache_dir)
            if state == "Tersedia lokal" and local_path is not None and all((local_path / name).is_file() for name in required):
                return local_path.resolve()
            alt_state, alt_path, _ = find_model_locally(self.model_id_or_path)
            if alt_state == "Tersedia lokal" and alt_path is not None:
                adopt_model(self.model_id_or_path, alt_path, self.cache_dir)
                state, local_path = inspect_model(self.model_id_or_path, self.cache_dir)
                if state == "Tersedia lokal" and local_path is not None and all((local_path / name).is_file() for name in required):
                    return local_path.resolve()
        except Exception:
            pass

        repo_id = self.model_id_or_path
        if repo_id == "nllb":
            repo_id = "mijuanlo/nllb-200-distilled-600M-ct2-int8"
        elif repo_id == "nllb-1.3b":
            repo_id = "mijuanlo/nllb-200-distilled-1.3B-int8-ct2"

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        # Resolve revision per repo catalog
        revision = "16bc5ff0482f9f1c0d35bdef950721ce58640789"
        if "1.3b" in repo_id.lower() or "1.3B" in repo_id:
            revision = "6eee5eda03ff1441d2a6117d34a02e44504ce321"

        kwargs = dict(
            repo_id=repo_id,
            revision=revision,
            cache_dir=self.cache_dir,
            allow_patterns=required,
        )
        try:
            cached = Path(snapshot_download(**kwargs, local_files_only=True))
            if all((cached / name).is_file() for name in required):
                return cached
        except LocalEntryNotFoundError:
            pass
        self.on_warning("Mengunduh model translasi NLLB-200 (~604 MB)... Harap tunggu.")
        cached = Path(snapshot_download(**kwargs))
        if not all((cached / name).is_file() for name in required):
            raise RuntimeError("Cache NLLB belum lengkap")
        return cached

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

    def _load(self, device: str) -> None:
        configure_cuda_runtime()
        import ctranslate2
        import sentencepiece as spm

        if self._model_path is None:
            self._model_path = self._resolve_model()
        compute_type = self._resolve_compute_type(device)
        self.on_warning(f"Memuat model translasi NLLB-200 ke {device.upper()} ({compute_type})...")
        self._translator = ctranslate2.Translator(
            str(self._model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=max(1, min(self.cpu_threads, 16)) if device == "cpu" else 1,
        )
        self._sentencepiece = spm.SentencePieceProcessor(
            model_file=str(self._model_path / "sentencepiece.bpe.model")
        )
        self.active_device = device
        self.active_compute_type = compute_type
        self.on_warning(f"NLLB aktif: {device.upper()} ({compute_type})")

    def _preferred_device(self) -> str:
        from lumacaption.gpu_runtime import is_cuda_available

        cuda_ok = is_cuda_available()
        if self.requested_device == "cuda":
            if not cuda_ok:
                self.on_warning("CUDA diminta namun library CUDA (cublas64_12.dll) tidak tersedia; NLLB dialihkan ke CPU")
                return "cpu"
            return "cuda"
        if self.requested_device == "auto":
            return "cuda" if cuda_ok else "cpu"
        return self.requested_device

    def _release(self) -> None:
        if self._translator is not None:
            try:
                self._translator.unload_model()
            except RuntimeError:
                pass
        self._translator = None
        self._sentencepiece = None
        self._cache.clear()
        self._item_cache.clear()
        gc.collect()

    def _lookup_cache(self, text: str, source_nllb: str, targets: Sequence[str]) -> dict[str, str] | None:
        clean = text.strip()
        if len(clean) > 160:
            return None
        t_targets = tuple(targets)
        # 1. Exact match
        k_exact = (clean, source_nllb, t_targets)
        if k_exact in self._cache:
            return dict(self._cache[k_exact])
        # 2. Punctuation-stripped match
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_stripped = (stripped, source_nllb, t_targets)
        if k_stripped in self._cache:
            return dict(self._cache[k_stripped])
        # 3. Case-insensitive fast keys
        k_lower = (stripped.lower(), source_nllb, t_targets)
        if k_lower in self._cache:
            return dict(self._cache[k_lower])
        k_title = (stripped.capitalize(), source_nllb, t_targets)
        if k_title in self._cache:
            return dict(self._cache[k_title])

        # 4. Fallback scan over bounded LRU cache (<= 128 entries)
        low = stripped.lower()
        for (ck, csrc, ctgts), val in self._cache.items():
            if csrc == source_nllb and ctgts == t_targets:
                if re.sub(r"[.!?…,:;\"'“”]+$", "", ck).strip().lower() == low:
                    return dict(val)
        return None

    def _put_cache(self, text: str, source_nllb: str, targets: Sequence[str], res: dict[str, str]) -> None:
        clean = text.strip()
        if len(clean) > 160:
            return
        t_targets = tuple(targets)
        k_exact = (clean, source_nllb, t_targets)
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_lower = (stripped.lower(), source_nllb, t_targets)

        for k in (k_exact, k_lower):
            if len(self._cache) >= 128 and k not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[k] = dict(res)

    def _lookup_target_cache(self, text: str, source_nllb: str, target_name: str) -> str | None:
        clean = text.strip()
        if len(clean) > 160:
            return None
        k_exact = (clean, source_nllb, target_name)
        if k_exact in self._item_cache:
            return self._item_cache[k_exact]
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_stripped = (stripped, source_nllb, target_name)
        if k_stripped in self._item_cache:
            return self._item_cache[k_stripped]
        k_lower = (stripped.lower(), source_nllb, target_name)
        if k_lower in self._item_cache:
            return self._item_cache[k_lower]
        return None

    def _put_target_cache(self, text: str, source_nllb: str, target_name: str, translation: str) -> None:
        clean = text.strip()
        if len(clean) > 160 or not translation:
            return
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_exact = (clean, source_nllb, target_name)
        k_lower = (stripped.lower(), source_nllb, target_name)
        for k in (k_exact, k_lower):
            if len(self._item_cache) >= 512 and k not in self._item_cache:
                oldest_key = next(iter(self._item_cache))
                del self._item_cache[oldest_key]
            self._item_cache[k] = translation

    def prepare(self, targets: Sequence[str]) -> None:
        """Exercise translation kernels, but never publish the warm-up text."""
        if not targets:
            return
        first_target = targets[0]
        try:
            target_code = target_nllb_code(first_target)
        except KeyError:
            target_code = "eng_Latn"
        source_code = "ind_Latn" if target_code != "ind_Latn" else "eng_Latn"
        self.on_warning(f"Uji coba translasi NLLB-200 [{first_target}]...")
        self._in_prepare = True
        try:
            self.translate("Tes inferensi.", source_code, [first_target])
        except Exception as exc:
            self.on_warning(f"NLLB warmup warning: {exc}")
            if self.active_device != "cpu":
                self.on_warning("Beralih model translasi NLLB-200 ke CPU...")
                self._release()
                self._load("cpu")
        finally:
            self._in_prepare = False

    def translate(self, text: str, source_nllb: str, targets: Sequence[str]) -> dict[str, str]:
        clean_original = text.strip()
        if not clean_original or not targets:
            return {}

        cached = self._lookup_cache(clean_original, source_nllb, targets)
        if cached is not None:
            return cached

        translations: dict[str, str] = {}
        needed_targets: list[str] = []
        for name in targets:
            try:
                code = target_nllb_code(name)
            except KeyError:
                needed_targets.append(name)
                continue
            if code == source_nllb:
                translations[name] = clean_original
            else:
                cached_item = self._lookup_target_cache(clean_original, source_nllb, name)
                if cached_item is not None:
                    translations[name] = cached_item
                else:
                    needed_targets.append(name)

        if not needed_targets:
            res = {name: translations[name] for name in targets if name in translations}
            self._put_cache(clean_original, source_nllb, targets, res)
            return res

        # Conversational stream formula fast-path for Indonesian
        if source_nllb == "ind_Latn":
            norm = re.sub(r"[^\w\s-]", "", clean_original.lower())
            norm = re.sub(r"\s+", " ", norm).strip()
            canonical = CONVERSATIONAL_ALIASES.get(norm, norm)
            if canonical in CONVERSATIONAL_MAP:
                formula = CONVERSATIONAL_MAP[canonical]
                if all(t in formula for t in needed_targets):
                    for t in needed_targets:
                        translations[t] = formula[t]
                    res = {name: translations[name] for name in targets if name in translations}
                    self._put_cache(clean_original, source_nllb, targets, res)
                    for t_name, t_val in res.items():
                        self._put_target_cache(clean_original, source_nllb, t_name, t_val)
                    return res

        if self._translator is None:
            preferred = self._preferred_device()
            try:
                self._load(preferred)
                self._consecutive_load_failures = 0
            except Exception as exc:
                self._consecutive_load_failures += 1
                self._release()
                err_str = str(exc).lower()
                is_device_fault = any(kw in err_str for kw in ("cuda", "cudnn", "out of memory", "cublas", "device"))
                if preferred == "cuda" and is_device_fault and self._consecutive_load_failures <= 3:
                    self.on_warning(f"NLLB CUDA unavailable: {exc}; using CPU")
                    try:
                        self._load("cpu")
                        self._consecutive_load_failures = 0
                        return self.translate(text, source_nllb, targets)
                    except Exception as cpu_exc:
                        self._release()
                        raise RuntimeError(f"All NLLB fallbacks failed: {cpu_exc}") from cpu_exc
                raise RuntimeError(f"All NLLB fallbacks failed: {exc}") from exc

        assert self._sentencepiece is not None
        # Normalize casing and slang before translation
        clean_text = text.strip()
        if source_nllb == "ind_Latn" and getattr(self, "normalize_slang", True):
            clean_text = normalize_indonesian_slang(clean_text)

        # Split compound utterances by sentence boundary to guarantee full clause translation
        from lumacaption.mt.slang_normalizer import split_into_sentences
        sentences = split_into_sentences(clean_text)
        if not sentences:
            return {}

        target_codes = [target_nllb_code(name) for name in needed_targets]
        batch_sources: list[list[str]] = []
        batch_prefixes: list[list[str]] = []
        sentence_data: list[list[str]] = []

        max_sentence_tokens = 32
        for s in sentences:
            s_clean = s.strip()
            if s_clean and s_clean[0].islower():
                s_clean = s_clean[0].upper() + s_clean[1:]
            if s_clean and s_clean[-1] not in ".?!…:;":
                s_clean += "."
            pieces = self._sentencepiece.encode(s_clean, out_type=str)
            if len(pieces) > 510:
                self.on_warning("Ucapan melebihi batas 510 token NLLB; input dipotong")
                pieces = pieces[:510]
            max_sentence_tokens = max(max_sentence_tokens, int(len(pieces) * 1.8) + 12)
            src_tokens = [source_nllb, *pieces, "</s>"]
            sentence_data.append(src_tokens)
            for code in target_codes:
                batch_sources.append(src_tokens.copy())
                batch_prefixes.append([code])

        max_tokens = min(256, max_sentence_tokens)
        effective_beam = max(1, self.beam_size)
        try:
            results = self._translator.translate_batch(
                batch_sources,
                target_prefix=batch_prefixes,
                beam_size=effective_beam,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                disable_unk=True,
                max_decoding_length=max_tokens,
                return_end_token=True,
                batch_type="tokens",
                max_batch_size=1024,
            )
        except Exception as exc:
            if getattr(self, "_in_prepare", False):
                raise
            err_str = str(exc).lower()
            is_device_fault = any(kw in err_str for kw in ("cuda", "cudnn", "out of memory", "cublas", "device"))
            if self.active_device == "cuda" and is_device_fault:
                self.on_warning(f"NLLB CUDA inference failed: {exc}; retrying on CPU")
                self._release()
                try:
                    self._load("cpu")
                except Exception as cpu_exc:
                    self._release()
                    raise RuntimeError(f"All NLLB fallbacks failed: {cpu_exc}") from cpu_exc
                return self.translate(text, source_nllb, targets)
            if self.active_device == "cpu" and not is_device_fault:
                self.on_warning(f"NLLB CPU inference error on utterance: {exc}")
                return {}
            self._release()
            raise RuntimeError(f"All NLLB fallbacks failed: {exc}") from exc

        # Collect decoded segments per target
        target_segments: dict[str, list[str]] = {name: [] for name in needed_targets}
        res_idx = 0
        hit_limits: dict[str, bool] = {name: False for name in needed_targets}

        for _s_tokens in sentence_data:
            for name, code in zip(needed_targets, target_codes, strict=True):
                result = results[res_idx]
                res_idx += 1
                tokens = list(result.hypotheses[0])
                if not tokens or tokens[-1] != "</s>":
                    hit_limits[name] = True
                else:
                    tokens.pop()
                if tokens and tokens[0] == code:
                    tokens.pop(0)
                decoded_part = self._sentencepiece.decode(tokens).strip()
                if decoded_part:
                    target_segments[name].append(decoded_part)

        for name, code in zip(needed_targets, target_codes, strict=True):
            join_delimiter = "" if code in ("jpn_Jpan", "zho_Hans", "zho_Hant") else " "
            decoded = join_delimiter.join(target_segments[name]).strip()

            # Clean leading dialog turn dashes if original text didn't start with a dash
            if decoded.startswith(("- ", "— ", "– ", "-")) and not text.lstrip().startswith(("-", "—", "–")):
                decoded = decoded.lstrip("-—– ").strip()

            # Clean unknown token artifacts and collapse whitespace
            decoded = decoded.replace("⁇", "").replace("\ufffd", "").strip()
            decoded = re.sub(r"\s+", " ", decoded)

            # CJK / Korean punctuation spacing cleanup
            if code in ("jpn_Jpan", "zho_Hans", "zho_Hant"):
                decoded = re.sub(r"\s*([。、！？，：])\s*", r"\1", decoded)
            elif code == "kor_Hang":
                decoded = re.sub(r"\s*([.!?])", r"\1", decoded)

            if hit_limits[name]:
                self.on_warning("Terjemahan mencapai batas decoder; teks diakhiri elipsis")
                if decoded:
                    decoded += "…"
            translations[name] = decoded
            self._put_target_cache(clean_original, source_nllb, name, decoded)

        # Cache short phrases (LRU 128 max entries)
        res = {name: translations[name] for name in targets if name in translations}
        self._put_cache(clean_original, source_nllb, targets, res)
        return res

    def close(self) -> None:
        self._release()
