from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.config import AppConfig, COMPUTE_TYPES, MODEL_SIZES, NLLB_MODELS
from lumacaption.stt.whisper_engine import WhisperEngine
from lumacaption.mt.nllb_engine import NllbEngine
from lumacaption.pipeline import CaptionPipeline


class EngineComputeAndCacheTests(unittest.TestCase):
    def test_config_compute_types_and_models(self):
        cfg = AppConfig(
            whisper_model="distil-large-v3",
            nllb_model="mijuanlo/nllb-200-distilled-1.3B-int8-ct2",
            stt_compute_type="int8_float16",
            mt_compute_type="float16",
        )
        self.assertIn(cfg.whisper_model, MODEL_SIZES)
        self.assertIn(cfg.nllb_model, NLLB_MODELS)
        self.assertIn(cfg.stt_compute_type, COMPUTE_TYPES)
        self.assertIn(cfg.mt_compute_type, COMPUTE_TYPES)

    def test_whisper_compute_type_resolution(self):
        engine = WhisperEngine(
            model_size="small",
            device="cuda",
            model_cache=Path("models/whisper"),
            compute_type="auto",
        )
        # Mock ctranslate2 supported compute types for CC >= 7.0 (Turing/Ampere/Ada)
        with patch("ctranslate2.get_supported_compute_types", return_value={"int8_float16", "float16", "float32"}):
            resolved = engine._resolve_compute_type("cuda")
            self.assertEqual(resolved, "int8_float16")

        # Mock ctranslate2 supported compute types for Pascal CC 6.1 (no int8_float16)
        with patch("ctranslate2.get_supported_compute_types", return_value={"int8_float32", "float32"}):
            resolved = engine._resolve_compute_type("cuda")
            self.assertEqual(resolved, "int8_float32")

        # Fallback to float16 if supported
        with patch("ctranslate2.get_supported_compute_types", return_value={"float16", "float32"}):
            resolved = engine._resolve_compute_type("cuda")
            self.assertEqual(resolved, "float16")

        # CPU fallback
        resolved_cpu = engine._resolve_compute_type("cpu")
        self.assertEqual(resolved_cpu, "int8")

    def test_nllb_compute_type_resolution(self):
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            compute_type="auto",
        )
        with patch("ctranslate2.get_supported_compute_types", return_value={"int8_float16", "float16"}):
            resolved = engine._resolve_compute_type("cuda")
            self.assertEqual(resolved, "int8_float16")

        # Explicit compute type override preserved if supported
        engine_explicit = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            compute_type="float16",
        )
        with patch("ctranslate2.get_supported_compute_types", return_value={"float16", "int8_float16"}):
            resolved_explicit = engine_explicit._resolve_compute_type("cuda")
            self.assertEqual(resolved_explicit, "float16")

    def test_nllb_lru_cache_hit_and_eviction(self):
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cpu",
            cache_dir=Path("models/nllb-cache"),
        )
        engine._translator = MagicMock()
        # Seed cache
        cache_key = ("Halo dunia", "ind_Latn", ("English",))
        engine._cache[cache_key] = {"English": "Hello world"}
        
        # Calling with cached item should hit cache without calling translator
        result = engine.translate("Halo dunia", "ind_Latn", ["English"])
        self.assertEqual(result, {"English": "Hello world"})
        # Normalized cache hit: punctuation and case variations hit cache without translator
        res_punct = engine.translate("Halo dunia!", "ind_Latn", ["English"])
        self.assertEqual(res_punct, {"English": "Hello world"})
        res_case = engine.translate("halo dunia...", "ind_Latn", ["English"])
        self.assertEqual(res_case, {"English": "Hello world"})
        engine._translator.translate_batch.assert_not_called()

        # Cache eviction when exceeding max size (128)
        for i in range(135):
            k = (f"text_{i}", "ind_Latn", ("English",))
            if len(engine._cache) >= 128:
                oldest = next(iter(engine._cache))
                del engine._cache[oldest]
            engine._cache[k] = {"English": f"trans_{i}"}

        self.assertLessEqual(len(engine._cache), 128)
        self.assertNotIn(cache_key, engine._cache)
        self.assertIn(("text_134", "ind_Latn", ("English",)), engine._cache)

        # Test item-level cache across different target combinations
        engine._item_cache[("Selamat pagi", "ind_Latn", "English")] = "Good morning"
        mock_res = MagicMock()
        mock_res.hypotheses = [["jpn_Jpan", "ohayou", "</s>"]]
        engine._translator.translate_batch.return_value = [mock_res]
        engine._sentencepiece = MagicMock()
        engine._sentencepiece.encode.return_value = ["selamat", "pagi"]
        engine._sentencepiece.decode.return_value = "おはよう"

        res_multi = engine.translate("Selamat pagi", "ind_Latn", ["English", "Japanese"])
        self.assertEqual(res_multi["English"], "Good morning")
        self.assertEqual(res_multi["Japanese"], "おはよう")
        # Translator only translated Japanese (1 sequence instead of 2)
        call_args = engine._translator.translate_batch.call_args
        self.assertEqual(len(call_args[0][0]), 1)

    def test_pipeline_instantiation_with_v130_options(self):
        cfg = AppConfig(
            whisper_model="distil-large-v3",
            nllb_model="mijuanlo/nllb-200-distilled-1.3B-int8-ct2",
            stt_compute_type="int8_float16",
            mt_compute_type="int8_float16",
        )
        events = []
        pipeline = CaptionPipeline(
            cfg,
            _root,
            on_event=events.append,
        )
        self.assertIsNotNone(pipeline)

    def test_nllb_graceful_decoder_limit(self):
        warnings = []
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cpu",
            cache_dir=Path("models/nllb-cache"),
            on_warning=warnings.append,
        )
        engine._translator = MagicMock()
        engine._sentencepiece = MagicMock()
        engine._sentencepiece.encode.return_value = ["halo", "dunia"]
        engine._sentencepiece.decode.return_value = "hello world"

        # Mock result where hypothesis does not end with </s> (hit max decoding length)
        mock_res = MagicMock()
        mock_res.hypotheses = [["eng_Latn", "hello", "world"]]
        engine._translator.translate_batch.return_value = [mock_res]

        res = engine.translate("Halo dunia", "ind_Latn", ["English"])
        self.assertIn("English", res)
        self.assertTrue(res["English"].endswith("…"))
        self.assertTrue(any("batas decoder" in w for w in warnings))

    def test_nllb_same_language_passthrough(self):
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cpu",
            cache_dir=Path("models/nllb-cache"),
        )
        engine._translator = MagicMock()
        # Direct passthrough when target matches source language
        res = engine.translate("Hello.", "eng_Latn", ["English"])
        self.assertEqual(res, {"English": "Hello."})
        engine._translator.translate_batch.assert_not_called()

        # Mixed targets: only non-matching target is sent to translate_batch
        engine._sentencepiece = MagicMock()
        engine._sentencepiece.encode.return_value = ["konnichiwa"]
        engine._sentencepiece.decode.return_value = "こんにちは"
        mock_res = MagicMock()
        mock_res.hypotheses = [["jpn_Jpan", "konnichiwa", "</s>"]]
        engine._translator.translate_batch.return_value = [mock_res]

        res_mixed = engine.translate("Hello.", "eng_Latn", ["English", "Japanese"])
        self.assertEqual(res_mixed["English"], "Hello.")
        self.assertEqual(res_mixed["Japanese"], "こんにちは")
        # Ensure only 1 batch item was passed (Japanese)
        self.assertEqual(engine._translator.translate_batch.call_count, 1)
        call_kwargs = engine._translator.translate_batch.call_args[1]
        self.assertEqual(call_kwargs["target_prefix"], [["jpn_Jpan"]])

    def test_nllb_cuda_inference_failure_falls_back_to_cpu(self):
        warnings = []
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            on_warning=warnings.append,
        )
        engine.active_device = "cuda"
        mock_cuda_translator = MagicMock()
        mock_cuda_translator.translate_batch.side_effect = RuntimeError("CUDA out of memory")
        engine._translator = mock_cuda_translator
        mock_sp = MagicMock()
        mock_sp.encode.return_value = ["test"]
        mock_sp.decode.return_value = "sukses"
        engine._sentencepiece = mock_sp

        mock_cpu_translator = MagicMock()
        mock_res = MagicMock()
        mock_res.hypotheses = [["eng_Latn", "test", "</s>"]]
        mock_cpu_translator.translate_batch.return_value = [mock_res]

        with patch.object(engine, "_load") as mock_load:
            def fake_load(device):
                engine.active_device = device
                engine._translator = mock_cpu_translator
                engine._sentencepiece = mock_sp
            mock_load.side_effect = fake_load
            res = engine.translate("Halo", "ind_Latn", ["English"])

        self.assertEqual(res, {"English": "sukses"})
        self.assertEqual(engine.active_device, "cpu")
        self.assertTrue(any("CUDA inference failed" in w for w in warnings))

    def test_indonesian_slang_normalization(self):
        from lumacaption.mt.nllb_engine import normalize_indonesian_slang

        raw = "Gue lagi mau makan nih, tapi mager bgt dan gak ada uang"
        normalized = normalize_indonesian_slang(raw)
        self.assertIn("saya", normalized.lower())
        self.assertIn("sedang", normalized.lower())
        self.assertIn("malas bergerak", normalized.lower())
        self.assertIn("sangat", normalized.lower())
        self.assertIn("tidak", normalized.lower())

        streamer_raw = "Yaudah gapapa kita mabar, dapet loot hoki kocak"
        norm2 = normalize_indonesian_slang(streamer_raw)
        self.assertIn("baiklah", norm2.lower())
        self.assertIn("tidak apa-apa", norm2.lower())
        self.assertIn("main bersama", norm2.lower())
        self.assertIn("dapat", norm2.lower())
        self.assertIn("beruntung", norm2.lower())
        self.assertIn("lucu", norm2.lower())

    def test_custom_whisper_model_resolution(self):
        from lumacaption.stt.whisper_engine import resolve_whisper_repo
        self.assertEqual(resolve_whisper_repo("whisper-small-id"), "ammaraldirawi/faster-whisper-small-id-int8")
        self.assertEqual(resolve_whisper_repo("whisper-medium-id"), "cahya/faster-whisper-medium-id")
        self.assertEqual(resolve_whisper_repo("small"), "small")

    def test_cpu_thread_budget_coordination(self):
        # When both STT and MT run on CPU with targets, thread budget is coordinated
        total_threads = 8
        both_cpu = True
        stt_threads = max(1, round(total_threads * 0.60))
        mt_threads = max(1, total_threads - stt_threads)
        self.assertEqual(stt_threads, 5)
        self.assertEqual(mt_threads, 3)
        self.assertEqual(stt_threads + mt_threads, total_threads)

    def test_whisper_warmup_failure_falls_back_to_cpu(self):
        from lumacaption.stt.whisper_engine import WhisperEngine

        warnings = []
        engine = WhisperEngine(
            model_size="base",
            device="cuda",
            model_cache=Path("models/whisper"),
            on_warning=warnings.append,
        )
        engine._candidates = [("base", "cuda"), ("base", "cpu")]
        engine._candidate_index = 0
        engine._model_path = "/fake/model/path"

        mock_cuda_model = MagicMock()
        mock_cuda_model.transcribe.side_effect = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

        mock_cpu_model = MagicMock()
        mock_cpu_model.transcribe.return_value = ([], MagicMock(language="id"))

        models_loaded = []
        def fake_load_next():
            model_size, device = engine._candidates[engine._candidate_index]
            engine._candidate_index += 1
            engine.active_model = model_size
            engine.active_device = device
            if device == "cuda":
                engine._model = mock_cuda_model
            else:
                engine._model = mock_cpu_model
            models_loaded.append(device)

        with patch.object(engine, "_load_next", side_effect=fake_load_next):
            engine.prepare("id")

        self.assertEqual(models_loaded, ["cuda", "cpu"])
        self.assertEqual(engine.active_device, "cpu")
        self.assertTrue(any("warmup gagal" in w for w in warnings))

    def test_nllb_prepare_exercises_translation_and_falls_back(self):
        warnings = []
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            on_warning=warnings.append,
        )
        engine.active_device = "cuda"
        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = RuntimeError("Library cublas64_12.dll is not found")
        engine._translator = mock_translator
        mock_sp = MagicMock()
        mock_sp.encode.return_value = ["tes"]
        mock_sp.decode.return_value = "test"
        engine._sentencepiece = mock_sp

        with patch.object(engine, "_load") as mock_load:
            def fake_load(device):
                engine.active_device = device
                engine._translator = MagicMock()
                engine._sentencepiece = mock_sp
            mock_load.side_effect = fake_load
            engine.prepare(["English"])

        self.assertEqual(engine.active_device, "cpu")
        self.assertTrue(any("warmup" in w.lower() for w in warnings))

    def test_whisper_avoids_cuda_candidate_when_dll_missing(self):
        from lumacaption.stt.whisper_engine import WhisperEngine

        engine = WhisperEngine(
            model_size="base",
            device="auto",
            model_cache=Path("models/whisper"),
        )
        with patch("lumacaption.gpu_runtime.is_cuda_available", return_value=False):
            candidates = engine._build_candidates()
        self.assertEqual(candidates, [("base", "cpu")])

    def test_nllb_preferred_device_selects_cpu_when_dll_missing(self):
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="auto",
            cache_dir=Path("models/nllb-cache"),
        )
        with patch("lumacaption.gpu_runtime.is_cuda_available", return_value=False):
            device = engine._preferred_device()
        self.assertEqual(device, "cpu")

    def test_nllb_load_failure_non_device_raises_all_fallbacks_failed(self):
        warnings = []
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            on_warning=warnings.append,
        )
        with patch.object(engine, "_preferred_device", return_value="cuda"), \
             patch.object(engine, "_load", side_effect=RuntimeError("Corrupt model archive")):
            with self.assertRaises(RuntimeError) as ctx:
                engine.translate("Halo", "ind_Latn", ["English"])
            self.assertIn("All NLLB fallbacks failed", str(ctx.exception))
            self.assertIn("Corrupt model archive", str(ctx.exception))
        # Ensure non-device error is not mislabeled as CUDA unavailable
        self.assertFalse(any("CUDA unavailable" in w for w in warnings))

    def test_nllb_load_failure_cuda_device_falls_back_to_cpu(self):
        warnings = []
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            on_warning=warnings.append,
        )
        mock_cpu_translator = MagicMock()
        mock_res = MagicMock()
        mock_res.hypotheses = [["eng_Latn", "hello", "</s>"]]
        mock_cpu_translator.translate_batch.return_value = [mock_res]
        mock_sp = MagicMock()
        mock_sp.encode.return_value = ["halo"]
        mock_sp.decode.return_value = "hello"

        def fake_load(device):
            if device == "cuda":
                raise RuntimeError("CUDA driver version is insufficient")
            engine.active_device = device
            engine._translator = mock_cpu_translator
            engine._sentencepiece = mock_sp

        with patch.object(engine, "_preferred_device", return_value="cuda"), \
             patch.object(engine, "_load", side_effect=fake_load):
            res = engine.translate("Halo", "ind_Latn", ["English"])

        self.assertEqual(res, {"English": "hello"})
        self.assertEqual(engine.active_device, "cpu")
        self.assertTrue(any("CUDA unavailable" in w for w in warnings))

    def test_nllb_load_failure_cuda_device_cpu_also_fails(self):
        warnings = []
        engine = NllbEngine(
            model_id_or_path="mijuanlo/nllb-200-distilled-600M-ct2-int8",
            device="cuda",
            cache_dir=Path("models/nllb-cache"),
            on_warning=warnings.append,
        )

        def fake_load(device):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            raise RuntimeError("CPU out of memory")

        with patch.object(engine, "_preferred_device", return_value="cuda"), \
             patch.object(engine, "_load", side_effect=fake_load):
            with self.assertRaises(RuntimeError) as ctx:
                engine.translate("Halo", "ind_Latn", ["English"])
            self.assertIn("All NLLB fallbacks failed", str(ctx.exception))
            self.assertIn("CPU out of memory", str(ctx.exception))
        self.assertTrue(any("CUDA unavailable" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()


