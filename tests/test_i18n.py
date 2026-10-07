from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.audio.capture import InputDevice, MicrophoneCapture
from lumacaption.config import AppConfig, ConfigStore
from lumacaption.i18n import t, DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from lumacaption.pipeline import PipelineEvent
from lumacaption.ui.control_panel import ControlPanel


class I18nAndLanguageSwitchTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.addCleanup(self.root.destroy)
        self.device = InputDevice("device:test", "Test Mic", 0, 48_000, "WASAPI")

    def test_i18n_translation_keys_and_fallbacks(self):
        self.assertEqual(DEFAULT_LANGUAGE, "id")
        self.assertIn("id", SUPPORTED_LANGUAGES)
        self.assertIn("en", SUPPORTED_LANGUAGES)

        # Indonesian: natural, modern, non-stiff phrasing
        id_title = t("card_template_title", "id")
        self.assertIn("TEMPLATE CEPAT", id_title)
        self.assertIn("Santai", t("mode_ez", "id"))
        self.assertEqual(t("btn_start_caption", "id"), "Mulai Caption")
        self.assertEqual(t("btn_stop_caption", "id"), "Hentikan")
        self.assertEqual(t("btn_advanced_open", "id"), "Buka")
        self.assertEqual(t("btn_advanced_close", "id"), "Tutup")
        self.assertEqual(t("btn_light_mode", "id"), "Mode Terang")
        self.assertEqual(t("btn_dark_mode", "id"), "Mode Gelap")

        # English: clean standard desktop terminology
        en_title = t("card_template_title", "en")
        self.assertIn("QUICK TEMPLATES", en_title)
        self.assertIn("EZ Mode", t("mode_ez", "en"))
        self.assertEqual(t("btn_start_caption", "en"), "Start Caption")
        self.assertEqual(t("btn_stop_caption", "en"), "Stop Caption")
        self.assertEqual(t("btn_advanced_open", "en"), "Open")
        self.assertEqual(t("btn_advanced_close", "en"), "Close")
        self.assertEqual(t("btn_light_mode", "en"), "Light Mode")
        self.assertEqual(t("btn_dark_mode", "en"), "Dark Mode")

        self.assertEqual(t("status_ready", "id"), "SIAP")
        self.assertEqual(t("status_ready", "en"), "READY")
        self.assertEqual(t("status_preparing", "id"), "MENYIAPKAN")
        self.assertEqual(t("status_preparing", "en"), "PREPARING")
        self.assertEqual(t("status_closing", "id"), "MENUTUP")
        self.assertEqual(t("status_closing", "en"), "CLOSING")
        self.assertEqual(t("transcript_placeholder", "id"), "Hasil tampil setelah ucapan selesai.")
        self.assertEqual(t("transcript_placeholder", "en"), "Results appear after speech ends.")
        self.assertEqual(t("metrics_placeholder", "id"), "Waktu STT / MT tampil setelah caption pertama")
        self.assertEqual(t("metrics_placeholder", "en"), "STT / MT time appears after first caption")
        self.assertEqual(t("runtime_preparing_local", "id"), "Menyiapkan model lokal…")
        self.assertEqual(t("runtime_preparing_local", "en"), "Preparing local models…")

        # Fallback for unknown key
        self.assertEqual(t("unknown_key_xyz", "id"), "unknown_key_xyz")

    def test_app_config_ui_language_validation_and_roundtrip(self):
        cfg = AppConfig()
        self.assertEqual(cfg.ui_language, "id")
        cfg.validate()

        cfg_en = AppConfig(ui_language="en")
        self.assertEqual(cfg_en.ui_language, "en")
        cfg_en.validate()

        with self.assertRaises(ValueError):
            AppConfig(ui_language="invalid").validate()

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            store = ConfigStore(p)
            store.save(cfg_en)
            loaded, _ = store.load()
            self.assertEqual(loaded.ui_language, "en")

    def test_ui_language_toggle_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MicrophoneCapture, "devices", return_value=[self.device]), \
                patch.object(ControlPanel, "_sync_overlay", return_value=True):
            store = ConfigStore(Path(directory) / "config.json")
            cfg = AppConfig(ui_language="id")
            panel = ControlPanel(self.root, store, cfg, _root)
            self.root.update()

            # Initial state: Indonesian
            self.assertEqual(panel.config.ui_language, "id")
            self.assertIn("English", panel.lang_btn.cget("text"))
            self.assertIn("MULAI CAPTION", panel.start_button.cget("text"))
            self.assertIn("Simpan", panel.save_button.cget("text"))
            self.assertEqual(panel.target_vars[1].get(), "Tidak digunakan")
            self.assertIn("Tidak digunakan", panel.target_boxes[1]["values"])
            self.assertTrue(panel.left.winfo_ismapped())
            self.assertTrue(panel.right.winfo_ismapped())

            # Toggle to English
            panel.toggle_ui_language()
            self.root.update()
            self.assertEqual(panel.config.ui_language, "en")
            self.assertIn("Bahasa Indonesia", panel.lang_btn.cget("text"))
            self.assertIn("START CAPTION", panel.start_button.cget("text"))
            self.assertIn("Save", panel.save_button.cget("text"))
            self.assertEqual(panel.target_vars[1].get(), "Not used")
            self.assertIn("Not used", panel.target_boxes[1]["values"])

            # Test target re-selection: choose a language, then can pick 'Not used' again
            panel.target_vars[1].set("Japanese")
            self.assertIn("Not used", panel.target_boxes[1]["values"])
            panel.target_vars[1].set("Not used")

            # Tab 1 widgets MUST remain mapped and visible (not blank canvas!)
            self.assertTrue(panel.left.winfo_ismapped())
            self.assertTrue(panel.right.winfo_ismapped())
            self.assertTrue(panel.engine_left.winfo_ismapped())
            self.assertTrue(panel.engine_right.winfo_ismapped())

            # Preset buttons and badge in English
            self.assertIn("Low Spec", panel.btn_preset_low.cget("text"))
            self.assertNotIn("Hemat", panel.btn_preset_low.cget("text"))
            self.assertIn("Recommended", panel.btn_preset_med.cget("text"))

            # Vocab terms status in English
            self.assertIn("terms learned", panel.vocab_status_var.get())

            # DTLN status in English
            self.assertTrue("Available" in panel.dtln_status_var.get() or "Not downloaded" in panel.dtln_status_var.get())

            # Tab 3 CaptionEditor must be in English
            self.assertEqual(panel.caption_editors[0].lang, "en")

            # Tab 4 Table headings must be in English
            col_heading = panel.res_tree.heading("model")["text"]
            self.assertEqual(col_heading, "Component & Model")

            # Check config persisted to store
            loaded_cfg, _ = store.load()
            self.assertEqual(loaded_cfg.ui_language, "en")

            # Toggle back to Indonesian
            panel.toggle_ui_language()
            self.root.update()
            self.assertEqual(panel.config.ui_language, "id")
            self.assertIn("English", panel.lang_btn.cget("text"))
            self.assertIn("MULAI CAPTION", panel.start_button.cget("text"))
            self.assertIn("Simpan", panel.save_button.cget("text"))
            self.assertIn("Hemat", panel.btn_preset_low.cget("text"))
            self.assertIn("kata dipelajari", panel.vocab_status_var.get())

            # Tab 1 widgets still visible
            self.assertTrue(panel.left.winfo_ismapped())
            self.assertTrue(panel.right.winfo_ismapped())
            self.assertEqual(panel.caption_editors[0].lang, "id")
            col_heading_id = panel.res_tree.heading("model")["text"]
            self.assertEqual(col_heading_id, "Komponen & Model")

            loaded_cfg2, _ = store.load()
            self.assertEqual(loaded_cfg2.ui_language, "id")

    def test_live_state_preserved_across_ui_reload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MicrophoneCapture, "devices", return_value=[self.device]), \
                patch.object(ControlPanel, "_sync_overlay", return_value=True):
            store = ConfigStore(Path(directory) / "config.json")
            cfg = AppConfig(ui_language="id")
            panel = ControlPanel(self.root, store, cfg, _root)
            self.root.update()

            mock_pipeline = SimpleNamespace(running=True, inference_busy=False)
            panel.pipeline = mock_pipeline
            panel._set_live_controls(True)
            self.assertEqual(panel.start_button.cget("text"), "HENTIKAN CAPTION")
            self.assertEqual(str(panel.save_button.cget("state")), "disabled")

            panel.toggle_ui_language()
            self.root.update()

            self.assertEqual(panel.config.ui_language, "en")
            self.assertEqual(panel.status_var.get(), "LIVE")
            self.assertEqual(panel.start_button.cget("text"), "STOP CAPTION")
            self.assertEqual(str(panel.save_button.cget("state")), "disabled")

    def test_model_manager_english_localization(self):
        from lumacaption.model_manager import resource_details, evaluate_vram_safety

        details_en = resource_details("small", "cpu", lang="en")
        self.assertIn("Language Suitability:", details_en)
        self.assertIn("Estimated RAM:", details_en)
        self.assertIn("Configuration:", details_en)
        self.assertIn("Estimates, not guarantees.", details_en)

        details_id = resource_details("small", "cpu", lang="id")
        self.assertIn("Kecocokan Bahasa:", details_id)
        self.assertIn("RAM estimasi:", details_id)
        self.assertIn("Konfigurasi:", details_id)

        # VRAM safety in English
        st_cpu, bdg_cpu, dtl_cpu = evaluate_vram_safety("small", "nllb", "cpu", "cpu", lang="en")
        self.assertEqual(st_cpu, "cpu")
        self.assertIn("CPU Mode", bdg_cpu)

        with patch("lumacaption.model_manager.get_vram_free_mb", return_value=8192.0):
            st_safe, bdg_safe, dtl_safe = evaluate_vram_safety("small", "nllb", "cuda", "cuda", lang="en")
            self.assertEqual(st_safe, "safe")
            self.assertIn("VRAM Safe", bdg_safe)
            self.assertIn("Free of OOM risk", dtl_safe)

    def test_control_panel_runtime_messages_localization(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MicrophoneCapture, "devices", return_value=[self.device]), \
                patch.object(ControlPanel, "_sync_overlay", return_value=True):
            store = ConfigStore(Path(directory) / "config.json")
            cfg = AppConfig(ui_language="en")
            panel = ControlPanel(self.root, store, cfg, _root)
            self.root.update()

            # Initial placeholders in English
            self.assertEqual(panel.status_var.get(), "READY")
            self.assertEqual(panel.transcript_var.get(), "Results appear after speech ends.")
            self.assertEqual(panel.metrics_var.get(), "STT / MT time appears after first caption")

            # Localize event message helper
            self.assertEqual(
                panel._localize_event_message("Memuat Whisper small ke CUDA (float16)..."),
                "Loading Whisper small on CUDA (float16)...",
            )
            self.assertEqual(
                panel._localize_event_message("Memuat model translasi NLLB-200 ke CUDA (float16)..."),
                "Loading NLLB-200 translation model on CUDA (float16)...",
            )
            self.assertEqual(
                panel._localize_event_message("LIVE • hasil tampil setelah jeda bicara"),
                "LIVE • results appear after speech pause",
            )
            self.assertEqual(
                panel._localize_event_message("Menyiapkan translasi NLLB-200 [tanpa translasi]…"),
                "Preparing NLLB-200 translation [no translation]…",
            )

            # Saving settings preserves ui_language="en"
            saved_config = panel.save(announce=False)
            self.assertIsNotNone(saved_config)
            self.assertEqual(saved_config.ui_language, "en")
            self.assertEqual(panel.config.ui_language, "en")

            # Events trigger English status and device info
            panel._handle_event(PipelineEvent("started", "LIVE • hasil tampil setelah jeda bicara", {"device": "Fake Mic", "sample_rate": 16000}))
            self.assertIn("Active • Fake Mic • 16000 Hz", panel.audio_device_var.get())
            self.assertEqual(panel.activity_var.get(), "LIVE • results appear after speech pause")

            panel._handle_event(PipelineEvent("speech", "Suara terdeteksi"))
            self.assertEqual(panel.activity_var.get(), "Voice detected • results after speech pause")


if __name__ == "__main__":
    unittest.main()

