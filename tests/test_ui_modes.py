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
from lumacaption.ui.control_panel import ControlPanel


class UiModesAndPresetsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.report_callback_exception = lambda *args: None
        self.device = InputDevice("device:test", "Test Mic", 0, 48_000, "WASAPI")

    def tearDown(self):
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_ez_mode_and_template_presets(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MicrophoneCapture, "devices", return_value=[self.device]), \
                patch.object(ControlPanel, "_sync_overlay", return_value=True):
            store = ConfigStore(Path(directory) / "config.json")
            panel = ControlPanel(self.root, store, AppConfig(), _root)
            self.root.update()

            # 1. Default Mode is EZ mode
            self.assertEqual(panel.ui_mode_var.get(), "ez")
            panel.set_ui_mode("advanced")
            self.assertEqual(panel.ui_mode_var.get(), "advanced")
            panel.set_ui_mode("ez")
            self.assertEqual(panel.ui_mode_var.get(), "ez")

            # 2. Apply Low Template (Hemat)
            panel.source_var.set("Indonesian")
            panel.apply_resource_preset("low")
            self.assertEqual(panel.resource_preset_var.get(), "Hemat")
            self.assertEqual(panel.model_var.get(), "base")
            self.assertEqual(panel.beam_var.get(), "1")
            self.assertEqual(panel.stt_compute_type_var.get(), "int8")
            self.assertEqual(panel.denoise_engine_var.get(), "clarity")
            self.assertIn("HEMAT", panel.preset_badge_var.get())

            # 3. Apply Medium Template (Seimbang) with Indonesian Language
            panel.source_var.set("Indonesian")
            panel.apply_resource_preset("medium")
            self.assertEqual(panel.resource_preset_var.get(), "Seimbang")
            self.assertEqual(panel.model_var.get(), "whisper-small-id")
            self.assertEqual(panel.beam_var.get(), "3")
            self.assertEqual(panel.stt_compute_type_var.get(), "int8_float16")
            self.assertEqual(panel.denoise_engine_var.get(), "dtln")
            self.assertIn("SEIMBANG", panel.preset_badge_var.get())

            # 4. Switch Language to English -> Model adapts dynamically
            panel.source_var.set("English")
            self.root.update()
            self.assertEqual(panel.model_var.get(), "small")

            # 5. Apply High Template (Akurasi) with Indonesian Language
            panel.source_var.set("Indonesian")
            panel.apply_resource_preset("high")
            self.assertEqual(panel.resource_preset_var.get(), "Akurasi")
            self.assertEqual(panel.model_var.get(), "whisper-medium-id")
            self.assertEqual(panel.beam_var.get(), "5")
            self.assertEqual(panel.stt_compute_type_var.get(), "float16")
            self.assertEqual(panel.denoise_engine_var.get(), "hybrid")
            self.assertIn("AKURASI", panel.preset_badge_var.get())

            # 6. Manual change in Advanced field triggers automatic switch to Custom
            panel.model_var.set("base")
            self.root.update()
            self.assertEqual(panel.resource_preset_var.get(), "Custom")
            self.assertIn("KUSTOM", panel.preset_badge_var.get())

            # 7. Save and round-trip persistence
            saved = panel.save(announce=False)
            self.assertIsNotNone(saved)
            loaded_cfg = store.load()[0]
            self.assertEqual(loaded_cfg.resource_preset, "Custom")
            self.assertEqual(loaded_cfg.ui_mode, "ez")

            # 8. Test scroll hint dynamic text
            panel._update_scroll_hint(0.0, 0.5)
            self.assertIn("bawah", panel.scroll_hint_label.cget("text").lower())
            panel._update_scroll_hint(0.3, 0.7)
            self.assertIn("posisi", panel.scroll_hint_label.cget("text").lower())
            panel._update_scroll_hint(0.5, 1.0)
            self.assertIn("bawah", panel.scroll_hint_label.cget("text").lower())
            panel.close()

    def test_engine_tab_scroll_and_dtln_separation(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MicrophoneCapture, "devices", return_value=[self.device]), \
                patch.object(ControlPanel, "_sync_overlay", return_value=True):
            store = ConfigStore(Path(directory) / "config.json")
            panel = ControlPanel(self.root, store, AppConfig(), _root)
            self.root.update()

            # 1. Tab 2: Mesin & VAD has canvas and scrollbar
            self.assertTrue(hasattr(panel, "engine_canvas"))
            self.assertTrue(hasattr(panel, "engine_scrollbar"))
            self.assertTrue(hasattr(panel, "engine_scroll_hint_label"))

            # Switch to Tab 2
            panel.notebook.select(panel.tab_engine)
            self.root.update()

            # Test responsive layout in Tab 2
            # Wide layout (>= 960) -> 2 columns
            panel._resize_engine_layout(SimpleNamespace(width=1000))
            self.root.update()
            self.assertEqual(panel.engine_left.grid_info()["column"], 0)
            self.assertEqual(panel.engine_right.grid_info()["column"], 1)

            # Narrow layout (< 960) -> stacked 1 column
            panel._resize_engine_layout(SimpleNamespace(width=800))
            self.root.update()
            self.assertEqual(panel.engine_left.grid_info()["column"], 0)
            self.assertEqual(panel.engine_right.grid_info()["column"], 0)

            # Test scroll hint on Tab 2
            panel._update_engine_scroll_hint(0.0, 0.5)
            self.assertIn("bawah", panel.engine_scroll_hint_label.cget("text").lower())
            panel._update_engine_scroll_hint(0.0, 1.0)
            self.assertIn("seluruh", panel.engine_scroll_hint_label.cget("text").lower())

            # Test scroll wheel on Tab 2
            panel.engine_canvas.yview_moveto(0)
            panel._scroll_wheel(SimpleNamespace(widget=panel.engine_body, delta=-120))
            self.assertIsNotNone(panel.engine_canvas.yview())

            # 2. Tab 4: Separation of AI Models and DTLN Denoiser
            self.assertTrue(hasattr(panel, "ai_model_keys"))
            self.assertNotIn("dtln", panel.ai_model_keys)
            self.assertNotIn("silero", panel.ai_model_keys)
            self.assertIn("whisper-small-id", panel.ai_model_keys)
            self.assertIn("nllb", panel.ai_model_keys)

            # Check DTLN dedicated card attributes
            self.assertTrue(hasattr(panel, "dtln_status_var"))
            self.assertTrue(hasattr(panel, "dtln_usage_var"))
            self.assertTrue(hasattr(panel, "dtln_download_btn"))
            self.assertTrue(hasattr(panel, "dtln_apply_btn"))

            # Test DTLN toggle when active/inactive
            panel.denoise_engine_var.set("clarity")
            panel._refresh_dtln_status()
            self.assertIn("Tidak Aktif", panel.dtln_usage_var.get())

            panel.denoise_engine_var.set("dtln")
            panel._refresh_dtln_status()
            self.assertIn("AKTIF", panel.dtln_usage_var.get())
            panel.close()


if __name__ == "__main__":
    unittest.main()
