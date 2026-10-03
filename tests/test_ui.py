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
from lumacaption.pipeline import PipelineEvent
from lumacaption.ui.control_panel import ControlPanel, UNUSED_TARGET


class DashboardTests(unittest.TestCase):
    def test_resize_settings_and_audio_diagnostics(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.addCleanup(root.destroy)
        errors = []
        root.report_callback_exception = lambda *args: errors.append(args)
        device = InputDevice("device:test", "Test Microphone", 0, 48_000, "WASAPI")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MicrophoneCapture, "devices", return_value=[device]), \
                patch.object(ControlPanel, "_sync_overlay", return_value=True):
            store = ConfigStore(Path(directory) / "config.json")
            panel = ControlPanel(root, store, AppConfig(), Path(__file__).parents[1])
            sizes = ((1180, 820), (620, 560), (960, 700), (1440, 900), (620, 560))
            for (width, height), live in [(size, live) for live in (False, True) for size in sizes]:
                panel._set_live_controls(live)
                root.geometry(f"{width}x{height}")
                root.update()
                expected_wide = panel.canvas.winfo_width() >= 1020
                self.assertEqual(panel.right.grid_info()["column"], 1 if expected_wide else 0)
                self.assertEqual(panel.body.winfo_width(), panel.canvas.winfo_width())
                for column in (panel.left, panel.right):
                    self.assertLessEqual(column.winfo_x() + column.winfo_width(), panel.body.winfo_width())
                for button in (panel.save_button, panel.test_button, panel.start_button):
                    self.assertTrue(button.winfo_viewable())
                    self.assertGreaterEqual(button.winfo_rootx(), root.winfo_rootx())
                    self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), root.winfo_rootx() + root.winfo_width())
                    self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), root.winfo_rooty() + root.winfo_height())
                    self.assertGreaterEqual(button.winfo_width(), button.winfo_reqwidth())

            panel._toggle_advanced()
            root.update()
            self.assertTrue(panel._advanced_body.winfo_viewable())
            panel.canvas.yview_moveto(0)
            panel._scroll_wheel(SimpleNamespace(widget=panel.body, delta=-120))
            self.assertGreater(panel.canvas.yview()[0], 0)
            panel._reveal_focus(SimpleNamespace(widget=panel.mic_box))
            root.update()
            self.assertGreaterEqual(panel.mic_box.winfo_rooty(), panel.canvas.winfo_rooty())
            self.assertLess(panel.mic_box.winfo_rooty(), panel.canvas.winfo_rooty() + panel.canvas.winfo_height())

            self.assertEqual([v.get() for v in panel.target_vars], ["English", UNUSED_TARGET, UNUSED_TARGET])
            for widget, state in panel._editable:
                if "textvariable" in widget.keys() and str(widget.cget("textvariable")) in {str(v) for v in panel.target_vars}:
                    self.assertEqual(state, "readonly")
                    self.assertNotIn("", widget.cget("values"))
            self.assertEqual(panel.beam_var.get(), "3")
            self.assertEqual(panel.hotwords_var.get(), "")
            self.assertEqual(panel.duration_var.get(), "20")

            panel.font_var.set("Georgia")
            panel.font_size_var.set("64")
            panel.theme_var.set("card")
            panel.text_color_var.set("#FFEE55")
            panel.outline_color_var.set("#000000")
            panel.beam_var.set("5")
            panel.hotwords_var.set("istilah_uji")
            panel.duration_var.set("35")
            saved = panel.save(announce=False)
            self.assertIsNotNone(saved)
            loaded_cfg = store.load()[0]
            self.assertEqual(loaded_cfg.overlay.font_family, "Georgia")
            self.assertEqual(loaded_cfg.overlay.font_size, 64)
            self.assertEqual(loaded_cfg.overlay.theme, "card")
            self.assertEqual(loaded_cfg.overlay.text_color, "#FFEE55")
            self.assertEqual(loaded_cfg.overlay.outline_color, "#000000")
            self.assertEqual(loaded_cfg.whisper_beam_size, 5)
            self.assertEqual(loaded_cfg.whisper_hotwords, "istilah_uji")
            self.assertEqual(loaded_cfg.max_utterance_seconds, 35)
            self.assertIn("font=Georgia", panel.overlay_url("English"))
            self.assertIn("size=64", panel.overlay_url("English"))
            self.assertIn("theme=card", panel.overlay_url("English"))
            self.assertIn("color=%23FFEE55", panel.overlay_url("English"))
            self.assertIn("lang=all", panel.overlay_url("all"))

            panel._handle_event(PipelineEvent("preparing", "Memuat model…"))
            self.assertEqual(panel.status_var.get(), "MENYIAPKAN")
            self.assertEqual(panel.runtime_var.get(), "Memuat model…")

            panel._handle_event(PipelineEvent("model_ready", "Model siap", {
                "model": "large-v3-turbo", "stt_device": "cuda", "mt_device": "cuda",
            }))
            self.assertIn("large-v3-turbo", panel.runtime_var.get())
            self.assertIn("CUDA", panel.runtime_var.get())

            panel._handle_event(PipelineEvent("metrics", "Waktu pemrosesan", {
                "stt_ms": 250.0, "mt_ms": 50.0, "after_vad_ms": 320.0,
            }))
            self.assertIn("STT 250 ms", panel.metrics_var.get())
            self.assertIn("MT 50 ms", panel.metrics_var.get())

            panel._reset_audio()
            for now in (100, 101, 102, 103):
                with patch("lumacaption.ui.control_panel.time.monotonic", return_value=now):
                    panel._handle_event(PipelineEvent("audio_level", "", {"rms": 0.004, "peak": 0.008}))
            self.assertIn("Sinyal tetap", panel.audio_hint_var.get())
            self.assertEqual(panel.rms_var.get(), "-48 dBFS")
            self.assertEqual(panel.peak_var.get(), "-42 dBFS")
            with patch("lumacaption.ui.control_panel.time.monotonic", return_value=104):
                panel._handle_event(PipelineEvent("vad_probability", "", {"probability": 0.9, "speaking": True}))
                panel._update_audio(0.1, 0.3)
                self.assertIn("Ucapan terdeteksi", panel.audio_hint_var.get())
                self.assertEqual(panel.vad_status_var.get(), "90%")
                panel._update_audio(0.5, 1)
                self.assertIn("clipping", panel.audio_hint_var.get())
            # Overlay diagnostics tests
            self.assertIn("0", panel.overlay_clients_var.get())
            panel._handle_event(PipelineEvent("clients_changed", "3"))
            self.assertIn("3 browser source terhubung", panel.overlay_clients_var.get())
            panel._handle_event(PipelineEvent("clients_changed", "1"))
            self.assertIn("1 browser source terhubung", panel.overlay_clients_var.get())
            panel._handle_event(PipelineEvent("clients_changed", "0"))
            self.assertIn("0 terhubung (tampilan pasif)", panel.overlay_clients_var.get())

            panel._handle_event(PipelineEvent("published", "Caption terkirim ke server overlay"))
            self.assertEqual(panel.activity_var.get(), "Caption terkirim ke server overlay")

            # Verify started and listening events lock live controls and stop button
            panel._handle_event(PipelineEvent("started", "Listening"))
            self.assertEqual(panel.status_var.get(), "LIVE")
            self.assertEqual(panel.start_button.cget("text"), "HENTIKAN CAPTION")
            self.assertEqual(str(panel.save_button.cget("state")), "disabled")

            panel._handle_event(PipelineEvent("listening", "Listening active", {"device": "Test Mic", "sample_rate": 16000}))
            self.assertEqual(panel.status_var.get(), "LIVE")
            self.assertEqual(panel.start_button.cget("text"), "HENTIKAN CAPTION")
            self.assertEqual(str(panel.save_button.cget("state")), "disabled")

            # Target reselection check: choose a language, verify 'Tidak digunakan' is available, then re-select it
            panel.target_vars[1].set("Japanese")
            self.assertIn(UNUSED_TARGET, panel.target_boxes[1]["values"])
            panel.target_vars[1].set(UNUSED_TARGET)
            self.assertEqual(panel.target_vars[1].get(), UNUSED_TARGET)

            # Multiple reload UI stability: no widget leak in _editable
            initial_count = len(panel._editable)
            panel._reload_ui()
            panel._reload_ui()
            self.assertEqual(len(panel._editable), initial_count)
            for widget, _ in panel._editable:
                self.assertTrue(widget.winfo_exists())

            panel._handle_event(PipelineEvent("error", "Test failure"))
            panel._handle_event(PipelineEvent("stopped", ""))
            panel._finish_stop()
            self.assertEqual(panel.status_var.get(), "ERROR")
            self.assertEqual(panel.activity_var.get(), "Test failure")
            self.assertEqual(panel.vad_status_var.get(), "— %")
            self.assertEqual(panel.start_button.cget("text"), "MULAI CAPTION")
            self.assertEqual(str(panel.save_button.cget("state")), "normal")
            self.assertFalse(errors, errors)
            panel._closed = True


if __name__ == "__main__":
    unittest.main()
