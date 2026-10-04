from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lumacaption.output.styles import CaptionStyle, PRESETS, preset_style
from lumacaption.output.style_io import export_css, import_css, read_css, write_css


class CaptionStylesAndIoTest(unittest.TestCase):
    def test_all_30_presets_valid(self):
        self.assertEqual(len(PRESETS), 30)
        for name in PRESETS:
            style = preset_style(name, f"Test {name}")
            style.validate()
            self.assertEqual(style.preset, name)

    def test_roundtrip_all_profiles(self):
        profiles = {
            1: preset_style("Clean White", "Output 1"),
            2: preset_style("Lavender Glow", "Output 2"),
            3: preset_style("Midnight Card", "Output 3"),
        }
        profiles[1].line_wrap = "nowrap"
        css = export_css(profiles)
        self.assertIn("#caption-1", css)
        self.assertIn("#caption-2", css)
        self.assertIn("#caption-3", css)
        self.assertIn("--lc-line-wrap: nowrap;", css)

        imported = import_css(css)
        self.assertEqual(set(imported.keys()), {1, 2, 3})
        self.assertEqual(imported[1].line_wrap, "nowrap")
        self.assertEqual(imported[2].line_wrap, "wrap")
        for slot in (1, 2, 3):
            self.assertEqual(imported[slot].name, profiles[slot].name)
            self.assertEqual(imported[slot].preset, profiles[slot].preset)
            self.assertEqual(imported[slot].font_family, profiles[slot].font_family)
            self.assertEqual(imported[slot].text_color, profiles[slot].text_color)
            self.assertEqual(imported[slot].font_size, profiles[slot].font_size)

    def test_roundtrip_single_profile(self):
        profiles = {2: preset_style("Cyan Edge", "Output 2")}
        css = export_css(profiles)
        self.assertNotIn("#caption-1", css)
        self.assertIn("#caption-2", css)
        imported = import_css(css)
        self.assertEqual(list(imported.keys()), [2])
        self.assertEqual(imported[2].preset, "Cyan Edge")

    def test_file_io_atomic_and_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "styles.css"
            profiles = {1: preset_style("Synthwave", "Output 1")}
            write_css(file_path, profiles)
            self.assertTrue(file_path.is_file())
            loaded = read_css(file_path)
            self.assertEqual(loaded[1].preset, "Synthwave")

    def test_rejects_arbitrary_or_corrupted_css(self):
        # Missing header
        with self.assertRaises(ValueError):
            import_css("#caption-1 { --lc-font-size: 48px; }")

        # Unknown selector
        with self.assertRaises(ValueError):
            import_css("/* LumaCaption CSS v1 — caption styles only */\nbody { color: red; }")

        # Invalid property
        with self.assertRaises(ValueError):
            import_css("/* LumaCaption CSS v1 — caption styles only */\n#caption-1 { --lc-malicious: evil; }")

        # Exceeds max bytes
        huge = "/* LumaCaption CSS v1 — caption styles only */\n" + ("/* x */\n" * 10000)
        with self.assertRaises(ValueError):
            import_css(huge)


if __name__ == "__main__":
    unittest.main()
