from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.updater import check_for_updates, parse_version


class UpdaterTests(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(parse_version("1.0.0"), (1, 0, 0))
        self.assertEqual(parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(parse_version("v2.0"), (2, 0))
        self.assertEqual(parse_version("invalid"), (0,))
        self.assertTrue(parse_version("1.1.0") > parse_version("1.0.0"))
        self.assertTrue(parse_version("2.0.0") > parse_version("1.9.9"))
        self.assertFalse(parse_version("1.0.0") > parse_version("1.0.0"))

    def test_check_for_updates_detects_newer_version(self):
        fake_payload = json.dumps({
            "tag_name": "v1.1.0",
            "html_url": "https://github.com/kenewjr/kizcaption/releases/tag/v1.1.0",
            "body": "Fixes and improvements",
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = fake_payload
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_for_updates("1.0.0")
            self.assertTrue(res["has_update"])
            self.assertEqual(res["latest_version"], "1.1.0")
            self.assertEqual(res["release_url"], "https://github.com/kenewjr/kizcaption/releases/tag/v1.1.0")
            self.assertIsNone(res["error"])

    def test_check_for_updates_same_or_older_version(self):
        fake_payload = json.dumps({
            "tag_name": "v1.0.0",
            "html_url": "https://github.com/kenewjr/kizcaption/releases/tag/v1.0.0",
            "body": "Initial release",
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = fake_payload
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_for_updates("1.0.0")
            self.assertFalse(res["has_update"])
            self.assertEqual(res["latest_version"], "1.0.0")

    def test_check_for_updates_handles_network_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("Connection timeout")):
            res = check_for_updates("1.0.0")
            self.assertFalse(res["has_update"])
            self.assertIsNotNone(res["error"])
            self.assertIn("Connection timeout", res["error"])


if __name__ == "__main__":
    unittest.main()
