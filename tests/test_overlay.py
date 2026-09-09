from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

_root = Path(__file__).resolve().parents[1]
for _p in (str(_root / "src" / "lumacaption"), str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from config import OverlayConfig, TargetConfig
from output.overlay_server import OverlayServer


class OverlayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = OverlayServer(
            OverlayConfig(port=0, font_family="Georgia", font_size=64),
            [TargetConfig(name) for name in ("English", "Japanese", "Chinese (Simplified)")],
            Path(__file__).parents[1] / "output" / "overlay.html",
            clear_after=0,
        )
        await self.server.start()
        self.addAsyncCleanup(self.server.stop)
        self.server.config.port = self.server.server.sockets[0].getsockname()[1]

    async def receive(self, socket):
        return json.loads(await asyncio.wait_for(socket.recv(), timeout=2))

    def socket_url(self, language):
        return self.server.url(language).replace("http://", "ws://", 1).replace("/overlay?", "/ws?", 1)

    async def test_http_style_and_unknown_language(self):
        def fetch():
            with urlopen(self.server.preview_url("Chinese (Simplified)"), timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertIn("default-src 'none'", response.headers["Content-Security-Policy"])
                self.assertIn("--caption-font", response.read().decode("utf-8"))
            with self.assertRaises(HTTPError) as error:
                urlopen(self.server.url("Unknown"), timeout=2)
            self.assertEqual(error.exception.code, 404)
            error.exception.close()

        await asyncio.to_thread(fetch)

    async def test_publish_reconnect_and_auto_clear(self):
        async with connect(self.socket_url("English")) as english, connect(self.socket_url("Japanese")) as japanese:
            self.assertEqual(await self.receive(english), {"lang": "English", "text": ""})
            self.assertEqual(await self.receive(japanese), {"lang": "Japanese", "text": ""})
            translations = {"English": "Hello <script>alert(1)</script>", "Japanese": "こんにちは"}
            await self.server.publish(translations)
            for socket, language in ((english, "English"), (japanese, "Japanese")):
                self.assertEqual(await self.receive(socket), {"lang": language, "text": translations[language]})
            async with connect(self.socket_url("English")) as reconnected:
                self.assertEqual((await self.receive(reconnected))["text"], translations["English"])
            self.server.clear_after = 0.1
            self.assertEqual((await self.receive(english))["text"], "")
            self.assertEqual((await self.receive(japanese))["text"], "")
        async with connect(self.socket_url("English")) as reconnected:
            self.assertEqual((await self.receive(reconnected))["text"], "")
        async with connect(self.socket_url("Unknown")) as invalid:
            with self.assertRaises(ConnectionClosedError) as error:
                await self.receive(invalid)
            self.assertEqual(error.exception.rcvd.code, 1008)

    async def test_all_languages_multiline_broadcast(self):
        async with connect(self.socket_url("all")) as all_client:
            initial = await self.receive(all_client)
            self.assertEqual(initial, {"all": {"English": "", "Japanese": "", "Chinese (Simplified)": ""}})
            translations = {
                "English": "Hello world",
                "Japanese": "こんにちは世界",
                "Chinese (Simplified)": "你好世界",
            }
            await self.server.publish(translations)
            received = await self.receive(all_client)
            self.assertEqual(received, {"all": translations})
            self.server.clear_after = 0.1
            cleared = await self.receive(all_client)
            self.assertEqual(cleared, {"all": {"English": "", "Japanese": "", "Chinese (Simplified)": ""}})


    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable for caption sizing check")
    async def test_caption_height_fit_and_resize(self):
        html = self.server.html
        function = html[html.index("    function fitCaption()"):html.index("    new ResizeObserver")]
        script = r"""
const assert = require('node:assert/strict');
const caption = {style: {fontSize: '', removeProperty() {this.fontSize = '';}}};
const document = {body: {}};
let requested = 64, innerHeight = 300, reads = 0;
const getComputedStyle = element => element === caption
  ? {fontSize: caption.style.fontSize || `${requested}px`}
  : {paddingTop: '18px', paddingBottom: '18px'};
const stage = {get offsetHeight() {
  assert(++reads < 20, 'Fit loop must be bounded');
  return 30 + 4 * parseFloat(getComputedStyle(caption).fontSize);
}};
""" + function + r"""
fitCaption();
assert.equal(caption.style.fontSize, '58px');
assert(stage.offsetHeight <= innerHeight - 36);
reads = 0; innerHeight = 500;
fitCaption();
assert.equal(caption.style.fontSize, '64px', 'Grow back to configured size on resize');
reads = 0; requested = 16;
fitCaption();
assert.equal(caption.style.fontSize, '16px', 'Never enlarge beyond configured size');
reads = 0; innerHeight = 110;
fitCaption();
assert.equal(caption.style.fontSize, '11px', 'Fit very short sources without truncating text');
"""
        result = await asyncio.to_thread(
            subprocess.run, [shutil.which("node"), "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
