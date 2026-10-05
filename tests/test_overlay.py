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

from lumacaption.config import OverlayConfig, TargetConfig
from lumacaption.output.overlay_server import OverlayServer, static_overlay_url


class OverlayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = OverlayServer(
            OverlayConfig(port=0, font_family="Georgia", font_size=64),
            [TargetConfig(name, profile=i) for i, name in enumerate(("English", "Japanese", "Chinese (Simplified)"), start=1)],
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
            # Verify clean static overlay.html endpoint
            static_url = static_overlay_url(self.server.config)
            self.assertTrue(static_url.endswith("/overlay.html"))
            with urlopen(static_url, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("KizCaption Browser Source", response.read().decode("utf-8"))

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

    async def test_slot_profiles_and_live_style_update(self):
        url_profile_1 = self.server.url(profile=1).replace("http://", "ws://", 1).replace("/overlay?", "/ws?", 1)
        async with connect(url_profile_1) as client_1:
            init_msg = await self.receive(client_1)
            self.assertEqual(init_msg["slot"], 1)
            self.assertEqual(init_msg["lang"], "English")

            # Update style live without server restart
            new_cfg = self.server.config
            new_cfg.profiles[0].font_size = 72
            await self.server.update_styles(new_cfg)

            style_msg = await self.receive(client_1)
            self.assertEqual(style_msg["type"], "style_update")
            self.assertEqual(style_msg["styles"]["1"]["font_size"], 72)

            # Publish translation to slot 1
            await self.server.publish({"English": "Slot 1 Updated"})
            pub_msg = await self.receive(client_1)
            self.assertEqual(pub_msg["text"], "Slot 1 Updated")
            self.assertEqual(pub_msg["slot"], 1)


    async def test_stage_gap_spacing_and_live_updates(self):
        html = self.server.html
        self.assertRegex(html, r'body\.multi-line \.caption-slot\s*\{[^}]*line-height:\s*0\s*;')
        self.assertRegex(html, r'body\.multi-line \.caption-card\s*\{[^}]*margin-block:\s*0(?:\s*!important)?\s*;')
        self.assertRegex(html, r'body\.multi-line \.caption-card\s*\{[^}]*vertical-align:\s*top\s*;')
        self.assertIn('if (data.gap !== undefined)', html)

        url = self.server.url(profile="all").replace("http://", "ws://", 1).replace("/overlay?", "/ws?", 1)
        async with connect(url) as client:
            self.assertEqual((await self.receive(client))["gap"], self.server.config.gap)
            for gap in (0, 1, 40, 80):
                self.server.config.gap = gap
                await self.server.update_styles(self.server.config)
                message = await self.receive(client)
                self.assertEqual(message["type"], "style_update")
                self.assertEqual(message["gap"], gap)
                self.assertEqual(message["styles"]["1"]["margin_y"], 16)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable for overlay routing check")
    async def test_stage_alignment_and_multilingual_routing(self):
        html = self.server.html
        bundled = _root / "src" / "lumacaption" / "output" / "overlay.html"
        self.assertEqual(html, bundled.read_text(encoding="utf-8"))
        for side in ("left", "center", "right"):
            self.assertRegex(html, rf'body\[data-anchor\$="{side}"\]\s*\{{[^}}]*text-align:\s*{side}\s*;')
        self.assertRegex(html, r'main\s*\{[^}]*justify-items:\s*inherit\s*;')
        self.assertNotIn("slotElem.style.textAlign", html,
                         "Profile text alignment must not move the card away from the stage anchor")
        self.assertIn('card.style.setProperty("--lc-text-align", s.align || "center")', html)

        parameters = html[html.index("    const params ="):html.index("    const allowedFonts")]
        query = next(line for line in html.splitlines() if "const query =" in line)
        script = r"""
const assert = require('node:assert/strict');
for (const [search, expected] of [
  ['', 'profile=all'],
  ['?profile=all', 'profile=all'],
  ['?lang=all', 'profile=all'],
  ['?lang=All', 'profile=all'],
  ['?lang=*', 'profile=all'],
  ['?profile=All', 'profile=all'],
  ['?profile=*', 'profile=all'],
  ['?profile=3', 'profile=3'],
  ['?profile=3&lang=all', 'profile=3'],
  ['?profile=all&lang=Japanese', 'profile=all'],
  ['?lang=Japanese', 'lang=Japanese'],
  ['?lang=Chinese%20(Simplified)', 'lang=Chinese%20(Simplified)'],
]) {
  const location = {search};
""" + parameters + query + r"""
  assert.equal(query, expected, search || 'Default multilingual URL');
}
"""
        result = await asyncio.to_thread(
            subprocess.run, [shutil.which("node"), "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    async def test_client_count_tracking_and_diagnostics(self):
        self.assertEqual(self.server.active_client_count, 0)
        diag0 = self.server.get_diagnostics()
        self.assertTrue(diag0["running"])
        self.assertEqual(diag0["clients"], 0)
        self.assertEqual(diag0["last_publish_clients"], 0)

        async with connect(self.socket_url("English")) as c1:
            await self.receive(c1)
            self.assertEqual(self.server.active_client_count, 1)

            async with connect(self.socket_url("Japanese")) as c2:
                await self.receive(c2)
                self.assertEqual(self.server.active_client_count, 2)

                await self.server.publish({"English": "Hello", "Japanese": "Konnichiwa"})
                await self.receive(c1)
                await self.receive(c2)

                diag2 = self.server.get_diagnostics()
                self.assertEqual(diag2["clients"], 2)
                self.assertEqual(diag2["last_publish_clients"], 2)
                self.assertGreater(diag2["last_publish_time"], 0)

            await asyncio.sleep(0.05)
            self.assertEqual(self.server.active_client_count, 1)

        await asyncio.sleep(0.05)
        self.assertEqual(self.server.active_client_count, 0)

    async def test_zero_clients_publish(self):
        self.assertEqual(self.server.active_client_count, 0)
        await self.server.publish({"English": "Zero client test", "Japanese": "Zero"})
        diag = self.server.get_diagnostics()
        self.assertEqual(diag["last_publish_clients"], 0)
        self.assertGreater(diag["last_publish_time"], 0)

    async def test_slow_client_timeout_isolation(self):
        class HangingClient:
            def __init__(self):
                self.closed = False

            async def send(self, msg):
                await asyncio.sleep(5.0)

            async def close(self):
                self.closed = True

        hanging = HangingClient()
        self.server.clients["test_group"].add(hanging)

        t0 = asyncio.get_event_loop().time()
        delivered = await self.server._broadcast_raw("test_group", "hello")
        elapsed = asyncio.get_event_loop().time() - t0
        self.assertEqual(delivered, 0)
        self.assertNotIn(hanging, self.server.clients["test_group"])
        self.assertTrue(hanging.closed, "Slow client must be explicitly closed on timeout")
        self.assertLess(elapsed, 2.5)


if __name__ == "__main__":
    unittest.main()
