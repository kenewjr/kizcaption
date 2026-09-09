from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
import http
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from config import OverlayConfig, TargetConfig


def overlay_url(config: OverlayConfig, language: str, **extra: str) -> str:
    host = f"[{config.host}]" if ":" in config.host else config.host
    query = urlencode({
        "lang": language,
        "font": config.font_family,
        "size": config.font_size,
        "color": config.text_color,
        "outline": config.outline_color,
        "theme": config.theme,
        **extra,
    })
    return f"http://{host}:{config.port}/overlay?{query}"


class OverlayServer:
    def __init__(
        self,
        config: OverlayConfig,
        targets: list[TargetConfig],
        html_path: str | Path,
        clear_after: int = 8,
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.languages = {target.language for target in targets}
        self.target_order = [target.language for target in targets]
        self.html = Path(html_path).read_text(encoding="utf-8")
        self.clear_after = clear_after
        self.on_status = on_status or (lambda _level, _message: None)
        self.clients: dict[str, set] = defaultdict(set)
        self.current: dict[str, tuple[str, float]] = {}
        self.server = None
        self.clear_task: asyncio.Task | None = None

    def url(self, language: str) -> str:
        return overlay_url(self.config, language)

    def preview_url(self, language: str, text: str = "LumaCaption siap tampil di OBS") -> str:
        return overlay_url(self.config, language, preview="1", text=text)

    async def start(self) -> None:
        if self.server is not None:
            return
        from websockets.asyncio.server import serve

        def process_request(connection, request):
            parts = urlsplit(request.path)
            if parts.path in ("/", "/overlay"):
                language = parse_qs(parts.query).get("lang", [""])[0]
                if language and language not in self.languages and language not in ("all", "All", "*"):
                    return connection.respond(http.HTTPStatus.NOT_FOUND, "Unknown language\n")
                response = connection.respond(http.HTTPStatus.OK, self.html)
                response.headers["Content-Type"] = "text/html; charset=utf-8"
                response.headers["Cache-Control"] = "no-store"
                response.headers["Content-Security-Policy"] = (
                    "default-src 'none'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src ws: wss:"
                )
                return response
            if parts.path != "/ws":
                return connection.respond(http.HTTPStatus.NOT_FOUND, "Not found\n")
            return None

        try:
            self.server = await serve(
                self._handler,
                self.config.host,
                self.config.port,
                process_request=process_request,
                ping_interval=20,
                ping_timeout=20,
                max_size=16_384,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Overlay port {self.config.port} unavailable. Close the other app or choose another port."
            ) from exc
        self.clear_task = asyncio.create_task(self._clear_stale(), name="overlay-clear")
        self.on_status("overlay_ready", f"Browser Source ready: http://{self.config.host}:{self.config.port}")

    async def _handler(self, websocket) -> None:
        parts = urlsplit(websocket.request.path)
        language = parse_qs(parts.query).get("lang", [""])[0]
        is_all = language in ("all", "All", "*")
        if parts.path != "/ws" or (not is_all and language not in self.languages):
            await websocket.close(1008, "Unknown language or endpoint")
            return
        group = "all" if is_all else language
        self.clients[group].add(websocket)
        try:
            if is_all:
                all_data = {
                    lang: self.current.get(lang, ("", 0.0))[0]
                    for lang in self.target_order
                }
                await websocket.send(json.dumps({"all": all_data}, ensure_ascii=False))
            else:
                text = self.current.get(group, ("", 0.0))[0]
                await websocket.send(json.dumps({"lang": group, "text": text}, ensure_ascii=False))
            await websocket.wait_closed()
        finally:
            self.clients[group].discard(websocket)

    async def publish(self, translations: dict[str, str]) -> None:
        now = time.monotonic()
        for language, text in translations.items():
            if language in self.languages:
                self.current[language] = (text, now)
                await self._broadcast(language, text)
        if self.clients.get("all"):
            all_dict = {
                lang: translations.get(lang, self.current.get(lang, ("", 0.0))[0])
                for lang in self.target_order
            }
            message = json.dumps({"all": all_dict}, ensure_ascii=False)
            await self._broadcast_raw("all", message)

    async def _broadcast(self, language: str, text: str) -> None:
        message = json.dumps({"lang": language, "text": text}, ensure_ascii=False)
        await self._broadcast_raw(language, message)

    async def _broadcast_raw(self, group: str, message: str) -> None:
        clients = list(self.clients[group])
        if clients:
            results = await asyncio.gather(
                *(client.send(message) for client in clients),
                return_exceptions=True,
            )
            for client, result in zip(clients, results, strict=True):
                if isinstance(result, Exception):
                    self.clients[group].discard(client)

    async def _clear_stale(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.5)
                if not self.clear_after:
                    continue
                now = time.monotonic()
                cleared_any = False
                for language, (text, updated_at) in list(self.current.items()):
                    if text and now - updated_at >= self.clear_after:
                        self.current[language] = ("", updated_at)
                        await self._broadcast(language, "")
                        cleared_any = True
                if cleared_any and self.clients.get("all"):
                    all_dict = {lang: self.current.get(lang, ("", 0.0))[0] for lang in self.target_order}
                    message = json.dumps({"all": all_dict}, ensure_ascii=False)
                    await self._broadcast_raw("all", message)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        if self.clear_task:
            self.clear_task.cancel()
            await asyncio.gather(self.clear_task, return_exceptions=True)
            self.clear_task = None
        for clients in self.clients.values():
            await asyncio.gather(
                *(client.close(1001) for client in list(clients)),
                return_exceptions=True,
            )
            clients.clear()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None


class OverlayService:
    """Keep Browser Source URLs alive independently from capture/inference sessions."""

    def __init__(
        self,
        config: OverlayConfig,
        targets: list[TargetConfig],
        html_path: str | Path,
        clear_after: int = 8,
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.targets = list(targets)
        self.html_path = Path(html_path)
        self.clear_after = clear_after
        self.on_status = on_status or (lambda _level, _message: None)
        self._thread: __import__("threading").Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._server: OverlayServer | None = None
        self._ready = __import__("threading").Event()
        self._startup_error: Exception | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._server is not None)

    @property
    def fingerprint(self) -> tuple:
        return (
            self.config.host,
            self.config.port,
            self.config.font_family,
            self.config.font_size,
            self.config.text_color,
            self.config.outline_color,
            self.config.theme,
            tuple(target.language for target in self.targets),
            self.clear_after,
        )

    def url(self, language: str) -> str:
        return overlay_url(self.config, language)

    def preview_url(self, language: str) -> str:
        return overlay_url(self.config, language, preview="1", text="LumaCaption siap tampil di OBS")

    def start(self, timeout: float = 4.0) -> None:
        if self.running:
            return
        import threading

        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="overlay-service",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Overlay server startup timed out")
        if self._startup_error:
            raise self._startup_error

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            self.on_status("error", str(exc))
        finally:
            self._ready.set()
            self._loop = None
            self._stop_event = None
            self._server = None

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        server = OverlayServer(
            self.config,
            self.targets,
            self.html_path,
            self.clear_after,
            self.on_status,
        )
        self._server = server
        try:
            await server.start()
            self._ready.set()
            await self._stop_event.wait()
        finally:
            await server.stop()

    def stop(self, timeout: float = 3.0) -> None:
        loop, stop_event = self._loop, self._stop_event
        if loop and loop.is_running() and stop_event:
            loop.call_soon_threadsafe(stop_event.set)
        thread = self._thread
        if thread and thread is not __import__("threading").current_thread():
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self._thread = None

    def publish_now(self, translations: dict[str, str], timeout: float = 2.0) -> None:
        loop, server = self._loop, self._server
        if not self.running or loop is None or server is None:
            raise RuntimeError("Overlay server is not running")
        future = asyncio.run_coroutine_threadsafe(server.publish(translations), loop)
        future.result(timeout)

    async def publish(self, translations: dict[str, str]) -> None:
        loop, server = self._loop, self._server
        if not self.running or loop is None or server is None:
            raise RuntimeError("Overlay server is not running")
        future = asyncio.run_coroutine_threadsafe(server.publish(translations), loop)
        await asyncio.wrap_future(future)

    async def clear(self) -> None:
        await self.publish({target.language: "" for target in self.targets})


class OverlayPublisher:
    """Pipeline output adapter that never owns persistent server lifecycle."""

    def __init__(self, service: OverlayService) -> None:
        self.service = service

    async def start(self) -> None:
        if not self.service.running:
            await asyncio.to_thread(self.service.start)

    async def publish(self, translations: dict[str, str]) -> None:
        await self.service.publish(translations)

    async def stop(self) -> None:
        if self.service.running:
            await self.service.clear()

