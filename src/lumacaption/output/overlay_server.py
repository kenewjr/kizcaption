from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
import http
import inspect
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from lumacaption.config import OverlayConfig, TargetConfig


def overlay_url(config: OverlayConfig, language: str = "", profile: int | str = "", **extra: str) -> str:
    host = f"[{config.host}]" if ":" in config.host else config.host
    params: dict[str, str | int] = {
        "font": config.font_family,
        "size": config.font_size,
        "color": config.text_color,
        "outline": config.outline_color,
        "theme": config.theme,
    }
    if profile:
        params["profile"] = profile
    elif language:
        params["lang"] = language
    else:
        params["profile"] = "all"
    params.update(extra)
    return f"http://{host}:{config.port}/overlay?{urlencode(params)}"


def static_overlay_url(config: OverlayConfig) -> str:
    host = f"[{config.host}]" if ":" in config.host else config.host
    return f"http://{host}:{config.port}/overlay.html"


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
        for idx, target in enumerate(targets, 1):
            if target.profile == 0:
                target.profile = idx
        self.targets = list(targets)
        self.languages = {target.language for target in targets}
        self.target_order = [target.language for target in targets]
        self.target_by_slot = {target.profile: target.language for target in targets}
        self.slot_by_target = {target.language: target.profile for target in targets}
        self.html_path = Path(html_path)
        self._html_fallback = self.html_path.read_text(encoding="utf-8")
        self.clear_after = clear_after
        self.on_status = on_status or (lambda _level, _message: None)
        self.clients: dict[str, set] = defaultdict(set)
        self.current: dict[str, tuple[str, float]] = {}
        self.server = None
        self.clear_task: asyncio.Task | None = None
        self.last_publish_time: float = 0.0
        self.last_publish_clients: int = 0

    @property
    def active_client_count(self) -> int:
        return sum(len(c) for c in self.clients.values())

    def get_diagnostics(self) -> dict:
        return {
            "running": self.server is not None,
            "port": self.config.port,
            "host": self.config.host,
            "clients": self.active_client_count,
            "last_publish_time": self.last_publish_time,
            "last_publish_clients": self.last_publish_clients,
        }

    @property
    def html(self) -> str:
        try:
            return self.html_path.read_text(encoding="utf-8")
        except Exception:
            return self._html_fallback

    @html.setter
    def html(self, content: str) -> None:
        self._html_fallback = content

    def url(self, language: str = "", profile: int | str = "") -> str:
        return overlay_url(self.config, language=language, profile=profile)

    def preview_url(self, language: str = "", profile: int | str = "", text: str = "KizCaption siap tampil di OBS") -> str:
        return overlay_url(self.config, language=language, profile=profile, preview="1", text=text)

    async def update_styles(self, config: OverlayConfig) -> None:
        self.config = config
        styles_dict = {
            slot: self.config.profiles[slot - 1].to_dict()
            for slot in (1, 2, 3)
            if slot - 1 < len(self.config.profiles)
        }
        msg = json.dumps({
            "type": "style_update",
            "styles": styles_dict,
            "anchor": self.config.anchor,
            "gap": self.config.gap,
            "order": self.config.order,
        }, ensure_ascii=False)
        for group in list(self.clients.keys()):
            await self._broadcast_raw(group, msg)

    async def start(self) -> None:
        if self.server is not None:
            return
        from websockets.asyncio.server import serve

        def process_request(connection, request):
            parts = urlsplit(request.path)
            if parts.path in ("/", "/overlay", "/overlay.html"):
                qs = parse_qs(parts.query)
                lang = qs.get("lang", [""])[0]
                prof = qs.get("profile", [""])[0]
                if lang and lang not in self.languages and lang not in ("all", "All", "*"):
                    return connection.respond(http.HTTPStatus.NOT_FOUND, "Unknown language\n")
                if prof and prof not in ("1", "2", "3", "all", "All", "*"):
                    return connection.respond(http.HTTPStatus.NOT_FOUND, "Unknown profile\n")
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
        qs = parse_qs(parts.query)
        language = qs.get("lang", [""])[0]
        profile = qs.get("profile", [""])[0]

        if parts.path != "/ws":
            await websocket.close(1008, "Invalid endpoint")
            return

        has_profile = bool(profile)
        has_lang = bool(language)
        if not has_profile and not has_lang:
            has_lang = True
            language = "all"

        if has_profile:
            is_all = profile in ("all", "All", "*")
            slot = int(profile) if profile in ("1", "2", "3") else None
            if not is_all and slot is None:
                await websocket.close(1008, "Unknown profile")
                return
            group = "profile_all" if is_all else f"profile_{slot}"
        else:
            is_all = language in ("all", "All", "*")
            if not is_all and language not in self.languages:
                await websocket.close(1008, "Unknown language")
                return
            group = "all" if is_all else language

        self.clients[group].add(websocket)
        self.on_status("clients_changed", str(self.active_client_count))
        try:
            if has_profile:
                styles_dict = {
                    s: self.config.profiles[s - 1].to_dict()
                    for s in (1, 2, 3)
                    if s - 1 < len(self.config.profiles)
                }
                if is_all:
                    all_data = {lang: self.current.get(lang, ("", 0.0))[0] for lang in self.target_order}
                    slots_data = {
                        s: self.current.get(self.target_by_slot.get(s, ""), ("", 0.0))[0]
                        for s in (1, 2, 3)
                    }
                    msg = json.dumps({
                        "all": all_data,
                        "slots": slots_data,
                        "styles": styles_dict,
                        "anchor": self.config.anchor,
                        "gap": self.config.gap,
                        "order": self.config.order,
                    }, ensure_ascii=False)
                    await websocket.send(msg)
                else:
                    lang_name = self.target_by_slot.get(slot, "")
                    text = self.current.get(lang_name, ("", 0.0))[0] if lang_name else ""
                    msg = json.dumps({
                        "slot": slot,
                        "lang": lang_name,
                        "text": text,
                        "style": styles_dict.get(slot, {}),
                    }, ensure_ascii=False)
                    await websocket.send(msg)
            else:
                if is_all:
                    all_data = {
                        lang: self.current.get(lang, ("", 0.0))[0]
                        for lang in self.target_order
                    }
                    await websocket.send(json.dumps({"all": all_data}, ensure_ascii=False))
                else:
                    text = self.current.get(language, ("", 0.0))[0]
                    await websocket.send(json.dumps({"lang": language, "text": text}, ensure_ascii=False))
            await websocket.wait_closed()
        finally:
            self.clients[group].discard(websocket)
            self.on_status("clients_changed", str(self.active_client_count))

    async def publish(self, translations: dict[str, str]) -> None:
        now = time.monotonic()
        total_delivered = 0
        has_content = any(bool(text.strip()) for text in translations.values())
        for language, text in translations.items():
            if language in self.languages:
                self.current[language] = (text, now)
                # 1. Broadcast to legacy language listeners
                total_delivered += await self._broadcast(language, text)
                # 2. Broadcast to slot profile listeners
                slot = self.slot_by_target.get(language)
                if slot:
                    slot_group = f"profile_{slot}"
                    if self.clients.get(slot_group):
                        msg = json.dumps({
                            "slot": slot,
                            "lang": language,
                            "text": text,
                        }, ensure_ascii=False)
                        total_delivered += await self._broadcast_raw(slot_group, msg)

        # 3. Broadcast to all listeners
        if self.clients.get("all"):
            all_dict = {
                lang: translations.get(lang, self.current.get(lang, ("", 0.0))[0])
                for lang in self.target_order
            }
            message = json.dumps({"all": all_dict}, ensure_ascii=False)
            total_delivered += await self._broadcast_raw("all", message)

        if self.clients.get("profile_all"):
            all_dict = {
                lang: translations.get(lang, self.current.get(lang, ("", 0.0))[0])
                for lang in self.target_order
            }
            slots_dict = {
                s: translations.get(self.target_by_slot.get(s, ""), self.current.get(self.target_by_slot.get(s, ""), ("", 0.0))[0])
                for s in (1, 2, 3)
            }
            message = json.dumps({
                "all": all_dict,
                "slots": slots_dict,
            }, ensure_ascii=False)
            total_delivered += await self._broadcast_raw("profile_all", message)

        if has_content:
            self.last_publish_time = time.time()
            self.last_publish_clients = total_delivered

    async def _broadcast(self, language: str, text: str) -> int:
        message = json.dumps({"lang": language, "text": text}, ensure_ascii=False)
        return await self._broadcast_raw(language, message)

    async def _broadcast_raw(self, group: str, message: str) -> int:
        clients = list(self.clients[group])
        delivered = 0
        if clients:
            async def _safe_send(ws):
                try:
                    await asyncio.wait_for(ws.send(message), timeout=1.5)
                except Exception:
                    try:
                        res = ws.close()
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        pass
                    raise

            results = await asyncio.gather(
                *(_safe_send(client) for client in clients),
                return_exceptions=True,
            )
            for client, result in zip(clients, results, strict=True):
                if isinstance(result, Exception):
                    self.clients[group].discard(client)
                else:
                    delivered += 1
        return delivered

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
                        slot = self.slot_by_target.get(language)
                        if slot and self.clients.get(f"profile_{slot}"):
                            await self._broadcast_raw(f"profile_{slot}", json.dumps({"slot": slot, "lang": language, "text": ""}))
                        cleared_any = True
                if cleared_any:
                    if self.clients.get("all"):
                        all_dict = {lang: self.current.get(lang, ("", 0.0))[0] for lang in self.target_order}
                        message = json.dumps({"all": all_dict}, ensure_ascii=False)
                        await self._broadcast_raw("all", message)
                    if self.clients.get("profile_all"):
                        all_dict = {lang: self.current.get(lang, ("", 0.0))[0] for lang in self.target_order}
                        slots_dict = {s: self.current.get(self.target_by_slot.get(s, ""), ("", 0.0))[0] for s in (1, 2, 3)}
                        message = json.dumps({"all": all_dict, "slots": slots_dict}, ensure_ascii=False)
                        await self._broadcast_raw("profile_all", message)
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
            tuple(target.language for target in self.targets),
            tuple(target.profile for target in self.targets),
        )

    def static_url(self) -> str:
        return static_overlay_url(self.config)

    def url(self, language: str = "", profile: int | str = "") -> str:
        return overlay_url(self.config, language=language, profile=profile)

    def preview_url(self, language: str = "", profile: int | str = "") -> str:
        return overlay_url(self.config, language=language, profile=profile, preview="1", text="KizCaption siap tampil di OBS")

    def get_diagnostics(self) -> dict:
        server = self._server
        if not self.running or server is None:
            return {
                "running": False,
                "port": self.config.port,
                "host": self.config.host,
                "clients": 0,
                "last_publish_time": 0.0,
                "last_publish_clients": 0,
            }
        return server.get_diagnostics()

    def update_styles(self, config: OverlayConfig) -> None:
        self.config = config
        if self._loop and self._server and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._server.update_styles(config), self._loop)

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
