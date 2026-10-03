from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parents[1]))
from lumacaption.config import AppConfig, OverlayConfig, TargetConfig
from lumacaption.output.overlay_server import OverlayService

def main():
    app_dir = Path(__file__).parents[1]
    cfg = OverlayConfig(
        host="127.0.0.1", port=8769, font_family="Segoe UI", font_size=48,
        text_color="#FFFFFF", outline_color="#806CFF", theme="vtuber",
    )
    targets = [
        TargetConfig("Japanese"),
        TargetConfig("English"),
        TargetConfig("Chinese (Simplified)"),
    ]
    srv = OverlayService(cfg, targets, app_dir / "output" / "overlay.html", 30, lambda k, m: None)
    srv.start()
    print("Overlay server running on 127.0.0.1:8769", flush=True)
    try:
        for _ in range(60):
            srv.publish_now({
                "Japanese": "お腹空きすぎたからちょっと",
                "English": "Thanks for paying attention.",
                "Chinese (Simplified)": "謝謝你這麼關心",
            })
            time.sleep(2)
    finally:
        srv.stop()

if __name__ == "__main__":
    main()
