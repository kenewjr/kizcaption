from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tkinter as tk
from tkinter import messagebox

# Ensure src and lumacaption packages are on sys.path
_current = Path(__file__).resolve().parent
_src = _current.parent
_repo = _src.parent
for _p in (str(_repo), str(_src), str(_current)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import AppConfig, ConfigStore
from ui.control_panel import ControlPanel


def app_directory() -> Path:
    """Return the base directory for runtime data (models, config, output)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def bundled_directory() -> Path:
    """Return the directory where read-only bundled assets are stored."""
    if getattr(sys, "frozen", False):
        internal = Path(sys.executable).resolve().parent / "_internal"
        if internal.is_dir():
            return internal
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def ensure_workspace(directory: Path) -> None:
    """Initialize folders and template files on first launch next to the executable."""
    bundled = bundled_directory()

    # 1. Models directory & silero_vad.onnx
    models_dir = directory / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_vad = models_dir / "silero_vad.onnx"
    if not target_vad.is_file():
        source_vad = bundled / "models" / "silero_vad.onnx"
        if source_vad.is_file():
            shutil.copy2(source_vad, target_vad)

    # 2. Output directory & overlay.html
    output_dir = directory / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_overlay = output_dir / "overlay.html"
    if not target_overlay.is_file():
        source_overlay = bundled / "output" / "overlay.html"
        if source_overlay.is_file():
            shutil.copy2(source_overlay, target_overlay)

    # 3. Logs directory
    logs_dir = directory / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 4. Config.json
    config_file = directory / "config.json"
    if not config_file.is_file():
        source_example = bundled / "config.example.json"
        if source_example.is_file():
            shutil.copy2(source_example, config_file)
        else:
            ConfigStore(config_file).save(AppConfig())


def main() -> None:
    directory = app_directory()
    ensure_workspace(directory)

    store = ConfigStore(directory / "config.json")
    config, warning = store.load()
    root = tk.Tk()
    try:
        panel = ControlPanel(root, store, config, directory)
        if warning:
            from pipeline import PipelineEvent
            panel.events.put(PipelineEvent("warning", warning))
        root.mainloop()
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("LumaCaption startup error", str(exc))
        raise


if __name__ == "__main__":
    main()
