from __future__ import annotations

import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
import sys
import tkinter as tk
from tkinter import messagebox

class _NullStream:
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass
    def isatty(self):
        return False

if getattr(sys, "stdout", None) is None:
    sys.stdout = _NullStream()
if getattr(sys, "stderr", None) is None:
    sys.stderr = _NullStream()

from lumacaption.config import AppConfig, ConfigStore, backup_file
from lumacaption.ui.control_panel import ControlPanel


def app_directory() -> Path:
    """Return the base directory for runtime data (models, config, output)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def bundled_directory() -> Path:
    """Return the directory where read-only bundled assets are stored."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        if (base / "lumacaption" / "assets").is_dir():
            return base / "lumacaption"
        return base
    return Path(__file__).resolve().parent


def _find_asset_source(bundled: Path, *rel_candidates: str) -> Path | None:
    search_dirs = [bundled]
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        for extra in (meipass, meipass / "lumacaption", Path(sys.executable).parent):
            if extra not in search_dirs:
                search_dirs.append(extra)
    for d in search_dirs:
        for rel in rel_candidates:
            candidate = d / rel
            if candidate.is_file():
                return candidate
    return None


def ensure_workspace(directory: Path) -> list[str]:
    """Initialize folders and template files on first launch next to the executable."""
    bundled = bundled_directory()
    warnings = []
    for folder in ("models", "output", "logs"):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    asset_specs = (
        (["assets/silero_vad.onnx", "models/silero_vad.onnx"], directory / "models" / "silero_vad.onnx"),
        (["output/overlay.html", "lumacaption/output/overlay.html"], directory / "output" / "overlay.html"),
        (["assets/vocabulary.json", "vocabulary.json"], directory / "vocabulary.json"),
    )
    for candidates, target in asset_specs:
        source = _find_asset_source(bundled, *candidates)
        if source is None:
            if target.is_file():
                continue
            raise FileNotFoundError(f"Aset bawaan hilang: {target.name}. Instal ulang aplikasi.")
        if not target.exists():
            # Exclusive creation protects files another instance just created.
            try:
                with target.open("xb") as out, source.open("rb") as data:
                    shutil.copyfileobj(data, out)
            except FileExistsError:
                pass
        elif target.name == "overlay.html" and target.read_bytes() != source.read_bytes():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest == "2a73c293bb357f4fbfb6020a52d1bb0f46de27aca8dbbdd41f661d5e20548a5f1":
                backup = backup_file(target)
                temporary = target.with_suffix(".html.tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
                warnings.append(f"Renderer diperbarui; backup {backup.name}")
            else:
                warnings.append("Renderer custom dipertahankan; profil baru memerlukan renderer terbaru. Pulihkan melalui tab OBS jika perlu.")
    config_file = directory / "config.json"
    if not config_file.exists():
        ConfigStore(config_file).save(AppConfig())
    try:
        from lumacaption.model_manager import detect_and_link_models
        adopted = detect_and_link_models(directory)
        if adopted:
            names = ", ".join(m["key"] for m in adopted)
            warnings.append(f"Model AI terdeteksi & dipulihkan dari instalasi sebelumnya: {names}")
    except Exception as exc:
        logging.getLogger("lumacaption").warning(f"Gagal deteksi model otomatis: {exc}")
    return warnings


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        directory = app_directory()
        warnings = ensure_workspace(directory)
        handler = RotatingFileHandler(directory / "logs" / "diagnostic.log", maxBytes=1_048_576, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger("lumacaption").addHandler(handler)
        store = ConfigStore(directory / "config.json")
        config, warning = store.load()
        panel = ControlPanel(root, store, config, directory)
        from lumacaption.pipeline import PipelineEvent
        for text in warnings + ([warning] if warning else []):
            panel.events.put(PipelineEvent("warning", text))

        def on_close() -> None:
            try:
                panel.close()
            except Exception:
                pass
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.deiconify()
        root.mainloop()
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("KizCaption startup error", str(exc), parent=root)
        root.destroy()
        raise


if __name__ == "__main__":
    main()
