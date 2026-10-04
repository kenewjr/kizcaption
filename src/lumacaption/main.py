from __future__ import annotations

from datetime import datetime
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


KNOWN_STOCK_OVERLAY_HASHES = {
    # Earlier stock overlay builds (64 hex characters)
    "2a73c293bb357f4fbfb6020a52d1bb0f46de27aca8dbbdd41f661d5e20548a5f",
    "b6e45454e698695e9103eace0f2f68e90646259787ecd01e55003504c20a856d",
}


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
            if digest in KNOWN_STOCK_OVERLAY_HASHES:
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


class LevelFilter(logging.Filter):
    """Filter records to match a specific level (and critical for error)."""
    def __init__(self, target_level: int) -> None:
        super().__init__()
        self.target_level = target_level

    def filter(self, record: logging.LogRecord) -> bool:
        if self.target_level == logging.ERROR:
            return record.levelno >= logging.ERROR
        return record.levelno == self.target_level


def setup_logging(directory: Path, timestamp: str | None = None) -> logging.Logger:
    logs_dir = directory / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lumacaption")
    logger.setLevel(logging.DEBUG)

    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    import platform
    from lumacaption import __version__

    banner_lines = [
        "=" * 65,
        f"KIZCAPTION SESSION STARTED (v{__version__})",
        f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}",
        f"Executable: {sys.executable} | App Dir: {directory}",
    ]
    try:
        from lumacaption.model_manager import hardware_info
        hw_str = hardware_info(directory)
        banner_lines.append(f"Hardware:\n{hw_str}")
    except Exception as e:
        banner_lines.append(f"Hardware info query: {e}")
    banner_lines.append("=" * 65)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = f"kizcaption-{timestamp}.log"
    log_path = logs_dir / filename
    header_text = "\n".join(
        f"{now_str} [INFO] [lumacaption] {line}" for line in banner_lines
    ) + "\n"
    try:
        log_path.write_text(header_text, encoding="utf-8")
    except Exception:
        pass

    handler = RotatingFileHandler(log_path, mode="a", maxBytes=10_485_760, backupCount=5, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    logger = None
    try:
        directory = app_directory()
        logger = setup_logging(directory)
        warnings = ensure_workspace(directory)
        store = ConfigStore(directory / "config.json")
        config, warning = store.load()
        panel = ControlPanel(root, store, config, directory)
        from lumacaption.pipeline import PipelineEvent
        for text in warnings + ([warning] if warning else []):
            panel.events.put(PipelineEvent("warning", text))

        def on_close() -> None:
            try:
                panel.close()
            except Exception as exc:
                if logger:
                    logger.warning(f"Error closing panel: {exc}")
            if logger:
                logger.info("=" * 65)
                logger.info("KIZCAPTION SESSION ENDED CLEANLY")
                logger.info("=" * 65)
                for h in list(logger.handlers):
                    try:
                        h.flush()
                        h.close()
                        logger.removeHandler(h)
                    except Exception:
                        pass
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.deiconify()
        root.mainloop()
    except Exception as exc:
        if logger:
            logger.exception(f"KizCaption startup crash: {exc}")
        root.withdraw()
        messagebox.showerror("KizCaption startup error", str(exc), parent=root)
        root.destroy()
        raise


if __name__ == "__main__":
    main()
