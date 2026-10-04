from __future__ import annotations

import os
from pathlib import Path
import sys
import threading

_lock = threading.Lock()
_dll_handles: list = []
_configured = False


def configure_cuda_runtime() -> None:
    """Expose optional NVIDIA wheels to this process, never the system PATH."""
    global _configured
    if sys.platform != "win32":
        return
    with _lock:
        if _configured:
            return
        candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).resolve().parent
            candidates.extend([
                base / "_internal" / "nvidia",
                base / "nvidia",
                base / "_internal",
                base,
            ])
        else:
            candidates.extend([
                Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
                Path(sys.prefix) / "lib" / "site-packages" / "nvidia",
            ])

        def _add_dir(d: Path) -> None:
            if not d.is_dir():
                return
            try:
                _dll_handles.append(os.add_dll_directory(str(d)))
            except (OSError, ValueError):
                pass
            d_str = str(d)
            cur = os.environ.get("PATH", "")
            if d_str not in cur.split(os.pathsep):
                os.environ["PATH"] = d_str + os.pathsep + cur

        for root in candidates:
            if not root.is_dir():
                continue
            # Add root itself if it contains DLLs directly or is a known bundle dir
            _add_dir(root)
            for package in ("cublas", "cudnn", "cuda_nvrtc"):
                for sub in ("bin", "lib", ""):
                    directory = (root / package / sub) if sub else (root / package)
                    _add_dir(directory)

        # Standard CUDA Toolkit paths on Windows if installed
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            _add_dir(Path(cuda_path) / "bin")
        for k, v in os.environ.items():
            if k.startswith("CUDA_PATH_V") and v:
                _add_dir(Path(v) / "bin")
        prog_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        cuda_base = Path(prog_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if cuda_base.is_dir():
            for ver_dir in cuda_base.glob("v*"):
                _add_dir(ver_dir / "bin")

        # Limit CTranslate2 CUB caching allocator to 200 MiB cache max (prevents GPU memory hoarding)
        if "CT2_CUDA_CACHING_ALLOCATOR_CONFIG" not in os.environ:
            os.environ["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = "4,3,12,209715200"
        _configured = True


def is_cuda_available() -> bool:
    """Return True only if an NVIDIA GPU is detected AND required cuBLAS DLLs are loadable."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False

    if sys.platform == "win32":
        configure_cuda_runtime()
        import ctypes
        loaded = False
        for dll_name in ("cublas64_12.dll", "cublas64_11.dll"):
            try:
                ctypes.CDLL(dll_name)
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            return False
    return True

