from __future__ import annotations

import os
from pathlib import Path
import sys
import threading

_lock = threading.Lock()
_dll_handles: list = []


def configure_cuda_runtime() -> None:
    """Expose optional NVIDIA wheels to this process, never the system PATH."""
    if sys.platform != "win32":
        return
    with _lock:
        if _dll_handles:
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

        for root in candidates:
            if not root.is_dir():
                continue
            for package in ("cublas", "cudnn", "cuda_nvrtc"):
                for sub in ("bin", "lib", ""):
                    directory = (root / package / sub) if sub else (root / package)
                    if directory.is_dir():
                        try:
                            _dll_handles.append(os.add_dll_directory(str(directory)))
                        except (OSError, ValueError):
                            pass
                        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")

        # Limit CTranslate2 CUB caching allocator to 200 MiB cache max (prevents GPU memory hoarding)
        if "CT2_CUDA_CACHING_ALLOCATOR_CONFIG" not in os.environ:
            os.environ["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = "4,3,12,209715200"
