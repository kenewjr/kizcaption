"""Model cache, bounded downloads, and honest resource metadata."""
from __future__ import annotations
from dataclasses import dataclass
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

NLLB_ID = 'mijuanlo/nllb-200-distilled-600M-ct2-int8'
NLLB_1_3B_ID = 'mijuanlo/nllb-200-distilled-1.3B-int8-ct2'
WHISPER_FILES = ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.*')
NLLB_FILES = ('model.bin', 'config.json', 'shared_vocabulary.json', 'sentencepiece.bpe.model')

@dataclass(frozen=True)
class ModelInfo:
    key: str
    repo: str
    revision: str
    files: tuple[str, ...]
    disk_mib: float
    load: str

CATALOG = {
    key: ModelInfo(key, f'Systran/faster-whisper-{key}', 'main', WHISPER_FILES, size, load)
    for key, size, load in (('tiny', 75, 'Ringan'), ('base', 145, 'Ringan'), ('small', 464, 'Sedang'), ('medium', 1460, 'Berat'))
}
CATALOG['large-v3-turbo'] = ModelInfo('large-v3-turbo', 'mobiuslabsgmbh/faster-whisper-large-v3-turbo', 'main', WHISPER_FILES, 1547, 'Berat; decoder lebih pendek')
CATALOG['distil-large-v3'] = ModelInfo('distil-large-v3', 'Systran/faster-distil-whisper-large-v3', 'c3058b475261292e64a0412df1d2681c06260fab', WHISPER_FILES, 1445, 'Cepat; khusus bahasa Inggris')
CATALOG['large-v3'] = ModelInfo('large-v3', 'Systran/faster-whisper-large-v3', 'main', WHISPER_FILES, 2900, 'Sangat Berat; akurasi tertinggi')
CATALOG['nllb'] = ModelInfo('nllb', NLLB_ID, '16bc5ff0482f9f1c0d35bdef950721ce58640789', NLLB_FILES, 604, 'Sedang; meningkat dengan target/beam')
CATALOG['nllb-1.3b'] = ModelInfo('nllb-1.3b', NLLB_1_3B_ID, '6eee5eda03ff1441d2a6117d34a02e44504ce321', NLLB_FILES, 1404, 'Berat; akurasi translasi lebih tinggi')
CATALOG['whisper-small-id'] = ModelInfo('whisper-small-id', 'ammaraldirawi/faster-whisper-small-id-int8', 'main', WHISPER_FILES, 480, 'Sedang; Akurasi slang/aksen Indonesia')
CATALOG['whisper-medium-id'] = ModelInfo('whisper-medium-id', 'cahya/faster-whisper-medium-id', 'main', WHISPER_FILES, 1460, 'Berat; Akurasi Indonesia resmi Cahya')
CATALOG['dtln'] = ModelInfo('dtln', 'niobures/DTLN', 'main', ('model_1.onnx', 'model_2.onnx'), 3.96, 'Ringan; Denoise neural network CPU')
CATALOG['silero'] = ModelInfo('silero', '', '', ('silero_vad.onnx',), 2.22, 'Ringan; CPU 1 thread')
_locks = {key: threading.Lock() for key in CATALOG}


def complete(path: Path, patterns: tuple[str, ...]) -> bool:
    return path.is_dir() and all(any(p.is_file() and p.stat().st_size > 0 for p in path.glob(pattern)) for pattern in patterns)


def inspect_model(key: str, cache: Path) -> tuple[str, Path | None]:
    info = CATALOG[key]
    if key in ('silero', 'dtln'):
        return ('Tersedia lokal' if complete(cache, info.files) else 'Belum ada'), cache
    repo = cache / ('models--' + info.repo.replace('/', '--'))
    revision = info.revision
    ref = repo / 'refs' / revision
    if ref.is_file():
        revision = ref.read_text('utf-8').strip()
    # Do not accept path traversal from a malformed ref file.
    if not revision or any(c not in '0123456789abcdef' for c in revision):
        return ('Belum lengkap' if repo.exists() else 'Belum ada'), None
    snapshot = repo / 'snapshots' / revision
    if complete(snapshot, info.files):
        return 'Tersedia lokal', snapshot
    return ('Belum lengkap' if repo.exists() else 'Belum ada'), None


def cache_for(directory: Path, key: str) -> Path:
    if key == 'silero':
        return directory / 'models'
    if key == 'dtln':
        return directory / 'models' / 'dtln'
    if key in ('nllb', 'nllb-1.3b'):
        return directory / 'models' / 'nllb-cache'
    return directory / 'models' / 'whisper'


@dataclass
class DownloadProgress:
    model: str
    state: str
    current: int = 0
    total: int | None = None
    downloaded_bytes: int = 0
    total_bytes: int = 0
    fraction: float = 0.0
    percent: float = 0.0
    speed_bps: float = 0.0
    speed_str: str = ""
    size_str: str = ""
    eta_str: str = ""
    detail: str = ""
    filename: str = ""

    def __getitem__(self, key):
        return getattr(self, key, None)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


def ensure_model(key: str, cache: Path, on_status=None) -> Path:
    notify = on_status or (lambda _data: None)
    info = CATALOG[key]

    def emit(state, **data):
        tot = data.get("total") or int(info.disk_mib * 1024 * 1024)
        cur = data.get("current", 0)
        frac = data.get("fraction", (cur / tot) if tot > 0 else (1.0 if state == "Tersedia lokal" else 0.0))
        prog = DownloadProgress(
            model=key,
            state=state,
            current=cur,
            total=tot,
            downloaded_bytes=cur,
            total_bytes=tot,
            fraction=frac,
            percent=frac * 100.0,
            speed_bps=data.get("speed_bps", 0.0),
            speed_str=data.get("speed_str", ""),
            size_str=data.get("size_str", ""),
            eta_str=data.get("eta_str", ""),
            detail=data.get("detail", ""),
            filename=data.get("filename", ""),
        )
        notify(prog)

    emit('Memeriksa')
    with _locks[key]:
        state, path = inspect_model(key, cache)
        if state == 'Tersedia lokal':
            emit(state)
            return path
        if key == 'silero':
            raise FileNotFoundError('Aset Silero tidak ada; pulihkan dari paket aplikasi')
        if key == 'dtln':
            cache.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import hf_hub_download
            emit('Mengunduh DTLN (1/2)')
            p1 = Path(hf_hub_download('niobures/DTLN', 'models/DTLN/onnx/model_1.onnx'))
            shutil.copyfile(p1, cache / 'model_1.onnx')
            emit('Mengunduh DTLN (2/2)')
            p2 = Path(hf_hub_download('niobures/DTLN', 'models/DTLN/onnx/model_2.onnx'))
            shutil.copyfile(p2, cache / 'model_2.onnx')
            emit('Tersedia lokal')
            return cache
        cache.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cache).free
        if free < info.disk_mib * 1024**2 * 1.2:
            raise OSError(f'Ruang disk kurang; sediakan sekitar {info.disk_mib * 1.2:.0f} MiB sementara')
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        class Progress(tqdm):
            def __init__(self, *args, **kwargs):
                self.last_sent = 0.0
                self.start_t = time.monotonic()
                self.last_t = self.start_t
                self.last_n = 0
                self.speed_bps = 0.0
                kwargs['disable'] = False
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                self._report()

            def display(self, *args, **kwargs):
                self._report()

            def _report(self):
                now = time.monotonic()
                dt = now - self.last_t
                if dt >= 0.3:
                    dn = self.n - self.last_n
                    instant = dn / dt if dt > 0 else 0
                    self.speed_bps = 0.7 * self.speed_bps + 0.3 * instant if self.speed_bps > 0 else instant
                    self.last_t = now
                    self.last_n = self.n

                if now - self.last_sent < 0.1 and self.n != self.total:
                    return
                self.last_sent = now

                tot = self.total or int(info.disk_mib * 1024 * 1024)
                frac = (self.n / tot) if (tot and tot > 0) else 0.0
                frac = max(0.0, min(1.0, frac))

                if self.speed_bps >= 1024 * 1024:
                    spd_str = f"{self.speed_bps / (1024 * 1024):.1f} MB/s"
                elif self.speed_bps >= 1024:
                    spd_str = f"{self.speed_bps / 1024:.0f} KB/s"
                elif self.speed_bps > 0:
                    spd_str = f"{self.speed_bps:.0f} B/s"
                else:
                    spd_str = "Menghubungkan…"

                cur_mb = self.n / (1024 * 1024)
                tot_mb = tot / (1024 * 1024)
                size_str = f"{cur_mb:.1f} MB / {tot_mb:.1f} MB"

                eta_str = ""
                if self.speed_bps > 1024 and tot > self.n:
                    rem_sec = int((tot - self.n) / self.speed_bps)
                    if rem_sec < 60:
                        eta_str = f"ETA: {rem_sec}s"
                    else:
                        eta_str = f"ETA: {rem_sec // 60}m {rem_sec % 60}s"

                filename = getattr(self, "desc", "") or ""
                emit(
                    'Mengunduh',
                    current=self.n,
                    total=tot,
                    fraction=frac,
                    speed_bps=self.speed_bps,
                    speed_str=spd_str,
                    size_str=size_str,
                    eta_str=eta_str,
                    filename=filename,
                    detail=filename,
                )

        emit('Mengunduh', detail=f'Perkiraan disk {info.disk_mib:g} MiB')
        try:
            path = Path(snapshot_download(
                info.repo,
                revision=info.revision,
                cache_dir=str(cache),
                allow_patterns=list(info.files) + ['preprocessor_config.json'],
                tqdm_class=Progress,
                max_workers=2,
            ))
            emit('Memverifikasi')
            if not complete(path, info.files):
                raise RuntimeError(f'Cache {key} belum lengkap')
            emit('Tersedia lokal')
            return path
        except Exception as exc:
            emit('Gagal', detail=str(exc))
            raise


LANGUAGE_SUITABILITY: dict[str, str] = {
    "small": "Indonesia, Jawa, Sunda, Melayu & Inggris santai (⭐ Rekomendasi Utama Streamer).",
    "base": "Indonesia & Inggris formal. Ringan, cocok untuk laptop/CPU hemat daya.",
    "tiny": "Inggris atau Indonesia kalimat sederhana (Akurasi dasar, paling cepat & ringan).",
    "medium": "Indonesia, Jepang, Korea, Mandarin, Arab & kalimat panjang kompleks.",
    "large-v3-turbo": "Multibahasa Global 99+ bahasa (Indonesia, CJK, Arab, Eropa). Cepat & presisi tinggi.",
    "distil-large-v3": "⚠️ Khusus Bahasa Inggris (English Only). Dilarang untuk bahasa Indonesia!",
    "large-v3": "Multibahasa Akurasi Studio 99+ bahasa dunia (Sangat berat, disarankan VRAM >= 8 GB).",
    "whisper-small-id": "🇮🇩 Khusus Bahasa Indonesia (Ammar/Cahya INT8) — Akurasi tinggi percakapan gaul & slang streamer.",
    "whisper-medium-id": "🇮🇩 Khusus Bahasa Indonesia (Cahya Wirawan) — Akurasi kosakata daerah & nama lokal Indonesia tertinggi.",
    "dtln": "🛡️ Peredam Bising Neural Network — Menghilangkan suara ketikan keyboard & desah kipas PC secara real-time.",
    "nllb": "Translasi 200 bahasa (Inggris, Jepang, Korea, Mandarin, Arab, Spanyol, dll.). Cepat & hemat.",
    "nllb-1.3b": "Translasi 200 bahasa tata bahasa presisi tinggi untuk kalimat sastra/teknis kompleks.",
    "silero": "Deteksi jeda hening dan suara manusia semua bahasa (Silero VAD v5).",
}

LANGUAGE_SUITABILITY_EN: dict[str, str] = {
    "small": "Indonesian, Javanese, Sundanese, Malay & casual English (⭐ Top Streamer Recommendation).",
    "base": "Indonesian & formal English. Lightweight, ideal for laptops & low-power CPUs.",
    "tiny": "English or simple Indonesian sentences (Basic accuracy, fastest & lightest).",
    "medium": "Indonesian, Japanese, Korean, Chinese, Arabic & long complex sentences.",
    "large-v3-turbo": "Global Multilingual 99+ languages (Indonesian, CJK, Arabic, European). Fast & high accuracy.",
    "distil-large-v3": "⚠️ English Only. Do not use for Indonesian!",
    "large-v3": "Studio-Grade Multilingual 99+ languages (Very heavy, VRAM >= 8 GB recommended).",
    "whisper-small-id": "🇮🇩 Indonesian Fine-Tuned (Ammar/Cahya INT8) — High accuracy for streamer slang & informal speech.",
    "whisper-medium-id": "🇮🇩 Indonesian Fine-Tuned (Cahya Wirawan) — Highest accuracy for local Indonesian vocabulary & dialects.",
    "dtln": "🛡️ Neural Network Noise Suppressor — Removes keyboard clicks & PC fan noise in real time.",
    "nllb": "Translates 200 languages (English, Japanese, Korean, Chinese, Arabic, Spanish, etc.). Fast & light.",
    "nllb-1.3b": "High-accuracy 200 language translation for complex literary/technical sentences.",
    "silero": "Voice activity and human speech pause detector for all languages (Silero VAD v5).",
}


def model_resource_table_rows(lang: str = "id") -> list[dict[str, str]]:
    """Return formatted resource estimation table rows for all models."""
    order = [
        "silero",
        "dtln",
        "tiny",
        "base",
        "small",
        "whisper-small-id",
        "medium",
        "whisper-medium-id",
        "large-v3-turbo",
        "large-v3",
        "nllb",
        "nllb-1.3b",
    ]
    notes_en = {
        "silero": "Silero VAD v5 ONNX voice pause detector",
        "dtln": "Real-time DTLN ONNX neural noise suppressor",
        "tiny": "Lightest, lowest latency, basic accuracy",
        "base": "Fast & lightweight for modest CPUs",
        "small": "Balanced for Indonesian & Javanese",
        "whisper-small-id": "Indonesian slang & accent accuracy (Ammar/Cahya)",
        "medium": "High accuracy, moderate-to-heavy CPU/VRAM",
        "whisper-medium-id": "Highest accuracy for formal Indonesian (Cahya)",
        "large-v3-turbo": "Highest accuracy, optimized for VRAM >= 4 GB",
        "large-v3": "Top multilingual accuracy, heavy VRAM (VRAM >= 8 GB recommended)",
        "distil-large-v3": "Distilled Whisper English-only, fast latency",
        "nllb": "Balanced multilingual batch translation engine",
        "nllb-1.3b": "Higher translation accuracy for multilingual, VRAM >= 4 GB recommended",
    }
    path = Path(__file__).parent / "assets" / "resources.json"
    data = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    models_data = data.get("models", {})
    rows = []
    for key in order:
        m = models_data.get(key)
        if not m:
            continue
        name = m.get("name", key)
        if key == "small":
            name += " [Recommended]" if lang == "en" else " [Rekomendasi]"
        disk_mib = m.get("disk_mib", 0.0)
        disk_str = f"~{disk_mib:.1f} MB" if disk_mib < 10 else f"{disk_mib:.0f} MB"
        ram = m.get("cpu_ram_mb", [0, 0])
        ram_str = f"{ram[0]:,} - {ram[1]:,} MB"
        vram = m.get("cuda_vram_mb", [0, 0])
        if vram == [0, 0]:
            vram_str = "0 MB (always CPU)" if lang == "en" else "0 MB (selalu CPU)"
        else:
            vram_str = f"{vram[0]:,} - {vram[1]:,} MB"
        threads = m.get("threads", [1, 1])
        th_unit = "threads" if lang == "en" else "thread"
        th_str = f"{threads[0]} - {threads[1]} {th_unit}" if threads[0] != threads[1] else f"{threads[0]} {th_unit}"
        note = notes_en.get(key, m.get("note", "")) if lang == "en" else m.get("note", "")
        rows.append({
            "key": key,
            "model": name,
            "disk": disk_str,
            "ram": ram_str,
            "vram": vram_str,
            "threads": th_str,
            "note": note,
        })
    return rows



def resource_details(key: str, device: str, threads: int = 4, beam: int = 3, targets: int = 1, lang: str = "id") -> str:
    info = CATALOG.get(key)
    is_en = (lang == "en")
    if not info:
        return "Unknown model" if is_en else "Model tidak dikenal"
    path = Path(__file__).parent / "assets" / "resources.json"
    data = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    models_data = data.get("models", {})
    measured = models_data.get(key)

    memory_parts = []
    if measured:
        ram_range = measured.get("cpu_ram_mb", [0, 0])
        vram_range = measured.get("cuda_vram_mb", [0, 0])
        note = measured.get("note", "")
        if is_en:
            status_label = "local calibration" if measured.get("status") == "local_measured" else "reference baseline"
            memory_parts.append(f"Estimated RAM: {ram_range[0]}–{ram_range[1]} MiB ({status_label})")
            if device == "cpu":
                memory_parts.append("VRAM: model does not use VRAM in CPU mode")
            else:
                memory_parts.append(f"Estimated VRAM: {vram_range[0]}–{vram_range[1]} MiB")
            if note:
                memory_parts.append(f"Note: {note}")
        else:
            status_label = "kalibrasi lokal" if measured.get("status") == "local_measured" else "acuan referensi"
            memory_parts.append(f"RAM estimasi: {ram_range[0]}–{ram_range[1]} MiB ({status_label})")
            if device == "cpu":
                memory_parts.append("VRAM: model tidak menggunakan VRAM pada mode CPU")
            else:
                memory_parts.append(f"VRAM estimasi: {vram_range[0]}–{vram_range[1]} MiB")
            if note:
                memory_parts.append(f"Catatan: {note}")
    else:
        if is_en:
            memory_parts.append("RAM/VRAM: not locally calibrated")
            if device == "cpu":
                memory_parts.append("Model VRAM not used")
        else:
            memory_parts.append("RAM/VRAM: belum dikalibrasi lokal")
            if device == "cpu":
                memory_parts.append("VRAM model tidak digunakan")

    lang_map = LANGUAGE_SUITABILITY_EN if is_en else LANGUAGE_SUITABILITY
    lang_suit = lang_map.get(key, "")

    if is_en:
        lang_line = f"Language Suitability: {lang_suit}\n" if lang_suit else ""
        mem_str = " · ".join(memory_parts)
        th_str = "1 (single)" if key == "silero" else f"{threads} threads"
        load_map = {
            "Sangat Rendah": "Very Low",
            "Rendah": "Low",
            "Sedang": "Moderate",
            "Sedang; Akurasi slang/aksen Indonesia": "Moderate; Indonesian slang/accent accuracy",
            "Tinggi": "High",
            "Sangat Tinggi": "Very High",
            "Sangat Ringan": "Very Lightweight",
            "Ringan": "Lightweight",
        }
        load_val = load_map.get(info.load, info.load)
        return (
            f"{lang_line}"
            f"{mem_str}\n"
            f"Disk: ~{info.disk_mib:g} MiB · Load: {load_val}\n"
            f"Configuration: CPU max {th_str} · Beam {beam} · Target {targets}\n"
            "Estimates, not guarantees. Process RAM includes application runtime."
        )
    else:
        lang_line = f"Kecocokan Bahasa: {lang_suit}\n" if lang_suit else ""
        mem_str = " · ".join(memory_parts)
        th_str = f"1 (tunggal)" if key == "silero" else f"{threads} thread"
        return (
            f"{lang_line}"
            f"{mem_str}\n"
            f"Disk: ~{info.disk_mib:g} MiB · Beban: {info.load}\n"
            f"Konfigurasi: CPU maks {th_str} · Beam {beam} · Target {targets}\n"
            "Perkiraan, bukan jaminan. RAM proses termasuk runtime aplikasi."
        )


def hardware_info(directory: Path, lang: str = "id") -> str:
    """Return formatted hardware summary for the dashboard panel."""
    cores = os.cpu_count() or 1
    disk_gb = shutil.disk_usage(directory).free / 1024**3
    lines = [
        f"⚡  CPU             : {cores} Logical Cores",
    ]
    is_en = (lang == "en")
    # RAM (Windows)
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram_avail = status.available / 1024**3
            ram_total = status.total / 1024**3
            if is_en:
                lines.append(f"🧠  RAM             : {ram_avail:.1f} GB Available / {ram_total:.1f} GB Total")
            else:
                lines.append(f"🧠  RAM             : {ram_avail:.1f} GB Tersedia / {ram_total:.1f} GB Total")
    # GPU
    smi = shutil.which('nvidia-smi')
    if smi:
        try:
            result = subprocess.run([smi, '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode:
                raise RuntimeError('GPU query gagal')
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 3:
                gpu_name = parts[0]
                vram_total = float(parts[1]) / 1024
                vram_free = float(parts[2]) / 1024
                lines.append(f"🎮  GPU             : {gpu_name}")
                if is_en:
                    lines.append(f"📊  VRAM            : {vram_free:.1f} GB Available / {vram_total:.1f} GB Total")
                else:
                    lines.append(f"📊  VRAM            : {vram_free:.1f} GB Tersedia / {vram_total:.1f} GB Total")
            else:
                lines.append(f"🎮  GPU             : {result.stdout.strip()}")
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError):
            lines.append("🎮  GPU             : Not detected" if is_en else "🎮  GPU             : Tidak terdeteksi")
    else:
        lines.append("🎮  GPU             : Not available (CPU mode)" if is_en else "🎮  GPU             : Tidak tersedia (mode CPU)")
    lines.append(f"💾  {'Free Disk' if is_en else 'Disk bebas'}      : {disk_gb:.1f} GB")
    return '\n'.join(lines)


def get_vram_free_mb() -> float | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        res = subprocess.run(
            [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0].strip()
            return float(first_line)
    except Exception:
        pass
    return None


def evaluate_vram_safety(
    whisper_key: str,
    nllb_model_str: str,
    stt_device: str,
    mt_device: str,
    stt_compute_type: str = "auto",
    mt_compute_type: str = "auto",
    lang: str = "id",
) -> tuple[str, str, str]:
    """
    Evaluasi kecukupan VRAM GPU vs model dan presisi compute yang dipilih.
    Returns: (status, badge_label, detail_message)
    status: 'safe' | 'caution' | 'danger' | 'cpu'
    """
    use_gpu_stt = stt_device in ("cuda", "auto")
    use_gpu_mt = mt_device in ("cuda", "auto")
    is_en = (lang == "en")
    if not use_gpu_stt and not use_gpu_mt:
        if is_en:
            return ("cpu", "🟢 CPU Mode", "Model runs on CPU with zero GPU VRAM usage.")
        return ("cpu", "🟢 Mode CPU", "Model berjalan di CPU tanpa konsumsi VRAM GPU.")

    vram_free = get_vram_free_mb()
    if vram_free is None:
        if is_en:
            return ("caution", "🟡 GPU Not Detected", "VRAM status could not be read via nvidia-smi. Ensure NVIDIA drivers are active.")
        return ("caution", "🟡 GPU Tidak Terdeteksi", "Status VRAM tidak dapat dibaca via nvidia-smi. Pastikan driver NVIDIA aktif.")

    path = Path(__file__).parent / "assets" / "resources.json"
    data = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    models_data = data.get("models", {})

    def _model_vram(key: str, compute_type: str, default: list[int]) -> float:
        m_info = models_data.get(key, {})
        limits = m_info.get("cuda_vram_mb", default)
        low, high = float(limits[0]), float(limits[1])
        if compute_type in ("int8_float16", "int8_float32", "int8"):
            return low
        elif compute_type == "float32":
            return high * 1.4
        elif compute_type == "float16":
            return high
        else:  # "auto"
            return low * 0.4 + high * 0.6

    vram_needed = 0.0
    if use_gpu_stt:
        vram_needed += _model_vram(whisper_key, stt_compute_type, [1000, 1500])
    if use_gpu_mt:
        nllb_key = "nllb-1.3b" if "1.3B" in nllb_model_str or "1.3b" in nllb_model_str else "nllb"
        vram_needed += _model_vram(nllb_key, mt_compute_type, [600, 1300])

    # Margin aman overhead OS/display/OBS window
    vram_needed += 400.0
    margin = vram_free - vram_needed

    tip = ""
    if margin < 1200 and (stt_compute_type == "float16" or mt_compute_type == "float16"):
        tip = " (Select 'int8_float16' to save ~40% VRAM)." if is_en else " (Pilih 'int8_float16' untuk menghemat ~40% VRAM)."

    if margin >= 1200:
        if is_en:
            return (
                "safe",
                f"🟢 VRAM Safe ({vram_free/1024:.1f} GB free)",
                f"Estimated load ~{vram_needed/1024:.1f} GB of {vram_free/1024:.1f} GB free VRAM. Free of OOM risk.",
            )
        return (
            "safe",
            f"🟢 VRAM Aman ({vram_free/1024:.1f} GB bebas)",
            f"Beban estimasi ~{vram_needed/1024:.1f} GB dari sisa {vram_free/1024:.1f} GB VRAM. Bebas risiko OOM.",
        )
    elif margin >= 0:
        if is_en:
            return (
                "caution",
                f"🟡 VRAM Caution ({vram_free/1024:.1f} GB free)",
                f"Estimated load ~{vram_needed/1024:.1f} GB is close to {vram_free/1024:.1f} GB free VRAM.{tip}",
            )
        return (
            "caution",
            f"🟡 VRAM Waspada ({vram_free/1024:.1f} GB bebas)",
            f"Beban estimasi ~{vram_needed/1024:.1f} GB mendekati sisa {vram_free/1024:.1f} GB VRAM.{tip}",
        )
    else:
        if is_en:
            return (
                "danger",
                f"🔴 OOM Danger (Needs ~{vram_needed/1024:.1f} GB)",
                f"{vram_free/1024:.1f} GB free VRAM is less than required ~{vram_needed/1024:.1f} GB.{tip} Use a smaller model.",
            )
        return (
            "danger",
            f"🔴 Bahaya OOM (Butuh ~{vram_needed/1024:.1f} GB)",
            f"Sisa {vram_free/1024:.1f} GB VRAM kurang dari kebutuhan ~{vram_needed/1024:.1f} GB.{tip} Gunakan model lebih kecil.",
        )

