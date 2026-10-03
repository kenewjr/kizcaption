# KizCaption

Companion live subtitle offline multibahasa ringan untuk OBS Studio. Suara mikrofon ditranskripsi lokal, diterjemahkan ke maksimal tiga bahasa secara simultan, lalu ditampilkan langsung lewat OBS **Browser Source**. 100% offline tanpa API cloud berbayar, tanpa OBS WebSocket plugin tambahan, dan hemat resource.

![KizCaption Logo](src/lumacaption/assets/kzp_logo.png)

> **Credit**: Dibuat dan dirancang oleh **kenewjr 2026**.  
> **Identitas Visual**: Logo orisinal teks tipografi **KZP** neon cyan & ultraviolet (bebas hak cipta).

---

## Arsitektur Pipeline Lokal

```text
Mikrofon Windows → 16 kHz Mono → Silero VAD v5 → Faster-Whisper → NLLB-200 INT8 → Browser Source OBS
```

- **Zero-Latency Pause Detection**: Deteksi jeda hening pintar Silero VAD hanya mengirim audio setelah pembicara selesai berbicara, tanpa memotong kata. Audio hening tidak membebani komputasi STT/MT.
- **Server Overlay WebSocket**: Port lokal mandiri (`127.0.0.1:8765`), tetap aktif saat caption dihentikan sehingga Browser Source di OBS tidak perlu disambung ulang (*zero reconnect penalty*).
- **Batch Translation**: Seluruh 3 bahasa target diterjemahkan dalam satu siklus inferensi batch NLLB yang sangat efisien.

---

## Fitur Unggulan KizCaption (v1.0.0)

### 1. Struktur Antarmuka 6 Tab Bersih (Tanpa Duplikasi)
Antarmuka pengguna tertata rapi menggunakan sistem tab Tkinter modern:
- **Tab 1 — Monitor & Pengaturan Cepat**:
  - Live Audio VU Meter responsif (RMS dBFS, PEAK dBFS, VAD Indicator).
  - Pilihan mikrofon & pengatur sensitivitas Volume Gain (dB).
  - Monitor teks transkripsi suara asli pembicara & 3 slot hasil terjemahan.
- **Tab 2 — Mesin & VAD**:
  - Pilihan model STT Whisper (`tiny` hingga `large-v3-turbo`), perangkat (`auto`, `cuda`, `cpu`), dan beam size.
  - Pilihan model MT NLLB-200, perangkat, beam size, dan batas thread CPU.
  - Parameter Silero VAD (ambang probabilitas suara, jeda hening ms, batas durasi kalimat).
  - Sensor Kata Kasar Otomatis (TOS Safe) dan normalisasi slang streamer.
  - Saluran audio (mix/left/right), normalisasi otomatis sinyal pelan, dan port overlay.
- **Tab 3 — Gaya Caption (3 Profil & Live In-App Preview)**:
  - 1-Click Platform Safe-Zone Presets (YouTube 1080p, Twitch, TikTok Live 9:16 portrait).
  - Konfigurasi independen untuk Slot 1, Slot 2, dan Slot 3.
  - **Live In-App Caption Preview**: Pratinjau kanvas langsung di dalam aplikasi untuk melihat hasil font, warna, outline, glow neon, dan background tanpa membuka browser luar.
  - Fitur **Impor CSS** dan **Ekspor CSS** per profil.
- **Tab 4 — Model & Resource**:
  - Inspeksi hardware otomatis (GPU, VRAM bebas, RAM sistem, jumlah core CPU).
  - Estimasi kebutuhan VRAM, RAM, dan disk untuk tiap model.
  - Download manager lokal dengan tombol unduh, cek status, dan progress bar.
- **Tab 5 — OBS Setup**:
  - Daftar URL Browser Source untuk Slot 1, 2, 3, dan mode Multi-Bahasa All-in-One.
  - Tombol Salin URL dan Buka Preview Browser.
- **Tab 6 — Tentang**:
  - Informasi versi KizCaption v1.0.0, tombol Cek Pembaruan GitHub, logo KZP, lisensi komponen, dan credit `by kenewjr 2026`.

### 2. Tampilan Scrollbar Modern & Dukungan Tema Gelap / Terang
- Scrollbar ramping minimalis (8px) dengan sudut membulat, tanpa panah atas/bawah kuno.
- Warna scrollbar otomatis menyatu dengan latar belakang tema aktif:
  - **Mode Gelap**: Background `#080B14`, thumb `#252F49`, aksen hover `#806CFF`.
  - **Mode Terang**: Background `#F1F4F9`, thumb `#CBD5E1`, aksen hover `#6366F1`.
- Navigasi mousewheel cerdas: scrolling bekerja mulus tanpa macet.

### 3. 30 Preset Gaya Caption Modern (Dari Imut hingga Keren)
Tersedia 30 preset visual siap pakai dalam 6 kategori:
- **Minimal**: `Clean White`, `Studio Subtitle`, `Mono Console`, `Bold Contrast`, `Soft Shadow`.
- **Anime / VTuber (Imut & Kawaii)**:
  - `Sakura`: Merah muda bunga sakura manis dengan pastel pink glow `#FB7185`.
  - `Lavender Glow`: Magical girl dreamy aesthetic dengan neon violet glow `#C084FC`.
  - `Candy Pop`: Warna ceria permen vanila `#FEF08A` & stroberi pop `#DB2777`.
  - `Pastel Mint`: Matcha & mint milkshake segar dengan aksen teal lembut.
  - `Manga Stroke`: Komik shonen ekspresif dengan outline tebal 5px dan shadow 3D pop.
- **Card**: `Midnight Card`, `Glass Lite`, `Rounded Slate`, `Paper Light`, `Compact Pill`.
- **Broadcast**: `Lower Third`, `News Accent`, `Sport Strip`, `Interview`, `Documentary`.
- **Neon (Keren & Cyberpunk)**:
  - `Cyber Violet`: Cyberpunk 2077 Night City dengan radial glow magenta `#D946EF`.
  - `Cyan Edge`: Tron Sci-Fi HUD electric ice cyan dengan glow blue `#06B6D4`.
  - `Electric Lime`: Terminal hacker matrix dengan glow lime emerald `#10B981`.
  - `Synthwave`: 80s outrun sunset gradient purple `#2E1065` ke `#701A75` ber-border pink.
  - `Sunset Duo`: Gradasi hangat magenta-oranye dengan outline tajam.
- **Creative**: `Retro Mono`, `Comic Bubble`, `Gradient Ribbon`, `Elegant Serif`, `Stream Badge`.

---

## Persyaratan Sistem

- **Sistem Operasi**: Windows 10 atau Windows 11 (64-bit)
- **Python**: 3.11, 3.12, atau 3.14
- **OBS Studio**: Versi 28 ke atas (dilengkapi Browser Source)
- **Kartu Grafis (Opsional)**: NVIDIA RTX Seri 20/30/40/50 dengan VRAM 4 GB+ untuk akselerasi CUDA penuh. Fallback ke CPU otomatis bekerja jika CUDA tidak tersedia.

---

## Cara Menjalankan

### Opsi A: Menggunakan Source Python
```powershell
# 1. Clone repository & buat virtual environment
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 2. Install dependensi inti
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. (Opsional) Install akselerasi GPU NVIDIA CUDA 12:
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt

# 4. Jalankan aplikasi
.\.venv\Scripts\python.exe main.py
# Atau cukup klik ganda file start.bat
```

### Opsi B: Standalone .EXE (Tanpa Instalasi Python)
1. Unduh rilis paket `KizCaption-windows-x64.zip` dari halaman Releases GitHub.
2. Ekstrak file zip ke folder mana saja.
3. Jalankan `KizCaption.exe`. Semua dependensi AI dan DLL CUDA sudah terbundel mandiri.

---

## Panduan Setup OBS Browser Source

1. Buka **KizCaption**, pilih mikrofon Anda, dan klik **Mulai Caption**.
2. Masuk ke **Tab 6 — OBS Setup**.
3. Salin salah satu URL Browser Source:
   - **Multi-Bahasa Sekaligus (All-in-One)**: Menampilkan 3 bahasa target bertumpuk rapi dalam 1 layer browser source:
     ```text
     http://127.0.0.1:8765/overlay.html
     ```
   - **Per Profil Slot**:
     ```text
     http://127.0.0.1:8765/overlay?profile=1
     http://127.0.0.1:8765/overlay?profile=2
     http://127.0.0.1:8765/overlay?profile=3
     ```
4. Di OBS Studio, tambahkan **Browser Source** baru pada Sources.
5. Tempelkan (*paste*) URL tersebut.
6. Atur resolusi Browser Source ke `1920 × 300` (atau sesuaikan dengan kebutuhan scene Anda).
7. Selesai! Caption live akan langsung muncul secara otomatis saat Anda berbicara di mikrofon.
8. **Diagnostik Live**: Tab OBS Setup menampilkan status koneksi secara real-time (jumlah Browser Source yang tersambung dan waktu transmisi caption terakhir).

---

## Lisensi & Atribusi

- **KizCaption UI & Pipeline**: Hak cipta © 2026 **kenewjr**.
- **Faster-Whisper**: Lisensi MIT (OpenAI Whisper & Syllable/CTranslate2).
- **Silero VAD**: Lisensi MIT.
- **NLLB-200-distilled-600M**: Lisensi Meta CC-BY-NC 4.0 (Non-komersial).
