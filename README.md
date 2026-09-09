# LumaCaption

Companion ringan untuk OBS Studio. Suara mikrofon ditranskripsi lokal, diterjemahkan ke maksimal tiga bahasa, lalu ditampilkan lewat OBS **Browser Source**. Tidak memakai API cloud, OBS WebSocket, atau file caption per bahasa.

## Pipeline lokal

```text
Mikrofon Windows → 16 kHz mono → Silero VAD → faster-whisper → NLLB-200 → Browser Source lokal
```

- Audio hening tidak menjalankan STT atau MT.
- Server overlay hanya bind ke loopback (`127.0.0.1`).
- Server tetap hidup saat caption dihentikan, jadi Browser Source tidak perlu disambungkan ulang.
- Semua target diterjemahkan dalam satu batch dari satu transcript.

> **Latensi:** caption dikirim setelah ucapan berakhir dan jeda VAD terpenuhi, bukan streaming kata per kata. Latensi bergantung pada panjang ucapan, model, dan perangkat; belum ada benchmark saat OBS aktif. Uji sintetis offline pada CPU mencatat cold-start Whisper small + transkripsi `9,38 detik`, lalu NLLB + tiga terjemahan `4,28 detik`. Angka ini mencakup pemuatan model, bukan latensi warm inference.

> **Lisensi:** NLLB-200-distilled-600M memakai **CC-BY-NC-4.0** dan tidak berlisensi untuk penggunaan komersial. Ganti model MT sebelum distribusi komersial atau penggunaan pada produk yang dimonetisasi. Terjemahan dapat salah; jangan gunakan untuk konteks medis, hukum, atau keselamatan.

## Persyaratan

- Windows 10/11
- Python 3.11, 3.12, atau 3.14
- OBS Studio dengan Browser Source
- NVIDIA GPU (opsional, disarankan RTX seri 20/30/40)

Untuk akselerasi GPU lokal tanpa mengubah PATH sistem Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Aplikasi otomatis mendaftarkan DLL NVIDIA (`cublas`, `cudnn`, `cuda_nvrtc`) ke proses saat runtime. Jika CUDA gagal, engine otomatis fallback ke CPU menggunakan **model yang sama** (tidak diam-diam menurunkan ukuran model).

## Menjalankan Aplikasi

### Opsi A: Standalone .EXE (Tanpa Python)
Masuk ke folder `dist/LumaCaption/` lalu klik dua kali:
```text
dist\LumaCaption\LumaCaption.exe
```
Folder `dist/LumaCaption` sudah berisi runtime Python mandiri, seluruh library AI, dan CUDA DLL. Bisa dipindah atau dibuat shortcut ke Desktop.

### Opsi B: Menggunakan Python / Source
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Opsional akselerasi GPU:
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe main.py
# Atau klik ganda start.bat
```

Model diperiksa dan disiapkan sebelum status **LIVE**:
- faster-whisper `large-v3-turbo` atau `medium`/`small`/`base`/`tiny`;
- checkpoint NLLB INT8 (~645 MB);
- Silero ONNX (`models/silero_vad.onnx`).
Setelah model tersimpan di cache lokal, seluruh proses berjalan 100% offline.

## Setup OBS Browser Source

1. Jalankan LumaCaption.
2. Pilih mikrofon dan satu sampai tiga bahasa target.
3. Klik **Simpan**.
4. Gunakan **Tes Overlay** untuk membuka preview dan memastikan server lokal aktif.
5. Pada kartu **Browser Sources**, tersedia opsi:
   - **★ Multi-Bahasa Sekaligus (All-in-One)**: 1 Browser Source untuk menampilkan semua bahasa target bertumpuk rapi sekaligus (seperti subtitle stream VTuber anime).
   - Opsi per bahasa jika ingin memisahkan source di OBS.
6. Di OBS, tambah satu **Browser Source**.
7. Paste URL, lalu set ukuran `1920 × 300` (atau sesuaikan).
8. Aktifkan **Shutdown source when not visible** hanya bila penghematan RAM lebih penting daripada reconnect instan.
9. Klik **Mulai Caption** di LumaCaption.

Contoh URL:

```text
# All-in-One: Semua bahasa target bertumpuk sekaligus dalam satu overlay (Gaya VTuber)
http://127.0.0.1:8765/overlay?lang=all&theme=vtuber&font=Segoe+UI&size=48&color=%23FFFFFF&outline=%23806CFF

# Individual:
http://127.0.0.1:8765/overlay?lang=English&theme=vtuber&font=Segoe+UI&size=48&color=%23FFFFFF&outline=%23806CFF
http://127.0.0.1:8765/overlay?lang=Japanese&theme=card&font=Georgia&size=64
```

Overlay transparan, meng-escape semua teks, memudarkan caption lama, dan mencoba reconnect otomatis setelah server restart. Caption dibersihkan saat koneksi putus agar teks lama tidak tertinggal. Tiap Browser Source menjalankan Chromium/CEF sendiri; biaya RAM perlu diukur pada setup OBS yang dipakai.

## Kontrol

- **Dashboard:** dua kolom pada jendela lebar, satu kolom saat dipersempit. Scroll mengakses semua kartu; tombol utama tetap di bawah. Ukuran minimum `620 × 560`.
- **Mikrofon:** daftar endpoint dideduplikasi; WASAPI diprioritaskan pada Windows. Capture memakai float32 pada sample rate perangkat, lalu downmix stereo ke PCM16 mono untuk VAD.
- **Meter audio:** aktif sebelum STT/MT dimuat. RMS menunjukkan energi rata-rata, peak menunjukkan puncak, VAD menunjukkan probabilitas ucapan. Meter memiliki smoothing ringan; bukan penguat suara.
- **Bahasa ucapan:** manual memberi routing lebih stabil; Auto-detect cocok untuk ucapan lebih panjang.
- **Target:** target 1 wajib; target 2/3 memakai **Tidak digunakan** bila tidak diperlukan. Tidak ada pilihan kosong. Target aktif harus berbeda.
- **Gaya Caption:**
  - **Tema:** `vtuber` (background 100% transparan dengan outline tebal & glowing stroke seperti subtitle VTuber/anime) atau `card` (kotak gelap rounded dengan blur).
  - **Font:** Segoe UI, Arial, Verdana, Tahoma, Trebuchet MS, Georgia, Consolas.
  - **Ukuran (px):** `16–96 px`. Auto-fit jika area OBS sempit.
  - **Warna Teks:** `#FFFFFF` (Putih), `#FFEE55` (Kuning), `#4DD8E7` (Cyan), `#FF718D` (Pink), `#58D6A8` (Hijau mint), atau ketik kode HEX.
  - **Warna Outline:** `#806CFF` (Ungu anime), `#A855F7` (Violet), `#000000` (Hitam pekat), `#4DD8E7` (Cyan), `#FF718D` (Pink), `#F0BA66` (Gold), `none` (tanpa outline).
  - Klik **Simpan**, lalu copy URL ke OBS atau tekan **Tes Overlay**.
- **Whisper:** `large-v3-turbo` (disarankan pada GPU), `medium`, `small` (default CPU), `base`, `tiny`.
- **Beam:** 1, 3 (default), 5. Nilai lebih tinggi mencari alternatif lebih banyak; beam 3 memberikan akurasi stabil tanpa lonjakan latensi.
- **Istilah khusus:** petunjuk kata penting opsional (nama, istilah teknis, singkatan) maks. 300 karakter.
- **STT device:** `cuda` atau `auto` (otomatis GPU bila tersedia) / `cpu`.
- **MT device:** `cuda` untuk terjemahan instan (~160–250 ms) atau `cpu`.
- **Ambang VAD:** probabilitas ucapan (default `0.50`).
- **Jeda (ms):** waktu hening sebelum ucapan dinyatakan selesai (default `650 ms`, rentang 100–3000 ms).
- **Batas (detik):** batas durasi ucapan bersambung tanpa jeda (default `20 detik`, rentang 3–60 detik). Ucapan yang melewati batas tanpa jeda dihentikan aman untuk mencegah caption terpotong.
- **Hapus caption:** waktu sebelum caption memudar di OBS (default `8 detik`).

## Hasil Pengukuran dan Benchmark

Benchmark resmi dijalankan pada sampel publik dataset **Google FLEURS Indonesian (`id_id`)** berlisensi CC-BY-4.0 dengan NVIDIA GeForce RTX 4060:

| Model | Beam | Perangkat | WER | CER | Rata-rata STT | RTF |
|---|---|---|---|---|---|---|
| `medium` | 1 | CUDA | 19.62% | 7.82% | 463.0 ms | 0.050 |
| `medium` | 3 | CUDA | 10.50% | 4.06% | 717.6 ms | 0.067 |
| `large-v3-turbo` | 1 | CUDA | 14.89% | 6.82% | 416.6 ms | 0.045 |
| `large-v3-turbo` | 3 | CUDA | **10.04%** | **3.99%** | **471.3 ms** | **0.044** |

**Terjemahan NLLB-200 (Batch English + Japanese):**
- Rata-rata latensi MT: **165–245 ms**
- P95 latensi MT: **509 ms**

**Total latensi setelah jeda bicara selesai:**
- STT (large-v3-turbo beam 3) + MT (2 bahasa) = **~630–720 ms**. Caption langsung muncul di OBS Browser Source setelah jeda hening VAD.

Untuk mereproduksi benchmark pada dataset publik:

```powershell
.\.venv\Scripts\python.exe tests\benchmark_caption.py --split evaluation --models medium large-v3-turbo --beams 3 --device cuda
```

Setelah font/ukuran diubah, **Copy ulang URL ke OBS**. Style tersimpan dalam parameter URL, bukan pengaturan global browser. URL lama tetap memakai style lamanya; URL tanpa style memakai default.

Pengaturan disimpan atomik ke [config.json](file:///e:/project/plugin/config.json). Konfigurasi lama otomatis mengabaikan field OBS WebSocket dan `obs_source`, serta mendapatkan default font/ukuran.

## Troubleshooting

### `Error starting stream` / `PaErrorCode -9999`

Pada versi sebelumnya, thread caption belum menginisialisasi COM Windows. WASAPI gagal mulai meskipun endpoint ditemukan; pesan PortAudio dapat menyebut `Windows WDM-KS` walau perangkat terpilih memakai WASAPI. Perbaikan menginisialisasi COM pada thread capture dan menutup stream yang gagal sebelum retry. Tutup aplikasi lama lalu jalankan ulang agar perbaikan aktif.

Jika error masih muncul setelah restart, kirim pesan lengkap dan nama mic yang dipilih; jangan mengubah ambang VAD untuk mengatasi kegagalan membuka perangkat.

### Meter mikrofon tidak bergerak

1. Pastikan Windows **Settings → Privacy & security → Microphone** mengizinkan desktop apps.
2. Klik **Refresh** dan pilih endpoint fisik atau virtual yang sama dengan input stream.
3. Pastikan mikrofon tidak mute di perangkat, SteelSeries Sonar, atau Windows mixer.
4. Hentikan aplikasi lain yang membuka perangkat dalam exclusive mode.
5. Mulai caption lagi. Status harus menampilkan nama perangkat dan sample rate.

### Level tetap sekitar −47/−48 dBFS saat bicara

Level konstan hanya membuktikan adanya energi audio, bukan ucapan. Nilai noise floor berbeda tiap perangkat/routing; jangan menjadikannya ambang ucapan.

1. Hentikan caption, lalu pilih mic fisik yang sedang dipakai, misalnya **Microphone (REXUS SNARE MICROPHONE)** atau **Microphone (HyperX Cloud Revolver S)**. Jangan pilih Steam Streaming Microphone untuk mic USB langsung.
2. Mulai caption, diam 3 detik, kemudian bicara dekat mic. RMS/peak seharusnya berubah; VAD harus melewati ambang agar ucapan diproses. **Sinyal tetap** atau **Sinyal sangat kecil** saat bicara berarti perlu memeriksa perangkat/mute/routing.
3. Jika memakai Sonar, pilih mic fisik yang benar sebagai input tab **Microphone** di SteelSeries GG. Periksa meter di Sonar, mute, noise gate, dan gain. Setelah meter Sonar bereaksi, pilih **SteelSeries Sonar - Microphone** di LumaCaption.
4. Bila meter Windows juga tidak bereaksi, periksa tombol mute, kabel/USB, level input Windows, dan izin mic. Mengubah VAD tidak memperbaiki input yang mute atau salah routing.
5. Bila level berubah jelas tetapi VAD tetap rendah, pastikan suara benar-benar ucapan, dekatkan mic, lalu uji ambang `0.35`. Nilai rendah dapat menghasilkan false positive.
6. Setelah ucapan terdeteksi, beri jeda sesuai **Jeda** agar transcript dan terjemahan diterbitkan.

### Browser Source kosong

1. Klik **Tes Overlay**. Preview browser harus langsung menampilkan caption contoh.
2. Copy ulang URL dari bahasa yang benar; nilai `lang` bersifat case-sensitive.
3. Pastikan port belum dipakai aplikasi lain.
4. Di OBS, klik **Refresh cache of current page** pada Browser Source.
5. Gunakan ukuran `1920 × 300`; background memang transparan.

OBS tidak perlu WebSocket, password, source Text (GDI+), atau koneksi keluar.

### Ucapan terdeteksi tetapi teks belum keluar

Model dimuat saat ucapan pertama. Unduhan pertama dapat lama dan membutuhkan jaringan. Monitor menampilkan status pemuatan/inference. Sesudah cache tersedia, sesi berikutnya tetap lokal.

### CUDA gagal

Install runtime CUDA/cuDNN yang cocok atau pilih CPU. Fallback Whisper: model terpilih pada CUDA, model sama pada CPU, `base` CPU, lalu `tiny` CPU. NLLB CUDA turun ke CPU bila gagal.

### Bahasa caption salah

Pilih bahasa ucapan manual. NLLB membutuhkan kode sumber; klip sangat pendek dapat keliru saat Auto-detect.

### Kata terpotong atau caption terlalu lambat

Ubah **Jeda**. Pipeline sengaja menunggu batas ucapan daripada berulang kali memproses audio overlap.

## Validasi

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q audio mt output stt ui config.py languages.py main.py pipeline.py
```

Tes mencakup migrasi/recovery config, font/ukuran dan URL aman, target opsional, downmix channel kanan, resampling 16/44.1/48 kHz, VAD, pipeline satu/tiga target, inisialisasi/cleanup COM pada thread Windows, cleanup stream gagal dan retry, shutdown, serta resize/scroll/fokus dashboard. Tes loopback memeriksa HTTP, pemisahan bahasa, publish, reconnect, dan auto-clear. Tes UI memakai Tk nyata dan dilewati jika display tidak tersedia; tes logika fit caption memakai Node.js bila tersedia. Node.js tidak diperlukan untuk menjalankan aplikasi. Suite tidak membuka mic atau memuat model STT/MT.

## Privasi dan atribusi

Audio hanya berada di memori proses dan tidak ditulis ke disk. Tidak ada telemetry. Koneksi internet hanya dibutuhkan saat model belum ada di cache.

- Silero VAD: MIT, ONNX resmi dari `snakers4/silero-vad`.
- faster-whisper/CTranslate2: runtime inference lokal.
- NLLB conversion: `mijuanlo/nllb-200-distilled-600M-ct2-int8`, pinned revision, turunan Meta NLLB berlisensi CC-BY-NC-4.0.

## Phase 2

Piper TTS dapat menambah dubbed audio. Versi sekarang hanya menghasilkan caption teks.
