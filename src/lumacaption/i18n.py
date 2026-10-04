"""Modul internasionalisasi (i18n) untuk KizCaption.
Menyediakan teks antarmuka dalam Bahasa Indonesia (santai, umum, tidak kaku) dan English.
"""
from __future__ import annotations

from typing import Any

DEFAULT_LANGUAGE = "id"
SUPPORTED_LANGUAGES = ("id", "en")

# Kamus terjemahan:
# 'id': Bahasa Indonesia umum, ramah pengguna, modern, tidak kaku
# 'en': Standard, clean modern desktop software terminology
STRINGS: dict[str, dict[str, str]] = {
    # --- Header & Global ---
    "app_kicker": {
        "id": "OFFLINE • PRIVAT • RINGAN & HEMAT DAYA",
        "en": "OFFLINE • PRIVATE • LOW RESOURCE",
    },
    "app_subtitle": {
        "id": "Transkripsi dan terjemahan suara offline langsung ke OBS Browser Source.",
        "en": "Real-time speech-to-text and offline translation direct to OBS Browser Source.",
    },
    "btn_light_mode": {
        "id": "Mode Terang",
        "en": "Light Mode",
    },
    "btn_dark_mode": {
        "id": "Mode Gelap",
        "en": "Dark Mode",
    },
    "btn_lang_toggle": {
        "id": "Bahasa: ID",
        "en": "Language: EN",
    },
    "mode_ez": {
        "id": "Mode Santai (EZ)",
        "en": "EZ Mode (Quick)",
    },
    "mode_adv": {
        "id": "Mode Lengkap (Advanced)",
        "en": "Advanced Mode",
    },
    "status_ready": {
        "id": "SIAP",
        "en": "READY",
    },
    "status_listening": {
        "id": "MENDENGARKAN",
        "en": "LISTENING",
    },
    "status_processing": {
        "id": "MEMPROSES",
        "en": "PROCESSING",
    },
    "status_stopped": {
        "id": "BERHENTI",
        "en": "STOPPED",
    },
    "status_error": {
        "id": "ERROR",
        "en": "ERROR",
    },

    # --- Tab Headers ---
    "tab_monitor": {
        "id": "  Monitor & Pengaturan Cepat  ",
        "en": "  Monitor & Quick Settings  ",
    },
    "tab_engine": {
        "id": "  Mesin & VAD  ",
        "en": "  Engine & VAD  ",
    },
    "tab_styles": {
        "id": "  Gaya Tampilan OBS  ",
        "en": "  OBS Caption Styles  ",
    },
    "tab_models": {
        "id": "  Model & Resource  ",
        "en": "  Models & Hardware  ",
    },
    "tab_vocab": {
        "id": "  Kamus & Filter Kata  ",
        "en": "  Vocabulary & Filter  ",
    },
    "tab_obs": {
        "id": "  Integrasi OBS  ",
        "en": "  OBS Integration  ",
    },
    "tab_about": {
        "id": "  Tentang  ",
        "en": "  About  ",
    },

    # --- Tab 1: Monitor & Quick Settings ---
    "card_template_title": {
        "id": "TEMPLATE CEPAT (EZ MODE)",
        "en": "QUICK TEMPLATES (EZ MODE)",
    },
    "card_template_sub": {
        "id": "Pilih bahasa bicara & beban PC, semua setelan langsung beres otomatis",
        "en": "Select speech language & PC load, all settings configure automatically",
    },
    "label_speech_lang": {
        "id": "Bahasa Obrolan:",
        "en": "Spoken Language:",
    },
    "label_pc_load": {
        "id": "Kualitas & Beban PC:",
        "en": "Quality & PC Load:",
    },
    "preset_low": {
        "id": "Hemat (PC Ringan)",
        "en": "Low (Lightweight PC)",
    },
    "preset_med": {
        "id": "Seimbang (Rekomendasi)",
        "en": "Medium (Recommended)",
    },
    "preset_high": {
        "id": "Akurasi Tinggi (PC Kuat)",
        "en": "High (Powerful PC)",
    },
    "card_runtime_title": {
        "id": "STATUS RUNTIME & LOG",
        "en": "RUNTIME STATUS & LOGS",
    },
    "card_runtime_sub": {
        "id": "Pantau proses kerja dan riwayat ucapan yang ditangkap",
        "en": "Monitor running status and captured speech history",
    },
    "card_controls_title": {
        "id": "KONTROL UTAMA",
        "en": "MAIN CONTROLS",
    },
    "card_controls_sub": {
        "id": "Mulai atau hentikan live caption ke OBS",
        "en": "Start or stop live captioning to OBS",
    },
    "btn_start_caption": {
        "id": "Mulai Caption",
        "en": "Start Caption",
    },
    "btn_stop_caption": {
        "id": "Hentikan",
        "en": "Stop Caption",
    },
    "btn_open_obs": {
        "id": "Buka OBS Browser Source",
        "en": "Open OBS Browser Source",
    },
    "card_audio_title": {
        "id": "LEVEL AUDIO & MIKROFON",
        "en": "AUDIO LEVEL & MICROPHONE",
    },
    "card_audio_sub": {
        "id": "Monitor kekuatan suara mikrofon secara real-time",
        "en": "Real-time microphone input volume monitor",
    },
    "card_preview_title": {
        "id": "PREVIEW TEKS OBS",
        "en": "OBS CAPTION PREVIEW",
    },
    "card_preview_sub": {
        "id": "Tampilan live teks yang sedang dikirim ke OBS",
        "en": "Live preview of text being broadcast to OBS",
    },
    "card_quick_title": {
        "id": "PENGATURAN CEPAT",
        "en": "QUICK SETTINGS",
    },
    "card_quick_sub": {
        "id": "Atur mikrofon, bahasa tujuan, dan tampilan teks",
        "en": "Configure microphone, target languages, and caption look",
    },
    "field_mic": {
        "id": "Mikrofon",
        "en": "Microphone",
    },
    "field_source_lang": {
        "id": "Bahasa Asal",
        "en": "Source Language",
    },
    "field_target_1": {
        "id": "Terjemahan 1",
        "en": "Translation 1",
    },
    "field_target_2": {
        "id": "Terjemahan 2 (Opsional)",
        "en": "Translation 2 (Optional)",
    },
    "field_target_3": {
        "id": "Terjemahan 3 (Opsional)",
        "en": "Translation 3 (Optional)",
    },
    "target_unused": {
        "id": "Tidak digunakan",
        "en": "Not used",
    },
    "field_font_family": {
        "id": "Jenis Huruf",
        "en": "Font Family",
    },
    "field_font_size": {
        "id": "Ukuran Huruf (pt)",
        "en": "Font Size (pt)",
    },
    "field_text_color": {
        "id": "Warna Teks",
        "en": "Text Color",
    },
    "field_outline_color": {
        "id": "Warna Garis Luar",
        "en": "Outline Color",
    },
    "field_theme": {
        "id": "Tema Tampilan",
        "en": "Caption Theme",
    },
    "btn_apply_config": {
        "id": "Terapkan Pengaturan",
        "en": "Apply Settings",
    },

    # --- Scroll Hints ---
    "scroll_hint_more": {
        "id": "⬇️ Ada pengaturan & monitor di bawah • Gulir mouse untuk melihat",
        "en": "⬇️ More settings & controls below • Scroll mouse to navigate",
    },
    "scroll_hint_all": {
        "id": "✅ Seluruh konten terlihat di layar",
        "en": "✅ All content is visible on screen",
    },
    "scroll_hint_bottom": {
        "id": "⬆️ Anda berada di paling bawah • Gulir ke atas untuk kontrol utama",
        "en": "⬆️ Reached the bottom • Scroll up for main controls",
    },
    "scroll_hint_pos": {
        "id": "↕️ Posisi: {pct}% • Gulir mouse ke atas atau bawah untuk navigasi",
        "en": "↕️ Position: {pct}% • Scroll mouse up or down to navigate",
    },

    # --- Tab 2: Mesin & VAD ---
    "card_stt_title": {
        "id": "MODEL STT (WHISPER)",
        "en": "STT ENGINE (WHISPER)",
    },
    "card_stt_sub": {
        "id": "Ubah ucapan jadi teks langsung di komputer Anda",
        "en": "High-speed local speech transcription via GPU/CPU",
    },
    "field_whisper_model": {
        "id": "Model Whisper",
        "en": "Whisper Model",
    },
    "field_stt_device": {
        "id": "Perangkat STT",
        "en": "STT Device",
    },
    "field_stt_precision": {
        "id": "Presisi GPU",
        "en": "GPU Precision",
    },
    "field_beam_size": {
        "id": "Beam Size",
        "en": "Beam Size",
    },
    "card_mt_title": {
        "id": "MODEL TRANSLASI (NLLB-200)",
        "en": "TRANSLATION ENGINE (NLLB-200)",
    },
    "card_mt_sub": {
        "id": "Terjemahkan teks ke berbagai bahasa tanpa internet",
        "en": "Offline multilingual machine translation",
    },
    "field_mt_device": {
        "id": "Perangkat MT",
        "en": "MT Device",
    },
    "field_mt_precision": {
        "id": "Presisi MT",
        "en": "MT Precision",
    },
    "field_mt_beam": {
        "id": "Beam Size MT",
        "en": "MT Beam Size",
    },
    "field_cpu_threads": {
        "id": "Thread CPU",
        "en": "CPU Threads",
    },
    "chk_slang_norm": {
        "id": "Normalisasi Slang & Gaul Indonesia (Otomatis)",
        "en": "Normalize Indonesian Slang Words (Instant)",
    },
    "card_vram_title": {
        "id": "KEAMANAN VRAM & GPU",
        "en": "VRAM & GPU SAFETY",
    },
    "card_vram_sub": {
        "id": "Status alokasi memori kartu grafis",
        "en": "Graphic card memory allocation status",
    },
    "card_vad_title": {
        "id": "DETEKSI SUARA (SILERO VAD)",
        "en": "VOICE DETECTION (SILERO VAD)",
    },
    "card_vad_sub": {
        "id": "Potong jeda bicara rapi tanpa memotong kata",
        "en": "Accurate sentence segmentation without clipping words",
    },
    "field_vad_threshold": {
        "id": "Ambang VAD",
        "en": "VAD Threshold",
    },
    "field_silence_gap": {
        "id": "Jeda Hening (ms)",
        "en": "Silence Gap (ms)",
    },
    "field_max_duration": {
        "id": "Batas Durasi (dtk)",
        "en": "Max Duration (sec)",
    },
    "field_clear_timeout": {
        "id": "Hapus Setelah (dtk)",
        "en": "Clear After (sec)",
    },
    "card_dsp_title": {
        "id": "AUDIO PRE-PROCESSING & PORT",
        "en": "AUDIO PRE-PROCESSING & PORT",
    },
    "card_dsp_sub": {
        "id": "Pengaturan saluran suara dan server lokal",
        "en": "Microphone audio channels and local server port",
    },
    "field_channel": {
        "id": "Saluran Audio",
        "en": "Audio Channel",
    },
    "field_mic_gain": {
        "id": "Gain Mic (dB)",
        "en": "Mic Gain (dB)",
    },
    "field_denoiser": {
        "id": "Peredam Bising",
        "en": "Noise Suppressor",
    },
    "chk_auto_normalize": {
        "id": "Normalisasi Otomatis jika suara pelan (< 0.65 peak)",
        "en": "Auto-boost if speech volume is too quiet (< 0.65 peak)",
    },
    "chk_clarity_boost": {
        "id": "Penjernih Suara & Artikulasi Kata (Clarity Boost)",
        "en": "Vocal Clarity Boost & Word Articulation",
    },
    "field_overlay_port": {
        "id": "Port Overlay",
        "en": "Overlay Port",
    },
    "card_vocab_hints_title": {
        "id": "KOSAKATA & KAMUS BELAJAR",
        "en": "VOCABULARY & ADAPTIVE LEARNING",
    },
    "card_vocab_hints_sub": {
        "id": "Injeksi istilah khusus & adaptasi kata berulang",
        "en": "Inject custom terms & adapt to repeated phrases",
    },
    "field_whisper_hints": {
        "id": "Whisper Hints",
        "en": "Whisper Hints",
    },
    "label_learned_vocab": {
        "id": "Kamus Belajar:",
        "en": "Learned Terms:",
    },
    "btn_open_vocab_file": {
        "id": "Buka File Kamus",
        "en": "Open Dictionary File",
    },
    "btn_reset_learned": {
        "id": "Reset Kata Belajar",
        "en": "Reset Learned Words",
    },

    # --- Tab 3: Gaya Tampilan OBS ---
    "label_stage_anchor": {
        "id": "Penempatan Panggung:",
        "en": "Stage Placement:",
    },
    "label_stage_gap": {
        "id": "Jarak Antar Caption (px):",
        "en": "Caption Spacing (px):",
    },
    "tab_slot_1": {
        "id": "  Profil Slot 1  ",
        "en": "  Profile Slot 1  ",
    },
    "tab_slot_2": {
        "id": "  Profil Slot 2  ",
        "en": "  Profile Slot 2  ",
    },
    "tab_slot_3": {
        "id": "  Profil Slot 3  ",
        "en": "  Profile Slot 3  ",
    },

    # --- Tab 4: Model & Resource ---
    "sec_hw_title": {
        "id": "1. Spesifikasi Komputer & Estimasi Beban RAM/VRAM",
        "en": "1. Computer Hardware & RAM/VRAM Estimates",
    },
    "sec_models_title": {
        "id": "2. Manajemen & Unduhan Model AI (Speech-to-Text & Translasi)",
        "en": "2. AI Model Management & Downloads (Speech-to-Text & Translation)",
    },
    "sec_denoiser_title": {
        "id": "3. Komponen Audio & Neural Denoiser (DTLN ONNX)",
        "en": "3. Audio & Neural Denoiser Component (DTLN ONNX)",
    },
    "label_choose_ai_model": {
        "id": "Pilih Model AI:",
        "en": "Select AI Model:",
    },
    "btn_use_ai_model": {
        "id": "Gunakan Model AI Ini",
        "en": "Use This AI Model",
    },
    "btn_download_ai_model": {
        "id": "Unduh / Siapkan Model AI",
        "en": "Download / Setup AI Model",
    },
    "btn_scan_models": {
        "id": "🔍 Pindai Model Lama",
        "en": "🔍 Scan Previous Models",
    },
    "scan_models_title": {
        "id": "Deteksi & Pemulihan Model AI",
        "en": "AI Model Detection & Recovery",
    },
    "scan_models_found": {
        "id": "Berhasil mendeteksi {count} model dari instalasi sebelumnya:\n\n{models}\n\nModel langsung siap digunakan tanpa perlu download ulang!",
        "en": "Successfully detected {count} models from previous installation:\n\n{models}\n\nReady to use immediately without re-downloading!",
    },
    "scan_models_prompt_manual": {
        "id": "Tidak ditemukan model baru di folder standar (AppData / Hugging Face).\n\nApakah Anda ingin memilih folder secara manual (misal folder KizCaption lama atau drive lain)?",
        "en": "No new models found in standard paths (AppData / Hugging Face).\n\nWould you like to select a folder manually (e.g. older KizCaption folder or external drive)?",
    },
    "scan_models_none_found": {
        "id": "Tidak ditemukan model AI yang cocok di folder tersebut.",
        "en": "No matching AI models found in the selected folder.",
    },
    "btn_dl_dtln": {
        "id": "Unduh Komponen DTLN (~3.96 MB)",
        "en": "Download DTLN Component (~3.96 MB)",
    },
    "btn_toggle_dtln": {
        "id": "Aktifkan Sebagai Peredam Bising",
        "en": "Activate as Noise Suppressor",
    },
    "dtln_desc": {
        "id": "DTLN (Dual-signal Transformation LSTM Network) adalah neural network audio mandiri (CPU ONNX) khusus untuk meredam kebisingan mic real-time (suara desah kipas angin, derau PC, dan ketukan keyboard mekanik).\nKomponen ini terpisah dari Model Bahasa Whisper/NLLB dan bekerja langsung pada pemrosesan sinyal mikrofon.",
        "en": "DTLN (Dual-signal Transformation LSTM Network) is a standalone audio neural network (CPU ONNX) designed for real-time mic noise suppression (PC fan hum, room noise, and mechanical keyboard clicks).\nThis component is separate from Whisper/NLLB language models and processes raw microphone input directly.",
    },
    "label_dtln_disk_status": {
        "id": "Status Komponen Disk:",
        "en": "Disk Component Status:",
    },
    "label_dtln_usage_status": {
        "id": "Status Pemakaian:",
        "en": "Usage Status:",
    },

    # --- Tab 5: Kamus & Filter Kata ---
    "sec_vocab_main": {
        "id": "Filter Kosakata & Proteksi Istilah",
        "en": "Vocabulary Filter & Term Protection",
    },
    "sec_vocab_sub": {
        "id": "Daftar kata yang diproteksi dari translasi dan disuntikkan ke Whisper",
        "en": "List of words protected from translation and injected into Whisper",
    },

    # --- Tab 6: Integrasi OBS ---
    "sec_obs_source_title": {
        "id": "URL OBS BROWSER SOURCE",
        "en": "OBS BROWSER SOURCE URL",
    },
    "sec_obs_source_sub": {
        "id": "Gunakan URL berikut di Browser Source OBS Studio",
        "en": "Use the following URL in OBS Studio Browser Source",
    },
    "btn_copy_link": {
        "id": "Salin URL",
        "en": "Copy URL",
    },
    "btn_open_browser": {
        "id": "Buka di Browser",
        "en": "Open in Browser",
    },

    # --- Tab 7: Tentang ---
    "sec_about_header": {
        "id": "TENTANG KIZCAPTION",
        "en": "ABOUT KIZCAPTION",
    },

    # --- Footer & Cards ---
    "btn_save": {
        "id": "Simpan",
        "en": "Save",
    },
    "btn_test_overlay": {
        "id": "Tes Overlay",
        "en": "Test Overlay",
    },
    "btn_start_caption_caps": {
        "id": "MULAI CAPTION",
        "en": "START CAPTION",
    },
    "btn_stop_caption_caps": {
        "id": "HENTIKAN CAPTION",
        "en": "STOP CAPTION",
    },
    "btn_stopping_caption_caps": {
        "id": "MENUTUP MESIN",
        "en": "STOPPING ENGINE",
    },
    "card_input_voice": {
        "id": "INPUT SUARA",
        "en": "VOICE INPUT",
    },
    "card_input_voice_sub": {
        "id": "Mikrofon yang didengarkan",
        "en": "Active recording microphone",
    },
    "btn_refresh_mic": {
        "id": "Segarkan",
        "en": "Refresh",
    },
    "meter_rms": {
        "id": "RMS • RATA-RATA",
        "en": "RMS • AVERAGE",
    },
    "meter_peak": {
        "id": "PEAK • PUNCAK",
        "en": "PEAK • MAXIMUM",
    },
    "meter_vad": {
        "id": "VAD • UCAPAN",
        "en": "VAD • SPEECH",
    },
    "card_caption_lang": {
        "id": "BAHASA CAPTION",
        "en": "CAPTION LANGUAGES",
    },
    "card_caption_lang_sub": {
        "id": "Hingga tiga bahasa target",
        "en": "Up to three target languages",
    },
    "card_status_inference": {
        "id": "STATUS INFERENCE",
        "en": "INFERENCE STATUS",
    },
    "card_status_inference_sub": {
        "id": "Diagnostik transkripsi dan terjemahan",
        "en": "Transcription and translation diagnostics",
    },
    "card_monitor_caption": {
        "id": "MONITOR CAPTION",
        "en": "CAPTION MONITOR",
    },
    "card_monitor_caption_sub": {
        "id": "Transkripsi asli dan teks terjemahan",
        "en": "Original transcript and translated text",
    },
    "transcript_placeholder": {
        "id": "Ucapan asli tampil di sini.",
        "en": "Original speech will appear here.",
    },

    # --- Preset Badges & Descriptions (EZ Mode) ---
    "preset_badge_med": {
        "id": "🔵 SEIMBANG (BALANCED) ⭐",
        "en": "🔵 BALANCED (RECOMMENDED) ⭐",
    },
    "preset_desc_med": {
        "id": "⭐ Rekomendasi Utama • Akurasi tinggi & latency rendah",
        "en": "⭐ Top Recommendation • High accuracy & low latency",
    },
    "preset_badge_low": {
        "id": "🟢 RINGAN & HEMAT (LOW RESOURCE)",
        "en": "🟢 LIGHT & LOW RESOURCE",
    },
    "preset_desc_low": {
        "id": "⚡ Cepat & hemat daya untuk PC kantor/laptop tanpa GPU",
        "en": "⚡ Fast & energy-efficient for office PCs/laptops without GPU",
    },
    "preset_badge_high": {
        "id": "🟣 AKURASI TINGGI (HIGH PRECISION)",
        "en": "🟣 HIGH ACCURACY (STUDIO)",
    },
    "preset_desc_high": {
        "id": "🎯 Model besar & beam luas untuk rekaman studio/podcast",
        "en": "🎯 Large model & wide beam for studio/podcast recordings",
    },

    # --- Placeholders & Activity ---
    "runtime_not_ready": {
        "id": "Model belum disiapkan",
        "en": "Model not initialized yet",
    },
    "metrics_placeholder": {
        "id": "Waktu STT / MT tampil setelah caption pertama",
        "en": "STT / MT time appears after first caption",
    },
    "audio_capture_inactive": {
        "id": "Capture belum aktif",
        "en": "Capture inactive",
    },
    "audio_hint_select_mic": {
        "id": "Pilih mikrofon, lalu mulai caption untuk melihat sinyal.",
        "en": "Select microphone, then start caption to monitor signal.",
    },
    "activity_ready_session": {
        "id": "Siap untuk sesi baru",
        "en": "Ready for new session",
    },
    "vocab_learned_status": {
        "id": "{count} kata dipelajari",
        "en": "{count} terms learned",
    },

    # --- Model Hints & Disk Status (Tab 2) ---
    "hint_model_whisper_small_id": {
        "id": "🇮🇩 Khusus Bahasa Indonesia (Small): Cepat, akurat, hemat VRAM & optimal aksen lokal.",
        "en": "🇮🇩 Indonesian Fine-Tuned (Small): Fast, accurate, low VRAM & optimal local accent.",
    },
    "hint_model_whisper_medium_id": {
        "id": "🇮🇩 Khusus Bahasa Indonesia (Medium): Akurasi tertinggi untuk bahasa & dialek Indonesia.",
        "en": "🇮🇩 Indonesian Fine-Tuned (Medium): Highest accuracy for Indonesian & regional dialects.",
    },
    "hint_model_small": {
        "id": "⭐ Rekomendasi Utama: Terbaik untuk Indonesia, Jawa, Sunda, Melayu & Inggris.",
        "en": "⭐ Top Recommendation: Best for Indonesian, Javanese, Sundanese, Malay & English.",
    },
    "hint_model_base": {
        "id": "🌐 Ringan & hemat CPU: Cocok untuk Indonesia & Inggris formal.",
        "en": "🌐 Lightweight & CPU-friendly: Good for Indonesian & formal English.",
    },
    "hint_model_tiny": {
        "id": "🌐 Tercepat & hemat daya: Inggris atau kalimat Indonesia sederhana.",
        "en": "🌐 Fastest & low power: English or simple Indonesian sentences.",
    },
    "hint_model_medium": {
        "id": "🌐 Akurasi tinggi: Indonesia, Jepang, Korea, Mandarin, Arab & kalimat kompleks.",
        "en": "🌐 High accuracy: Indonesian, Japanese, Korean, Chinese, Arabic & complex speech.",
    },
    "hint_model_large_v3_turbo": {
        "id": "🌐 Multibahasa Global 99+ bahasa: Cepat & presisi tinggi (VRAM >= 4GB).",
        "en": "🌐 Global 99+ languages: Fast & high precision (VRAM >= 4GB).",
    },
    "hint_model_distil_large_v3": {
        "id": "⚠️ Khusus Bahasa Inggris (English Only). Dilarang untuk bahasa Indonesia!",
        "en": "⚠️ English Only. Not suitable for Indonesian!",
    },
    "hint_model_large_v3": {
        "id": "🌐 Akurasi Studio 99+ bahasa dunia (Sangat berat, disarankan VRAM >= 8GB).",
        "en": "🌐 Studio-grade 99+ languages (Heavy, recommended VRAM >= 8GB).",
    },
    "hint_model_default": {
        "id": "Transkripsi ucapan lokal real-time.",
        "en": "Real-time local speech transcription.",
    },
    "disk_status_ready_prefix": {
        "id": "✔ [SIAP DI DISK] ",
        "en": "✔ [READY ON DISK] ",
    },
    "disk_status_missing_msg": {
        "id": "⚠️ [BELUM TERUNDUH] Model '{model}' belum ada di disk (~{size:g} MiB).\nHarap unduh di Tab 'Model & Resource' sebelum mulai transkripsi!\n",
        "en": "⚠️ [NOT DOWNLOADED] Model '{model}' not found on disk (~{size:g} MiB).\nPlease download in 'Models & Hardware' tab before starting caption!\n",
    },
    "disk_status_incomplete_msg": {
        "id": "⚠ [UNDUHAN BELUM LENGKAP] File model '{model}' belum lengkap.\nBuka Tab 'Model & Resource' untuk melanjutkan unduhan.\n",
        "en": "⚠ [INCOMPLETE DOWNLOAD] Model file '{model}' is incomplete.\nOpen 'Models & Hardware' tab to resume download.\n",
    },

    # --- VAD & Engine Hints (Tab 2) ---
    "vad_prob_note": {
        "id": "Ambang 0.50 = 50% probabilitas suara manusia. Ucapan hanya dikirim setelah pembicara berhenti bicara.",
        "en": "Threshold 0.50 = 50% voice probability. Speech is emitted only after speaker pauses.",
    },
    "engine_scroll_all": {
        "id": "✔ Seluruh pengaturan mesin tampak di layar",
        "en": "✔ All engine settings are visible on screen",
    },
    "engine_scroll_bottom": {
        "id": "✔ Pengaturan bagian paling bawah tercapai",
        "en": "✔ Reached the bottom of engine settings",
    },
    "engine_scroll_more": {
        "id": "⬇️ Ada pengaturan di bawah ({pct}% tersisa) • Gulir mouse untuk melihat",
        "en": "⬇️ More settings below ({pct}% remaining) • Scroll mouse to view",
    },

    # --- Tab 3: Style Editor ---
    "editor_slot_label": {
        "id": "Profil Slot {slot}:",
        "en": "Profile Slot {slot}:",
    },
    "editor_preset_label": {
        "id": "Preset:",
        "en": "Preset:",
    },
    "btn_export_css": {
        "id": "Ekspor CSS",
        "en": "Export CSS",
    },
    "btn_import_css": {
        "id": "Impor CSS",
        "en": "Import CSS",
    },
    "editor_tab_typography": {
        "id": "Tipografi",
        "en": "Typography",
    },
    "editor_tab_colors": {
        "id": "Warna & Outline",
        "en": "Color & Outline",
    },
    "editor_tab_background": {
        "id": "Latar Belakang",
        "en": "Background",
    },
    "editor_tab_layout": {
        "id": "Tata Letak & Waktu",
        "en": "Layout & Timing",
    },
    "editor_font_family": {
        "id": "Font Family:",
        "en": "Font Family:",
    },
    "editor_font_size": {
        "id": "Ukuran (px):",
        "en": "Size (px):",
    },
    "editor_font_weight": {
        "id": "Ketebalan (Weight):",
        "en": "Font Weight:",
    },
    "editor_italic": {
        "id": "Italic (Miring)",
        "en": "Italic",
    },
    "editor_letter_spacing": {
        "id": "Spasi Huruf:",
        "en": "Letter Spacing:",
    },
    "editor_text_color": {
        "id": "Warna Teks:",
        "en": "Text Color:",
    },
    "editor_outline_color": {
        "id": "Warna Outline:",
        "en": "Outline Color:",
    },
    "editor_outline_width": {
        "id": "Tebal Outline (px):",
        "en": "Outline Width (px):",
    },
    "editor_shadow_color": {
        "id": "Warna Bayangan:",
        "en": "Shadow Color:",
    },
    "editor_shadow_blur": {
        "id": "Blur Bayangan:",
        "en": "Shadow Blur:",
    },
    "editor_btn_choose": {
        "id": "Pilih",
        "en": "Choose",
    },
    "editor_bg_type": {
        "id": "Tipe Latar:",
        "en": "Background Type:",
    },
    "editor_bg_color": {
        "id": "Warna Latar:",
        "en": "Background Color:",
    },
    "editor_bg_opacity": {
        "id": "Opacity Latar (0–1):",
        "en": "Background Opacity (0–1):",
    },
    "editor_radius": {
        "id": "Sudut Radius (px):",
        "en": "Corner Radius (px):",
    },
    "editor_margin_y": {
        "id": "Margin Bawah (Y px):",
        "en": "Bottom Margin (Y px):",
    },
    "editor_platform_preset": {
        "id": "Safe Zone Platform:",
        "en": "Platform Safe Zone:",
    },
    "label_platform_preset": {
        "id": "Platform Preset:",
        "en": "Platform Preset:",
    },
    "chk_profanity_filter": {
        "id": "🛡️ Sensor Kata Kasar Otomatis (Aman TOS Streaming)",
        "en": "🛡️ Auto Censor Profanity (Stream TOS Safe)",
    },
    "editor_line_wrap": {
        "id": "Arah / Bentuk Tulisan:",
        "en": "Text Layout Mode:",
    },
    "opt_wrap_nowrap": {
        "id": "Memanjang (1 Baris Lurus)",
        "en": "Single-Line (Horizontal)",
    },
    "opt_wrap_wrap": {
        "id": "Melipat ke Atas (Multi-Baris)",
        "en": "Multi-Line (Wrap Upward)",
    },
    "editor_align": {
        "id": "Perataan Teks:",
        "en": "Text Alignment:",
    },
    "editor_max_width": {
        "id": "Lebar Maks (%):",
        "en": "Max Width (%):",
    },
    "editor_animation": {
        "id": "Transisi (Animasi):",
        "en": "Transition (Animation):",
    },
    "editor_timeout": {
        "id": "Durasi Tampil (detik):",
        "en": "Display Duration (sec):",
    },
    "editor_show_label": {
        "id": "Tampilkan Label Bahasa/Slot",
        "en": "Show Language/Slot Label",
    },
    "editor_preview_title": {
        "id": "Pratinjau Gaya Caption (Live In-App Preview)",
        "en": "Caption Style Preview (Live In-App)",
    },
    "editor_sample_text_1": {
        "id": "Halo semuanya! Selamat datang di live streaming KizCaption.",
        "en": "Hello everyone! Welcome to KizCaption live broadcast.",
    },
    "editor_sample_text_2": {
        "id": "Hello everyone! Welcome to KizCaption live broadcast.",
        "en": "Hello everyone! Welcome to KizCaption live broadcast.",
    },
    "editor_sample_text_3": {
        "id": "Halo semuanya! Selamat datang di siaran KizCaption.",
        "en": "Hello everyone! Welcome to KizCaption broadcast.",
    },

    # --- Tab 4: Models & Hardware ---
    "sec_hw_box": {
        "id": "Spesifikasi Hardware",
        "en": "Hardware Specifications",
    },
    "btn_refresh_hw": {
        "id": "🔄  Refresh Hardware",
        "en": "🔄  Refresh Hardware",
    },
    "sec_res_est_title": {
        "id": "1. Estimasi Detail Resource per Model",
        "en": "1. Model Resource Estimates",
    },
    "sec_res_est_desc": {
        "id": "Perbandingan kebutuhan RAM & VRAM riil tiap model. Klik salah satu baris untuk memilih dan mengelola model di bawah.",
        "en": "Real RAM & VRAM requirements per model. Click any row to select and manage the model below.",
    },
    "tbl_col_component": {
        "id": "Komponen & Model",
        "en": "Component & Model",
    },
    "tbl_col_disk": {
        "id": "Ukuran Disk",
        "en": "Disk Size",
    },
    "tbl_col_ram": {
        "id": "RAM (Mode CPU)",
        "en": "RAM (CPU Mode)",
    },
    "tbl_col_vram": {
        "id": "VRAM (Mode CUDA)",
        "en": "VRAM (CUDA Mode)",
    },
    "tbl_col_threads": {
        "id": "CPU Threads",
        "en": "CPU Threads",
    },
    "tbl_col_recommendation": {
        "id": "Karakter, Latensi & Rekomendasi",
        "en": "Character, Latency & Recommendation",
    },
    "dl_status_ready": {
        "id": "Status: Siap",
        "en": "Status: Ready",
    },

    # --- Tab 5: OBS Integration ---
    "obs_server_status_label": {
        "id": "Status Server:",
        "en": "Server Status:",
    },
    "obs_connected_clients_label": {
        "id": "Klien Terhubung:",
        "en": "Connected Clients:",
    },
    "obs_last_activity_label": {
        "id": "Aktivitas Terakhir:",
        "en": "Last Activity:",
    },
    "obs_client_count_desc": {
        "id": "* Klien terhubung menghitung koneksi live Browser Source / browser aktif.\n  Jika subtitle belum tampak di siaran, periksa visibilitas layer source di OBS.",
        "en": "* Connected clients track active Browser Source / browser instances.\n  If subtitle does not appear in stream, check the source visibility in OBS.",
    },
    "obs_all_in_one_title": {
        "id": "★ Overlay Multi-Bahasa (All-in-One)",
        "en": "★ Multi-Language Overlay (All-in-One)",
    },
    "obs_all_in_one_desc": {
        "id": "Cukup gunakan 1 URL ini di OBS Studio. Semua bahasa aktif (Slot 1, 2, 3) akan tampil otomatis dengan profil gayanya masing-masing.\nBahasa yang tidak aktif atau hening akan hilang sendiri.",
        "en": "Just use this single URL in OBS Studio. All active languages (Slot 1, 2, 3) will display automatically with their own styles.\nInactive or silent languages fade away.",
    },
    "obs_guide_title": {
        "id": "PANDUAN PEMASANGAN OBS",
        "en": "OBS SETUP GUIDE",
    },
    "obs_guide_sub": {
        "id": "Langkah cepat menambahkan subtitle ke siaran",
        "en": "Quick steps to add subtitles to your stream",
    },
    "obs_guide_step1": {
        "id": "1. Buka OBS Studio pada scene siaran Anda.",
        "en": "1. Open OBS Studio on your broadcast scene.",
    },
    "obs_guide_step2": {
        "id": "2. Tambahkan Source baru bertipe 'Browser' (Browser Source).",
        "en": "2. Add a new Source of type 'Browser' (Browser Source).",
    },
    "obs_guide_step3": {
        "id": "3. Salin URL Multi-Bahasa di samping lalu Paste ke kolom URL pada OBS.",
        "en": "3. Copy the Multi-Language URL on the left and paste into the URL field in OBS.",
    },
    "obs_guide_step4": {
        "id": "4. Atur resolusi Browser Source:\n   • Lebar: 1920 px (atau sesuai canvas OBS Anda)\n   • Tinggi: 300 px (rekomendasi untuk subtitle bawah)",
        "en": "4. Set the Browser Source resolution:\n   • Width: 1920 px (or match your OBS canvas width)\n   • Height: 300 px (recommended for bottom subtitles)",
    },
    "obs_guide_step5": {
        "id": "5. Centang 'Shutdown source when not visible'.",
        "en": "5. Check 'Shutdown source when not visible'.",
    },
    "obs_guide_step6": {
        "id": "6. Hapus isi 'Custom CSS' di OBS (CSS bawaan KizCaption sudah mengatur transparansi & font).",
        "en": "6. Clear the 'Custom CSS' field in OBS (KizCaption built-in CSS handles transparency & fonts).",
    },
    "obs_guide_step7": {
        "id": "7. Tekan 'Tes Overlay' untuk menguji tampilan subtitle secara langsung.",
        "en": "7. Click 'Test Overlay' to preview live subtitle appearance.",
    },
    "obs_server_preparing": {
        "id": "Menyiapkan server overlay…",
        "en": "Preparing overlay server…",
    },
    "obs_server_ready": {
        "id": "Server overlay aktif (Port {port})",
        "en": "Overlay server active (Port {port})",
    },
    "obs_clients_count_fmt": {
        "id": "{count} terhubung (tampilan pasif)",
        "en": "{count} connected (passive view)",
    },
    "obs_no_caption_sent": {
        "id": "Belum ada caption terkirim",
        "en": "No caption sent yet",
    },

    # --- Tab 6: About ---
    "about_desc": {
        "id": "KizCaption adalah aplikasi pendamping siaran OBS mandiri untuk transkripsi dan terjemahan multibahasa secara offline tanpa cloud API, menjaga privasi suara dan performa sistem.",
        "en": "KizCaption is a standalone OBS companion app for offline multilingual speech-to-text and translation without cloud APIs, preserving voice privacy and system performance.",
    },
    "about_components": {
        "id": "Komponen Utama:",
        "en": "Core Components:",
    },
    "btn_open_config_folder": {
        "id": "Buka Folder Config",
        "en": "Open Config Folder",
    },
    "btn_open_models_folder": {
        "id": "Buka Folder Models",
        "en": "Open Models Folder",
    },
    "btn_open_logs_folder": {
        "id": "Buka Folder Logs",
        "en": "Open Logs Folder",
    },
    "btn_check_updates": {
        "id": "🔄  Periksa Pembaruan",
        "en": "🔄  Check for Updates",
    },
    "update_title": {
        "id": "Pembaruan KizCaption",
        "en": "KizCaption Updates",
    },
    "update_found_msg": {
        "id": "Versi baru tersedia: v{latest}!\n(Versi saat ini: v{current})",
        "en": "A new version is available: v{latest}!\n(Current version: v{current})",
    },
    "update_ask_open": {
        "id": "Apakah Anda ingin membuka halaman rilis GitHub untuk mengunduh?",
        "en": "Would you like to open the GitHub release page to download?",
    },
    "update_none_msg": {
        "id": "KizCaption sudah versi terbaru (v{version}).",
        "en": "KizCaption is already up to date (v{version}).",
    },
    "update_err_msg": {
        "id": "Gagal memeriksa pembaruan dari GitHub",
        "en": "Failed to check for updates from GitHub",
    },
}


def t(key: str, lang: str | None = None, **kwargs: Any) -> str:
    """Ambil teks terlokalisasi berdasarkan key dan bahasa yang aktif."""
    target_lang = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    entry = STRINGS.get(key)
    if not entry:
        return key
    text = entry.get(target_lang) or entry.get(DEFAULT_LANGUAGE) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text
