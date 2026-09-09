from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

from audio.capture import InputDevice, MicrophoneCapture
from config import (
    AppConfig, CAPTION_FONTS, CAPTION_FONT_MIN_SIZE, CAPTION_FONT_MAX_SIZE,
    CAPTION_THEMES, CAPTION_TEXT_COLORS, CAPTION_OUTLINE_COLORS,
    ConfigStore, MODEL_SIZES, OverlayConfig, TargetConfig,
)
from languages import BY_NAME, SOURCE_CHOICES, TARGET_CHOICES, source_display_name
from output.overlay_server import OverlayService, overlay_url
from pipeline import CaptionPipeline, PipelineEvent


BG = "#080B14"
SURFACE = "#0E1322"
SURFACE_2 = "#141B2E"
SURFACE_3 = "#1A2238"
BORDER = "#252F49"
TEXT = "#F4F6FF"
MUTED = "#8C96B4"
SUBTLE = "#626D8B"
VIOLET = "#806CFF"
VIOLET_HOVER = "#9585FF"
CYAN = "#4DD8E7"
GREEN = "#58D6A8"
AMBER = "#F0BA66"
RED = "#FF718D"
UNUSED_TARGET = "Tidak digunakan"


def selected_targets(choices: list[str]) -> list[TargetConfig]:
    if not choices or choices[0].strip() not in TARGET_CHOICES:
        raise ValueError("Target 1 wajib diisi dengan bahasa dari daftar")
    targets = []
    for choice in choices:
        name = choice.strip()
        if name == UNUSED_TARGET:
            continue
        if name not in TARGET_CHOICES:
            raise ValueError("Pilih bahasa dari daftar atau 'Tidak digunakan'")
        targets.append(TargetConfig(name))
    return targets


class FilterCombobox(ttk.Combobox):
    def __init__(self, master, *, choices: tuple[str, ...], **kwargs):
        self._choices = choices
        super().__init__(master, values=choices, **kwargs)
        self.bind("<KeyRelease>", self._filter, add=True)
        self.bind("<FocusOut>", lambda _event: self.configure(values=self._choices), add=True)

    def _filter(self, event) -> None:
        if event.keysym in {"Up", "Down", "Return", "Escape", "Tab"}:
            return
        query = self.get().casefold()
        matches = [item for item in self._choices if query in item.casefold()]
        self.configure(values=matches or self._choices)
        if matches and query:
            self.event_generate("<Down>")


class ControlPanel:
    def __init__(self, root: tk.Tk, config_store: ConfigStore, config: AppConfig, app_dir: Path) -> None:
        self.root = root
        self.config_store = config_store
        self.config = config
        self.app_dir = app_dir
        self.events: queue.Queue[PipelineEvent] = queue.Queue()
        self.pipeline: CaptionPipeline | None = None
        self.overlay_service: OverlayService | None = None
        self._overlay_fingerprint: tuple | None = None
        self._devices: dict[str, InputDevice] = {}
        self._editable: list[tuple[tk.Widget, str]] = []
        self._last_translations: dict[str, str] = {}
        self._advanced_visible = False
        self._closed = False
        self._wide_layout: bool | None = None
        self._level_history: deque[tuple[float, float]] = deque(maxlen=64)
        self._smoothed_db: float | None = None
        self._last_speech_at = 0.0
        self._vad_probability = 0.0
        self._vad_speaking = False
        self._pipeline_error: str | None = None
        self._stopping = False
        self._closing_since = 0.0
        self._processing = False

        self._build_style()
        self._build_ui()
        self._load(config)
        self.refresh_microphones()
        self.root.after(50, lambda: self._sync_overlay(show_error=False))
        self.root.after(80, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_style(self) -> None:
        self.root.title("LumaCaption — Offline OBS Translator")
        self.root.geometry("1180x820")
        self.root.minsize(620, 560)
        self.root.configure(bg=BG)
        self.root.option_add("*Font", "{Segoe UI} 10")
        self.root.option_add("*TCombobox*Listbox.background", SURFACE_2)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", VIOLET)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, bordercolor=BORDER, darkcolor=BG, lightcolor=BG)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE, borderwidth=1, relief="solid")
        style.configure("Inner.TFrame", background=SURFACE)
        style.configure("Soft.TFrame", background=SURFACE_2)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Soft.TLabel", background=SURFACE_2, foreground=TEXT)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 24))
        style.configure("Kicker.TLabel", background=BG, foreground=CYAN, font=("Segoe UI Semibold", 9))
        style.configure("CardTitle.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 12))
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("PageMuted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=SURFACE_2, foreground=GREEN, font=("Segoe UI Semibold", 9), padding=(12, 6))
        style.configure("TButton", background=SURFACE_3, foreground=TEXT, borderwidth=0, padding=(12, 8), font=("Segoe UI Semibold", 9))
        style.map(
            "TButton",
            background=[("active", "#222D48"), ("disabled", SURFACE_2)],
            foreground=[("disabled", SUBTLE)],
        )
        style.configure("Accent.TButton", background=VIOLET, foreground="#FFFFFF", padding=(18, 11), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", VIOLET_HOVER), ("disabled", "#353751")])
        style.configure("Danger.TButton", background="#6B3042", foreground="#FFFFFF", padding=(18, 11), font=("Segoe UI Semibold", 10))
        style.map("Danger.TButton", background=[("active", "#824057")])
        style.configure("Link.TButton", background=SURFACE, foreground=CYAN, padding=(8, 5))
        style.map("Link.TButton", background=[("active", SURFACE_2)])
        style.configure("TEntry", fieldbackground=SURFACE_2, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, padding=8)
        style.configure("TCombobox", fieldbackground=SURFACE_2, background=SURFACE_3, foreground=TEXT, arrowcolor=MUTED, bordercolor=BORDER, padding=7)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", SURFACE_2), ("disabled", SURFACE)],
            foreground=[("disabled", SUBTLE)],
        )
        style.configure("TSpinbox", fieldbackground=SURFACE_2, foreground=TEXT, arrowcolor=MUTED, bordercolor=BORDER, padding=7)
        style.configure("Horizontal.TScale", background=SURFACE, troughcolor=SURFACE_3)
        style.configure(
            "Audio.Horizontal.TProgressbar",
            troughcolor=SURFACE_3,
            background=CYAN,
            bordercolor=SURFACE_3,
            lightcolor=CYAN,
            darkcolor=CYAN,
        )
        style.configure("TSeparator", background=BORDER)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=(22, 18))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        brand = ttk.Frame(header)
        brand.grid(row=0, column=0, sticky="ew")
        ttk.Label(brand, text="OFFLINE • PRIVATE • LOCAL", style="Kicker.TLabel").pack(anchor="w")
        ttk.Label(brand, text="LumaCaption", style="Title.TLabel").pack(anchor="w")
        self._wrapped_label(
            brand,
            text="Suara lokal menjadi caption multibahasa untuk OBS.",
            style="PageMuted.TLabel",
        ).pack(fill="x", pady=(2, 0))
        self.status_var = tk.StringVar(value="SIAP")
        self.status_label = ttk.Label(header, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=0, column=1, sticky="ne", padx=(12, 0), pady=10)

        viewport = ttk.Frame(shell)
        viewport.grid(row=1, column=0, sticky="nsew")
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewport, bg=BG, highlightthickness=0, yscrollincrement=20)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.body = ttk.Frame(self.canvas)
        self._body_window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.left = left = ttk.Frame(self.body)
        self.right = right = ttk.Frame(self.body)
        self.body.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._resize_layout)
        self.root.bind("<MouseWheel>", self._scroll_wheel, add=True)
        for widget_class in ("TCombobox", "TSpinbox", "Combobox", "Spinbox"):
            self.root.bind_class(widget_class, "<MouseWheel>", self._scroll_wheel)
        self.root.bind("<Key-Tab>", self._on_key_tab, add=True)
        self.root.bind("<Shift-Key-Tab>", self._on_key_tab, add=True)

        self.mic_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.beam_var = tk.StringVar()
        self.hotwords_var = tk.StringVar()
        self.duration_var = tk.StringVar()
        self.runtime_var = tk.StringVar(value="Model belum disiapkan")
        self.metrics_var = tk.StringVar(value="Waktu STT / MT tampil setelah caption pertama")
        self.stt_device_var = tk.StringVar()
        self.mt_device_var = tk.StringVar()
        self.target_vars = [tk.StringVar() for _ in range(3)]
        self.overlay_port_var = tk.StringVar()
        self.theme_var = tk.StringVar()
        self.font_var = tk.StringVar()
        self.font_size_var = tk.StringVar()
        self.text_color_var = tk.StringVar()
        self.outline_color_var = tk.StringVar()
        self.vad_var = tk.DoubleVar()
        self.silence_var = tk.IntVar()
        self.timeout_var = tk.IntVar()
        self.audio_device_var = tk.StringVar(value="Capture belum aktif")
        self.audio_hint_var = tk.StringVar(value="Pilih mikrofon, lalu mulai caption untuk melihat sinyal.")
        self.rms_var = tk.StringVar(value="— dBFS")
        self.peak_var = tk.StringVar(value="— dBFS")
        self.vad_status_var = tk.StringVar(value="— %")
        self.activity_var = tk.StringVar(value="Siap untuk sesi baru")
        self.transcript_var = tk.StringVar(value="Ucapan asli akan tampil di sini.")
        self.translation_vars = [tk.StringVar(value="Menunggu caption…") for _ in range(3)]

        audio = self._card(left, "INPUT AUDIO", "Endpoint Windows bersih, diprioritaskan lewat WASAPI")
        mic_line = ttk.Frame(audio, style="Inner.TFrame")
        mic_line.pack(fill="x", pady=(2, 10))
        mic_line.columnconfigure(0, weight=1)
        self.mic_box = ttk.Combobox(mic_line, textvariable=self.mic_var, state="readonly", width=1)
        self.mic_box.grid(row=0, column=0, sticky="ew")
        self.refresh_button = ttk.Button(mic_line, text="↻  Refresh", command=self.refresh_microphones)
        self.refresh_button.grid(row=0, column=1, padx=(8, 0))
        self._track(self.mic_box, "readonly")
        self._track(self.refresh_button)
        self.audio_meter = ttk.Progressbar(
            audio,
            mode="determinate",
            maximum=100,
            style="Audio.Horizontal.TProgressbar",
        )
        self.audio_meter.pack(fill="x", pady=(0, 8))
        meters = ttk.Frame(audio, style="Soft.TFrame", padding=10)
        meters.pack(fill="x", pady=(0, 9))
        for index, (title, variable) in enumerate((
            ("RMS · RATA-RATA", self.rms_var),
            ("PEAK · PUNCAK", self.peak_var),
            ("VAD · UCAPAN", self.vad_status_var),
        )):
            meters.columnconfigure(index, weight=1, uniform="meters")
            ttk.Label(meters, text=title, style="Soft.TLabel", foreground=MUTED,
                      font=("Segoe UI", 8)).grid(row=0, column=index, sticky="w")
            ttk.Label(meters, textvariable=variable, style="Soft.TLabel", foreground=CYAN,
                      font=("Segoe UI Semibold", 12)).grid(row=1, column=index, sticky="w")
        self._wrapped_label(audio, textvariable=self.audio_device_var, style="Muted.TLabel").pack(fill="x")
        self._wrapped_label(audio, textvariable=self.audio_hint_var, style="Card.TLabel").pack(fill="x", pady=(5, 0))

        language = self._card(left, "BAHASA", "Satu suara masuk, maksimal tiga terjemahan sekaligus")
        source_row = ttk.Frame(language, style="Inner.TFrame")
        source_row.pack(fill="x", pady=4)
        source_row.columnconfigure(1, weight=1)
        ttk.Label(source_row, text="Bahasa ucapan", style="Card.TLabel", width=19).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        source_box = FilterCombobox(
            source_row,
            choices=SOURCE_CHOICES,
            textvariable=self.source_var,
            width=1,
        )
        source_box.grid(row=0, column=1, sticky="ew")
        self._track(source_box)
        ttk.Separator(language).pack(fill="x", pady=11)
        for index in range(3):
            row = ttk.Frame(language, style="Inner.TFrame")
            row.pack(fill="x", pady=4)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=f"{index + 1:02d}", style="Muted.TLabel", width=4).grid(row=0, column=0, sticky="w")
            chooser = ttk.Combobox(
                row, values=TARGET_CHOICES if index == 0 else (UNUSED_TARGET, *TARGET_CHOICES),
                textvariable=self.target_vars[index], state="readonly", width=1,
            )
            chooser.grid(row=0, column=1, sticky="ew")
            self._track(chooser, "readonly")
        self._wrapped_label(
            language, text="Target 1 wajib. Pilih 'Tidak digunakan' untuk target 2/3 yang tidak diperlukan.",
            style="Muted.TLabel",
        ).pack(fill="x", pady=(8, 0))

        appearance = self._card(left, "GAYA CAPTION", "Tema, font, ukuran, dan warna untuk Browser Source")
        self._compact_combo_field(appearance, "Tema", CAPTION_THEMES, self.theme_var)
        self._compact_combo_field(appearance, "Font caption", CAPTION_FONTS, self.font_var)
        size_row = self._field_row(appearance, "Ukuran (px)", compact=True)
        size = ttk.Spinbox(
            size_row, from_=CAPTION_FONT_MIN_SIZE, to=CAPTION_FONT_MAX_SIZE,
            textvariable=self.font_size_var, width=1,
        )
        size.grid(row=0, column=1, sticky="ew")
        self._track(size)
        self._compact_combo_field(appearance, "Warna teks", CAPTION_TEXT_COLORS, self.text_color_var)
        self._compact_combo_field(appearance, "Warna outline", CAPTION_OUTLINE_COLORS, self.outline_color_var)
        self._wrapped_label(
            appearance,
            text="vtuber: transparan + stroke & glowing outline. card: kotak gelap transparan.",
            style="Muted.TLabel",
        ).pack(fill="x", pady=(8, 0))

        advanced = self._card(
            left,
            "PENGATURAN MESIN",
            "Model diperiksa sebelum LIVE; hasil hanya setelah jeda bicara",
            collapsible=True,
        )
        self._advanced_body = ttk.Frame(advanced, style="Inner.TFrame")
        left_advanced = ttk.Frame(self._advanced_body, style="Inner.TFrame")
        right_advanced = ttk.Frame(self._advanced_body, style="Inner.TFrame")
        left_advanced.pack(fill="x")
        right_advanced.pack(fill="x")
        self._compact_combo_field(
            left_advanced,
            "Whisper",
            MODEL_SIZES,
            self.model_var,
        )
        self._compact_combo_field(left_advanced, "Beam", ("1", "3", "5"), self.beam_var)
        hint_row = self._field_row(left_advanced, "Istilah khusus", compact=True)
        hint = ttk.Entry(hint_row, textvariable=self.hotwords_var, width=1)
        hint.grid(row=0, column=1, sticky="ew")
        self._track(hint)
        self._wrapped_label(
            left_advanced,
            text="Istilah opsional, maks. 300 karakter. Beam lebih besar belum tentu lebih akurat; proses bisa lebih lama.",
            style="Muted.TLabel",
        ).pack(fill="x", pady=(4, 8))
        self._compact_combo_field(
            left_advanced,
            "STT device",
            ("auto", "cuda", "cpu"),
            self.stt_device_var,
        )
        self._compact_combo_field(
            left_advanced,
            "MT device",
            ("auto", "cuda", "cpu"),
            self.mt_device_var,
        )
        port_row = self._field_row(left_advanced, "Port overlay", compact=True)
        port = ttk.Spinbox(port_row, from_=1024, to=65535, textvariable=self.overlay_port_var, width=1)
        port.grid(row=0, column=1, sticky="ew")
        self._track(port)
        sensitivity_row = self._field_row(right_advanced, "Ambang VAD", compact=True)
        sensitivity = ttk.Spinbox(
            sensitivity_row, from_=0.05, to=0.95, increment=0.05,
            textvariable=self.vad_var, width=1,
        )
        sensitivity.grid(row=0, column=1, sticky="ew")
        self._track(sensitivity)
        silence_row = self._field_row(right_advanced, "Jeda (ms)", compact=True)
        silence = ttk.Spinbox(silence_row, from_=100, to=3000, increment=50, textvariable=self.silence_var, width=1)
        silence.grid(row=0, column=1, sticky="ew")
        self._track(silence)
        duration_row = self._field_row(right_advanced, "Batas (detik)", compact=True)
        duration = ttk.Spinbox(duration_row, from_=3, to=60, textvariable=self.duration_var, width=1)
        duration.grid(row=0, column=1, sticky="ew")
        self._track(duration)
        timeout_row = self._field_row(right_advanced, "Hapus (detik)", compact=True)
        timeout = ttk.Spinbox(timeout_row, from_=0, to=60, textvariable=self.timeout_var, width=1)
        timeout.grid(row=0, column=1, sticky="ew")
        self._track(timeout)
        self._wrapped_label(
            self._advanced_body,
            text="Ambang 0.50 = 50% probabilitas ucapan, bukan gain mic. Beri jeda sebelum batas durasi; ucapan terlalu panjang menghentikan sesi, bukan menerbitkan teks terpotong.",
            style="Muted.TLabel",
        ).pack(fill="x", pady=(8, 0))
        self.advanced_button.configure(command=self._toggle_advanced)

        monitor = self._card(right, "LIVE MONITOR", "Status capture, transcript, dan hasil terakhir")
        activity = ttk.Frame(monitor, style="Soft.TFrame", padding=(12, 10))
        activity.pack(fill="x", pady=(0, 11))
        activity.columnconfigure(1, weight=1)
        ttk.Label(activity, text="●", style="Soft.TLabel", foreground=GREEN).grid(row=0, column=0, sticky="n")
        self._wrapped_label(activity, textvariable=self.activity_var, style="Soft.TLabel").grid(
            row=0, column=1, sticky="ew", padx=(8, 0),
        )
        self._wrapped_label(monitor, textvariable=self.runtime_var, style="Muted.TLabel").pack(fill="x")
        self._wrapped_label(monitor, textvariable=self.metrics_var, style="Muted.TLabel").pack(fill="x", pady=(3, 10))
        ttk.Label(monitor, text="TERDENGAR", style="Muted.TLabel").pack(anchor="w")
        self._wrapped_label(
            monitor,
            textvariable=self.transcript_var,
            style="Card.TLabel",
            font=("Segoe UI Semibold", 11),
        ).pack(fill="x", pady=(3, 12))
        self.monitor_language_labels: list[ttk.Label] = []
        for index in range(3):
            block = ttk.Frame(monitor, style="Soft.TFrame", padding=(12, 9))
            block.pack(fill="x", pady=4)
            language_label = ttk.Label(
                block,
                text=f"TARGET {index + 1}",
                style="Soft.TLabel",
                foreground=CYAN,
                font=("Segoe UI Semibold", 8),
            )
            language_label.pack(anchor="w")
            self._wrapped_label(
                block,
                textvariable=self.translation_vars[index],
                style="Soft.TLabel",
            ).pack(fill="x", pady=(3, 0))
            self.monitor_language_labels.append(language_label)

        browser = self._card(right, "BROWSER SOURCES", "Server lokal tetap aktif saat caption berhenti")
        self.overlay_rows_container = ttk.Frame(browser, style="Inner.TFrame")
        self.overlay_rows_container.pack(fill="x")
        self.overlay_state_var = tk.StringVar(value="Menyiapkan server lokal…")
        self._wrapped_label(browser, textvariable=self.overlay_state_var, style="Muted.TLabel").pack(fill="x", pady=(9, 0))
        self._wrapped_label(
            browser,
            text="OBS Browser Source • 1920 × 300 • background transparan",
            style="Muted.TLabel",
        ).pack(fill="x", pady=(3, 0))

        self.footer = ttk.Frame(shell)
        self.footer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.save_button = ttk.Button(self.footer, text="Simpan", command=self.save)
        self.test_button = ttk.Button(self.footer, text="Tes Overlay", command=self.test_overlay)
        self.start_button = ttk.Button(
            self.footer,
            text="MULAI CAPTION",
            style="Accent.TButton",
            command=self.toggle,
        )
        self.footer.bind("<Configure>", self._resize_footer)

    @staticmethod
    def _wrapped_label(parent, **kwargs) -> ttk.Label:
        label = ttk.Label(parent, width=1, anchor="w", justify="left", **kwargs)
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(40, event.width)))
        return label

    def _resize_layout(self, event) -> None:
        self.canvas.itemconfigure(self._body_window, width=event.width)
        wide = event.width >= 1020
        if wide == self._wide_layout:
            return
        self._wide_layout = wide
        self.body.columnconfigure(0, weight=1, uniform="body" if wide else "")
        self.body.columnconfigure(1, weight=1 if wide else 0, uniform="body" if wide else "")
        self.left.grid(row=0, column=0, sticky="new", padx=(0, 8) if wide else 0)
        self.right.grid(row=0 if wide else 1, column=1 if wide else 0,
                        sticky="new", padx=(8, 0) if wide else 0)

    def _resize_footer(self, event) -> None:
        narrow = event.width < 650
        for column in range(3):
            inactive = narrow and column == 2
            self.footer.columnconfigure(column, weight=0 if inactive else 1, uniform="" if inactive else "actions")
        self.save_button.grid(row=1 if narrow else 0, column=0, sticky="ew", padx=(0, 4))
        self.test_button.grid(row=1 if narrow else 0, column=1, sticky="ew", padx=(4, 0) if narrow else 4)
        self.start_button.grid(row=0, column=0 if narrow else 2, columnspan=2 if narrow else 1,
                               sticky="ew", padx=0 if narrow else (4, 0), pady=(0, 8) if narrow else 0)

    def _on_key_tab(self, _event) -> None:
        focused = self.root.focus_get()
        if focused and isinstance(focused, tk.Widget):
            self.root.after_idle(lambda: self._reveal_focus(SimpleNamespace(widget=focused)))

    def _scroll_wheel(self, event):
        if self.body.winfo_height() > self.canvas.winfo_height() and getattr(event, "delta", 0):
            steps = max(1, abs(event.delta) // 120)
            self.canvas.yview_scroll(-steps * 3 if event.delta > 0 else steps * 3, "units")
        return "break"

    def _reveal_focus(self, event) -> None:
        widget = getattr(event, "widget", None)
        if not widget or not isinstance(widget, tk.Widget):
            return
        if not str(widget).startswith(str(self.body)):
            return
        try:
            widget_y = widget.winfo_rooty()
            canvas_y = self.canvas.winfo_rooty()
            canvas_h = self.canvas.winfo_height()
            widget_h = widget.winfo_height()
            if widget_y <= 0 or canvas_y <= 0 or canvas_h <= 10:
                return
            if canvas_y <= widget_y and (widget_y + widget_h) <= (canvas_y + canvas_h):
                return
            top = widget_y - canvas_y
            bottom = top + widget_h
            delta = top - 8 if top < 0 else bottom - canvas_h + 8
            target = (self.canvas.canvasy(0) + delta) / max(1, self.body.winfo_height())
            self.canvas.yview_moveto(max(0.0, min(1.0, target)))
        except Exception:
            pass

    def _card(self, parent, title: str, subtitle: str, collapsible: bool = False) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame", padding=(16, 14))
        outer.pack(fill="x", pady=(0, 12))
        heading = ttk.Frame(outer, style="Inner.TFrame")
        heading.pack(fill="x", pady=(0, 11))
        labels = ttk.Frame(heading, style="Inner.TFrame")
        labels.pack(side="left", fill="x", expand=True)
        ttk.Label(labels, text=title, style="CardTitle.TLabel").pack(anchor="w")
        self._wrapped_label(labels, text=subtitle, style="Muted.TLabel").pack(fill="x", pady=(2, 0))
        if collapsible:
            self.advanced_button = ttk.Button(heading, text="Buka", style="Link.TButton")
            self.advanced_button.pack(side="right")
        return outer

    def _field_row(self, parent, label: str, *, compact: bool = False) -> ttk.Frame:
        row = ttk.Frame(parent, style="Inner.TFrame")
        row.pack(fill="x", pady=4)
        row.columnconfigure(1, weight=1)
        width = 12 if compact else 19
        ttk.Label(row, text=label, style="Card.TLabel", width=width).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        return row

    def _compact_combo_field(
        self,
        parent,
        label: str,
        values: tuple[str, ...],
        variable: tk.StringVar,
    ) -> None:
        row = self._field_row(parent, label, compact=True)
        widget = ttk.Combobox(row, values=values, state="readonly", textvariable=variable, width=1)
        widget.grid(row=0, column=1, sticky="ew")
        self._track(widget, "readonly")

    def _track(self, widget: tk.Widget, idle_state: str = "normal") -> None:
        self._editable.append((widget, idle_state))

    def _toggle_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self._advanced_body.pack(fill="x")
            self.advanced_button.configure(text="Tutup")
        else:
            self._advanced_body.pack_forget()
            self.advanced_button.configure(text="Buka")

    def _load(self, config: AppConfig) -> None:
        self.source_var.set(source_display_name(config.source_language))
        for index, variable in enumerate(self.target_vars):
            variable.set(config.targets[index].language if index < len(config.targets) else UNUSED_TARGET)
        self.overlay_port_var.set(str(config.overlay.port))
        self.theme_var.set(config.overlay.theme)
        self.font_var.set(config.overlay.font_family)
        self.font_size_var.set(str(config.overlay.font_size))
        self.text_color_var.set(config.overlay.text_color)
        self.outline_color_var.set(config.overlay.outline_color)
        self.model_var.set(config.whisper_model)
        self.beam_var.set(str(config.whisper_beam_size))
        self.hotwords_var.set(config.whisper_hotwords)
        self.duration_var.set(str(config.max_utterance_seconds))
        self.stt_device_var.set(config.stt_device)
        self.mt_device_var.set(config.mt_device)
        self.vad_var.set(config.vad_threshold)
        self.silence_var.set(config.min_silence_ms)
        self.timeout_var.set(config.caption_timeout_seconds)
        self._update_monitor_languages()
        self._rebuild_overlay_rows()

    def refresh_microphones(self) -> None:
        try:
            entries = MicrophoneCapture.devices()
            self._devices = {item.name: item for item in entries}
            self.mic_box.configure(values=list(self._devices))
            configured = self.config.microphone_device
            selected = next(
                (item.name for item in entries if item.key == configured or item.index == configured),
                "",
            )
            if not selected and configured is not None and entries:
                resolved = MicrophoneCapture.resolve_device(configured)
                selected = next((item.name for item in entries if item.key == resolved.key), "")
            if selected:
                self.mic_var.set(selected)
            elif entries:
                self.mic_var.set(entries[0].name)
            else:
                self.mic_var.set("")
                self.audio_hint_var.set("Mikrofon tidak ditemukan. Periksa izin Windows.")
        except Exception as exc:
            self.audio_hint_var.set(f"Gagal membaca mikrofon: {exc}")
            self._set_activity("Mikrofon tidak tersedia", "error")

    def _collect(self) -> AppConfig:
        targets = selected_targets([variable.get() for variable in self.target_vars])
        source_choice = self.source_var.get().strip()
        if source_choice not in SOURCE_CHOICES:
            raise ValueError("Pilih bahasa ucapan dari daftar")
        selected = self._devices.get(self.mic_var.get())
        if selected is None:
            raise ValueError("Pilih mikrofon yang tersedia")
        source = "auto" if source_choice == "Auto-detect" else BY_NAME[source_choice].whisper
        return AppConfig(
            microphone_device=selected.key,
            source_language=source,
            targets=targets,
            overlay=OverlayConfig(
                host=self.config.overlay.host,
                port=int(self.overlay_port_var.get()),
                font_family=self.font_var.get(),
                font_size=int(self.font_size_var.get()),
                text_color=self.text_color_var.get().strip() or "#FFFFFF",
                outline_color=self.outline_color_var.get().strip() or "#806CFF",
                theme=self.theme_var.get().strip() or "vtuber",
            ),
            whisper_model=self.model_var.get(),
            whisper_beam_size=int(self.beam_var.get()),
            whisper_hotwords=self.hotwords_var.get().strip(),
            stt_device=self.stt_device_var.get(),
            mt_device=self.mt_device_var.get(),
            nllb_model=self.config.nllb_model,
            vad_threshold=round(self.vad_var.get(), 2),
            min_silence_ms=int(self.silence_var.get()),
            min_speech_ms=self.config.min_speech_ms,
            max_utterance_seconds=int(self.duration_var.get()),
            caption_timeout_seconds=int(self.timeout_var.get()),
        )

    def save(self, *, announce: bool = True) -> AppConfig | None:
        scroll_pos = self.canvas.yview()[0]
        try:
            config = self._collect()
            config.validate()
            self.config_store.save(config)
            self.config = config
            self._update_monitor_languages()
            self._rebuild_overlay_rows()
            self.root.after_idle(lambda: self.canvas.yview_moveto(scroll_pos))
            if not (self.pipeline and self.pipeline.running):
                self._sync_overlay(show_error=True)
            if announce:
                self._set_activity("Pengaturan tersimpan", "ok")
            return config
        except Exception as exc:
            messagebox.showerror("Pengaturan tidak valid", str(exc), parent=self.root)
            return None

    def _overlay_signature(self, config: AppConfig) -> tuple:
        return (
            config.overlay.host,
            config.overlay.port,
            config.overlay.font_family,
            config.overlay.font_size,
            config.overlay.text_color,
            config.overlay.outline_color,
            config.overlay.theme,
            tuple(target.language for target in config.targets),
            config.caption_timeout_seconds,
        )

    def _sync_overlay(self, *, show_error: bool = True) -> bool:
        signature = self._overlay_signature(self.config)
        if (
            self.overlay_service
            and self.overlay_service.running
            and signature == self._overlay_fingerprint
        ):
            return True
        if self.overlay_service:
            self.overlay_service.stop()
        self.overlay_service = OverlayService(
            self.config.overlay,
            self.config.targets,
            self.app_dir / "output" / "overlay.html",
            self.config.caption_timeout_seconds,
            lambda kind, message: self.events.put(PipelineEvent(kind, message)),
        )
        try:
            self.overlay_service.start()
            self._overlay_fingerprint = signature
            self.overlay_state_var.set(f"Aktif lokal • 127.0.0.1:{self.config.overlay.port}")
            return True
        except Exception as exc:
            self.overlay_state_var.set(f"Server gagal • {exc}")
            self._overlay_fingerprint = None
            if show_error:
                messagebox.showerror("Overlay tidak dapat dimulai", str(exc), parent=self.root)
            return False

    def toggle(self) -> None:
        if self._stopping:
            return
        if self.pipeline and (self.pipeline.running or self.pipeline.inference_busy):
            self.pipeline.request_stop()
            self._stopping = True
            self._closing_since = time.monotonic()
            self._set_status("MENUTUP", AMBER)
            self._set_activity("Capture dihentikan; menutup mesin caption…", "warning")
            self.start_button.configure(state="disabled", text="MENUTUP MESIN")
            return
        config = self.save(announce=False)
        if not config or not self._sync_overlay(show_error=True):
            return
        self._reset_audio()
        self._pipeline_error = None
        self._processing = False
        self._closing_since = 0.0
        self.transcript_var.set("Hasil tampil setelah ucapan selesai.")
        self.runtime_var.set("Menyiapkan model lokal…")
        self.metrics_var.set("Waktu STT / MT tampil setelah caption pertama")
        self.audio_hint_var.set("Mikrofon dibuka setelah model siap.")
        self._last_translations.clear()
        self._show_translations({})
        self.pipeline = CaptionPipeline(
            config,
            self.app_dir,
            self.events.put,
            overlay_service=self.overlay_service,
        )
        self.pipeline.start()
        self._set_live_controls(True)
        self._set_status("MENYIAPKAN", CYAN)
        self._set_activity("Memuat dan menguji model sebelum membuka mikrofon…", "working")

    def _poll_events(self) -> None:
        if self._closed:
            return
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        if self._stopping:
            self._finish_stop()
        self.root.after(80, self._poll_events)

    def _finish_stop(self) -> None:
        if self.pipeline and (self.pipeline.running or self.pipeline.inference_busy):
            self.start_button.configure(state="disabled", text="MENUTUP MESIN")
            if not self._closing_since:
                self._closing_since = time.monotonic()
            if time.monotonic() - self._closing_since >= 10:
                self.runtime_var.set(
                    "Panggilan model native belum selesai dan tidak bisa dipaksa batal. "
                    "Jika tetap macet, tutup aplikasi lalu jalankan start.bat kembali."
                )
            return
        self._stopping = False
        self._processing = False
        self._set_status("ERROR" if self._pipeline_error else "SIAP", RED if self._pipeline_error else GREEN)
        self._set_activity("Caption berhenti • Browser Source tetap aktif", "idle")
        self.runtime_var.set("Mesin berhenti • siap memulai sesi baru")
        self._set_live_controls(False)

    def _handle_event(self, event: PipelineEvent) -> None:
        data = event.data or {}
        if event.kind == "audio_level":
            self._update_audio(float(data.get("rms", 0.0)), float(data.get("peak", 0.0)))
        elif event.kind == "vad_probability":
            self._vad_probability = float(data.get("probability", 0.0))
            self._vad_speaking = bool(data.get("speaking", False))
            self.vad_status_var.set(f"{self._vad_probability:.0%}")
            if self._vad_probability >= self.config.vad_threshold:
                self._last_speech_at = time.monotonic()
        elif event.kind == "preparing":
            if not self._stopping:
                self._set_status("MENYIAPKAN", CYAN)
                self.runtime_var.set(event.message)
                self._set_activity(event.message, "working")
        elif event.kind == "model_ready":
            self.runtime_var.set(
                f"Whisper {data['model']} • STT {data['stt_device'].upper()} • MT {data['mt_device'].upper()}"
            )
        elif event.kind == "metrics":
            self.metrics_var.set(
                f"STT {data['stt_ms']:.0f} ms • MT {data['mt_ms']:.0f} ms • "
                f"setelah keputusan VAD {data['after_vad_ms']:.0f} ms (di luar jeda hening)"
            )
        elif event.kind == "started":
            self._set_status("LIVE", GREEN)
            self._set_activity(event.message, "ok")
        elif event.kind == "listening":
            self._set_status("LIVE", GREEN)
            self._set_activity(event.message, "ok")
            if data.get("device"):
                self.audio_device_var.set(
                    f"Aktif • {data['device']} • {data.get('sample_rate', 16000)} Hz"
                )
        elif event.kind == "speech":
            self._last_speech_at = time.monotonic()
            if not self._processing:
                self._set_activity("Suara terdeteksi • hasil setelah jeda bicara", "working")
        elif event.kind in {"queued", "processing"}:
            self._processing = True
            self._set_activity(event.message, "working")
        elif event.kind == "transcript":
            self.transcript_var.set(str(data.get("text", event.message)))
            self._set_activity("Ucapan dikenali • menerjemahkan…", "working")
        elif event.kind == "translations":
            self._last_translations = {str(key): str(value) for key, value in data.items()}
            self._show_translations(self._last_translations)
            self._set_activity("Terjemahan siap • mengirim overlay…", "working")
        elif event.kind == "published":
            self._processing = False
            self._set_activity("Caption terkirim ke Browser Source", "ok")
        elif event.kind == "overlay_ready":
            self.overlay_state_var.set(f"Aktif lokal • 127.0.0.1:{self.config.overlay.port}")
        elif event.kind == "stopped":
            self._reset_audio()
            self.audio_device_var.set("Capture berhenti")
            self.audio_hint_var.set("Mulai caption untuk memeriksa sinyal lagi.")
            self._stopping = True
            if not self._pipeline_error:
                self._set_status("MENUTUP", AMBER)
            self._finish_stop()
        elif event.kind == "inference_idle":
            if self._stopping:
                self._finish_stop()
        elif event.kind == "error":
            self._pipeline_error = event.message
            self._set_status("ERROR", RED)
            self._set_activity(event.message, "error")
        elif event.kind == "warning":
            self._set_activity(event.message, "warning")
        elif event.kind == "background":
            self._set_activity(event.message, "warning")

    def _reset_audio(self) -> None:
        self._level_history.clear()
        self._smoothed_db = None
        self._last_speech_at = 0.0
        self._vad_probability = 0.0
        self._vad_speaking = False
        self.audio_meter["value"] = 0
        self.rms_var.set("— dBFS")
        self.peak_var.set("— dBFS")
        self.vad_status_var.set("— %")

    def _update_audio(self, rms: float, peak: float) -> None:
        now = time.monotonic()
        db = 20 * math.log10(max(rms, 0.000001))
        peak_db = 20 * math.log10(max(peak, 0.000001))
        previous = self._smoothed_db
        alpha = 0.65 if previous is None or db > previous else 0.22
        self._smoothed_db = db if previous is None else previous + alpha * (db - previous)
        self.audio_meter["value"] = min(100, max(0, (self._smoothed_db + 60) / 60 * 100))
        self.rms_var.set(f"{self._smoothed_db:.0f} dBFS")
        self.peak_var.set(f"{peak_db:.0f} dBFS")
        self._level_history.append((now, db))
        while self._level_history and now - self._level_history[0][0] > 4:
            self._level_history.popleft()
        levels = [value for _, value in self._level_history]
        spread = max(levels) - min(levels)
        if peak >= 0.98:
            hint = "Sinyal clipping • turunkan gain mikrofon."
        elif self._vad_speaking or now - self._last_speech_at < 1.5:
            hint = "Ucapan terdeteksi • beri jeda agar caption diproses."
        elif db < -65 and peak_db < -55:
            hint = "Sinyal sangat kecil • periksa mute, gain, atau pilih mic fisik."
        elif now - self._level_history[0][0] >= 2 and spread < 4:
            hint = "Sinyal tetap, belum ada ucapan. Jika sedang bicara: cek routing Sonar atau pilih mic fisik."
        elif spread >= 6:
            hint = "Level berubah, belum lolos VAD. Dekatkan mic; periksa ambang VAD bila suara jelas."
        else:
            hint = "Audio masuk, menunggu ucapan • RMS/peak bukan bukti suara bicara."
        self.audio_hint_var.set(hint)

    def _show_translations(self, translations: dict[str, str]) -> None:
        for index in range(3):
            if index >= len(self.config.targets):
                self.translation_vars[index].set("Tidak digunakan")
                continue
            language = self.config.targets[index].language
            self.translation_vars[index].set(translations.get(language, "Menunggu caption…"))

    def _update_monitor_languages(self) -> None:
        for index, label in enumerate(self.monitor_language_labels):
            name = (
                self.config.targets[index].language
                if index < len(self.config.targets)
                else f"Target {index + 1}"
            )
            label.configure(text=name.upper())
        self._show_translations(self._last_translations)

    def _set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    def _set_activity(self, message: str, _level: str) -> None:
        self.activity_var.set(self._pipeline_error or message)

    def _set_live_controls(self, live: bool) -> None:
        for widget, idle_state in self._editable:
            widget.configure(state="disabled" if live else idle_state)
        self.save_button.configure(state="disabled" if live else "normal")
        self.test_button.configure(state="disabled" if live else "normal")
        self.start_button.configure(
            text="HENTIKAN CAPTION" if live else "MULAI CAPTION",
            style="Danger.TButton" if live else "Accent.TButton",
            state="normal",
        )

    def _rebuild_overlay_rows(self) -> None:
        if not hasattr(self, "overlay_rows_container"):
            return
        for child in self.overlay_rows_container.winfo_children():
            child.destroy()

        all_row = ttk.Frame(self.overlay_rows_container, style="Soft.TFrame", padding=(11, 8))
        all_row.pack(fill="x", pady=3)
        all_row.columnconfigure(0, weight=1)
        self._wrapped_label(
            all_row, text="★ Multi-Bahasa Sekaligus (All-in-One)", style="Soft.TLabel",
            foreground=CYAN, font=("Segoe UI Semibold", 9),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            all_row, text="Copy", command=lambda: self.copy_url("all"),
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            all_row, text="Preview", command=lambda: self.open_preview("all"),
        ).grid(row=0, column=2)
        self._wrapped_label(
            all_row, text=self.overlay_url("all"), style="Soft.TLabel",
            foreground=MUTED, font=("Consolas", 8),
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))

        for target in self.config.targets:
            row = ttk.Frame(self.overlay_rows_container, style="Soft.TFrame", padding=(11, 8))
            row.pack(fill="x", pady=3)
            row.columnconfigure(0, weight=1)
            self._wrapped_label(
                row, text=target.language, style="Soft.TLabel", font=("Segoe UI Semibold", 9),
            ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
            ttk.Button(
                row, text="Copy", command=lambda language=target.language: self.copy_url(language),
            ).grid(row=0, column=1, padx=(0, 6))
            ttk.Button(
                row, text="Preview", command=lambda language=target.language: self.open_preview(language),
            ).grid(row=0, column=2)
            self._wrapped_label(
                row, text=self.overlay_url(target.language), style="Soft.TLabel",
                foreground=MUTED, font=("Consolas", 8),
            ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))

    def overlay_url(self, language: str) -> str:
        return overlay_url(self.config.overlay, language)

    def copy_url(self, language: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.overlay_url(language))
        self.root.update_idletasks()
        label = "Multi-Bahasa (All)" if language == "all" else language
        self._set_activity(f"URL {label} disalin", "ok")

    def open_preview(self, language: str) -> None:
        if self._sync_overlay(show_error=True) and self.overlay_service:
            webbrowser.open(self.overlay_service.preview_url(language))

    def test_overlay(self) -> None:
        config = self.save(announce=False)
        if not config or not self._sync_overlay(show_error=True) or not self.overlay_service:
            return
        samples = {
            target.language: f"{target.language} • LumaCaption siap tampil di OBS"
            for target in config.targets
        }
        try:
            self.overlay_service.publish_now(samples)
            self._last_translations = samples
            self._show_translations(samples)
            preview_target = "all" if len(config.targets) > 1 else config.targets[0].language
            webbrowser.open(self.overlay_service.preview_url(preview_target))
            self._set_activity("Caption tes terkirim • preview dibuka", "ok")
        except Exception as exc:
            messagebox.showerror("Tes overlay gagal", str(exc), parent=self.root)

    def close(self) -> None:
        self._closed = True
        if self.pipeline:
            self.pipeline.request_stop()
        if self.overlay_service:
            self.overlay_service.stop(timeout=2.0)
        self.root.destroy()
