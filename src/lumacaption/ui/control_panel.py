from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from types import SimpleNamespace
from typing import Any
import webbrowser
from lumacaption import CREDIT, __version__
from lumacaption.audio.capture import InputDevice, MicrophoneCapture
from lumacaption.config import (
    AppConfig, CAPTION_FONTS, CAPTION_FONT_MIN_SIZE, CAPTION_FONT_MAX_SIZE,
    CAPTION_THEMES, CAPTION_TEXT_COLORS, CAPTION_OUTLINE_COLORS,
    COMPUTE_TYPES, ConfigStore, MODEL_SIZES, OverlayConfig, TargetConfig,
)
from lumacaption.languages import BY_NAME, SOURCE_CHOICES, TARGET_CHOICES, source_display_name
from lumacaption.output.overlay_server import OverlayService, overlay_url, static_overlay_url
from lumacaption.output.styles import ANCHORS, CaptionStyle
from lumacaption.pipeline import CaptionPipeline, PipelineEvent
from lumacaption.model_manager import (
    CATALOG, inspect_model, cache_for, ensure_model, resource_details, hardware_info, evaluate_vram_safety,
    model_resource_table_rows, detect_and_link_models, find_model_locally, adopt_model,
)
from lumacaption.ui.caption_editor import CaptionEditor
from lumacaption.vocabulary import VocabularyManager
from lumacaption.i18n import t, DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES


PALETTES = {
    "dark": {
        "bg": "#080B14",
        "surface": "#0E1322",
        "surface_2": "#141B2E",
        "surface_3": "#1A2238",
        "border": "#252F49",
        "text": "#F4F6FF",
        "muted": "#8C96B4",
        "subtle": "#626D8B",
        "violet": "#806CFF",
        "violet_hover": "#9585FF",
        "cyan": "#4DD8E7",
        "green": "#58D6A8",
        "amber": "#F0BA66",
        "red": "#FF718D",
        "table_even": "#0E1322",
        "table_odd": "#141B2E",
        "btn_bg": "#1A2238",
        "btn_active": "#253152",
    },
    "light": {
        "bg": "#F1F4F9",
        "surface": "#FFFFFF",
        "surface_2": "#E5EAF2",
        "surface_3": "#D4DCE8",
        "border": "#BAC7D8",
        "text": "#111827",
        "muted": "#4B5563",
        "subtle": "#6B7280",
        "violet": "#6366F1",
        "violet_hover": "#4F46E5",
        "cyan": "#0891B2",
        "green": "#059669",
        "amber": "#D97706",
        "red": "#E11D48",
        "table_even": "#FFFFFF",
        "table_odd": "#F8FAFC",
        "btn_bg": "#E5EAF2",
        "btn_active": "#D4DCE8",
    },
}

# Module-level defaults for backward compatibility
BG = PALETTES["dark"]["bg"]
SURFACE = PALETTES["dark"]["surface"]
SURFACE_2 = PALETTES["dark"]["surface_2"]
SURFACE_3 = PALETTES["dark"]["surface_3"]
BORDER = PALETTES["dark"]["border"]
TEXT = PALETTES["dark"]["text"]
MUTED = PALETTES["dark"]["muted"]
SUBTLE = PALETTES["dark"]["subtle"]
VIOLET = PALETTES["dark"]["violet"]
VIOLET_HOVER = PALETTES["dark"]["violet_hover"]
CYAN = PALETTES["dark"]["cyan"]
GREEN = PALETTES["dark"]["green"]
AMBER = PALETTES["dark"]["amber"]
RED = PALETTES["dark"]["red"]
UNUSED_TARGET = "Tidak digunakan"


def selected_targets(choices: list[str]) -> list[TargetConfig]:
    if not choices or choices[0].strip() not in TARGET_CHOICES:
        raise ValueError("Target 1 wajib diisi dengan bahasa dari daftar")
    targets = []
    for slot_idx, choice in enumerate(choices, 1):
        name = choice.strip()
        if name == UNUSED_TARGET:
            continue
        if name not in TARGET_CHOICES:
            raise ValueError("Pilih bahasa dari daftar atau 'Tidak digunakan'")
        targets.append(TargetConfig(name, profile=slot_idx))
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
        self.vocab_manager = VocabularyManager(self.app_dir)
        self.vocab_status_var = tk.StringVar(value=self.t("vocab_learned_status", count=self.vocab_manager.learned_count))
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
        self._wide_engine_layout: bool | None = None
        self.tab_models: ttk.Frame | None = None
        self._level_history: deque[tuple[float, float]] = deque(maxlen=64)
        self._smoothed_db: float | None = None
        self._last_speech_at = 0.0
        self._vad_probability = 0.0
        self._vad_speaking = False
        self._pipeline_error: str | None = None
        self._stopping = False
        self._closing_since = 0.0
        self._processing = False
        self._closed = False
        self._poll_timer: str | None = None
        self._scroll_idle_id: str | None = None
        self._focus_idle_id: str | None = None

        self._theme_name = getattr(self.config, "ui_theme", "dark")
        if self._theme_name not in PALETTES:
            self._theme_name = "dark"
        self.colors = PALETTES[self._theme_name]

        self._build_style()
        self._build_ui()
        self._load(config)
        self.refresh_microphones()
        self._overlay_timer: str | None = self.root.after(50, self._deferred_sync_overlay)
        self._poll_timer = self.root.after(80, self._poll_events)
        self.root.bind("<Destroy>", self._on_root_destroy, add="+")

    def _build_style(self) -> None:
        self.root.title("KizCaption — Offline OBS Multilingual Companion")
        self.root.geometry("1180x820")
        self.root.minsize(640, 580)
        self.root.option_add("*Font", "{Segoe UI} 10")
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "kzp_icon.png"
        if icon_path.is_file():
            try:
                self._app_icon = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(False, self._app_icon)
            except Exception:
                pass
        for widget_class in ("TCombobox", "TSpinbox", "Combobox", "Spinbox"):
            self.root.bind_class(widget_class, "<MouseWheel>", lambda e: "break")
        self._apply_theme(self._theme_name)

    def _apply_theme(self, mode: str) -> None:
        self._theme_name = mode
        self.colors = PALETTES.get(mode, PALETTES["dark"])
        c = self.colors

        self.root.configure(bg=c["bg"])
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.configure(bg=c["bg"])
        if hasattr(self, "engine_canvas") and self.engine_canvas:
            self.engine_canvas.configure(bg=c["bg"])

        self.root.option_add("*TCombobox*Listbox.background", c["surface_2"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["violet"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=c["bg"], foreground=c["text"], bordercolor=c["border"], darkcolor=c["bg"], lightcolor=c["bg"])
        style.configure("TFrame", background=c["bg"])
        style.configure("Card.TFrame", background=c["surface"], borderwidth=1, relief="solid")
        style.configure("Inner.TFrame", background=c["surface"])
        style.configure("Soft.TFrame", background=c["surface_2"])
        style.configure("TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=c["surface"], foreground=c["text"])
        style.configure("Soft.TLabel", background=c["surface_2"], foreground=c["text"])
        style.configure("Title.TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI Semibold", 22))
        style.configure("Kicker.TLabel", background=c["bg"], foreground=c["cyan"], font=("Segoe UI Semibold", 9))
        style.configure("CardTitle.TLabel", background=c["surface"], foreground=c["text"], font=("Segoe UI Semibold", 12))
        style.configure("Muted.TLabel", background=c["surface"], foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("PageMuted.TLabel", background=c["bg"], foreground=c["muted"], font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=c["surface_2"], foreground=c["green"], font=("Segoe UI Semibold", 9), padding=(12, 6))

        style.configure("TButton", background=c["btn_bg"], foreground=c["text"], borderwidth=0, padding=(12, 8), font=("Segoe UI Semibold", 9))
        style.map("TButton", background=[("active", c["btn_active"]), ("disabled", c["surface_2"])], foreground=[("disabled", c["subtle"])])
        style.configure("Accent.TButton", background=c["violet"], foreground="#FFFFFF", padding=(18, 11), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", c["violet_hover"]), ("disabled", c["surface_3"])])
        style.configure("Danger.TButton", background=c["red"], foreground="#FFFFFF", padding=(18, 11), font=("Segoe UI Semibold", 10))
        style.map("Danger.TButton", background=[("active", "#B91C1C")])
        style.configure("Link.TButton", background=c["surface"], foreground=c["cyan"], padding=(8, 5))
        style.map("Link.TButton", background=[("active", c["surface_2"])])

        style.configure("TEntry", fieldbackground=c["surface_2"], foreground=c["text"], insertcolor=c["text"], bordercolor=c["border"], padding=7)
        style.configure("TCombobox", fieldbackground=c["surface_2"], background=c["surface_3"], foreground=c["text"], arrowcolor=c["muted"], bordercolor=c["border"], padding=7)
        style.map("TCombobox", fieldbackground=[("readonly", c["surface_2"]), ("disabled", c["surface"])], foreground=[("disabled", c["subtle"])])
        style.configure("TSpinbox", fieldbackground=c["surface_2"], foreground=c["text"], arrowcolor=c["muted"], bordercolor=c["border"], padding=7)
        style.configure("Horizontal.TScale", background=c["surface"], troughcolor=c["surface_3"])
        style.configure("Audio.Horizontal.TProgressbar", troughcolor=c["surface_3"], background=c["cyan"], bordercolor=c["surface_3"], lightcolor=c["cyan"], darkcolor=c["cyan"])
        style.configure("TSeparator", background=c["border"])

        style.configure("TNotebook", background=c["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["surface_2"], foreground=c["muted"], padding=(14, 7), font=("Segoe UI Semibold", 9))
        style.map("TNotebook.Tab", background=[("selected", c["violet"]), ("active", c["surface_3"])], foreground=[("selected", "#FFFFFF"), ("active", c["text"])])

        # Modern high-contrast Treeview styling
        style.configure(
            "Treeview",
            background=c["surface"],
            foreground=c["text"],
            fieldbackground=c["surface"],
            bordercolor=c["border"],
            rowheight=30,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=c["surface_3"],
            foreground=c["text"],
            bordercolor=c["border"],
            font=("Segoe UI Semibold", 9),
            padding=(6, 5),
        )
        style.map(
            "Treeview",
            background=[("selected", c["violet"])],
            foreground=[("selected", "#FFFFFF")],
        )
        style.map(
            "Treeview.Heading",
            background=[("active", c["surface_2"])],
        )

        # Modern minimalist scrollbar styling (no legacy arrow buttons, sleek rounded thumb)
        sb_layout = [
            ("Vertical.Scrollbar.trough", {
                "sticky": "ns",
                "children": [
                    ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})
                ],
            })
        ]
        style.layout("Vertical.TScrollbar", sb_layout)
        style.layout("Card.Vertical.TScrollbar", sb_layout)

        thumb_bg = "#475569" if mode == "dark" else "#94A3B8"
        thumb_hover = c["violet"]
        thumb_pressed = c["violet_hover"]
        trough_bg = "#0B0F19" if mode == "dark" else "#E2E8F0"
        trough_card = c["surface_2"]

        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=thumb_bg,
            darkcolor=thumb_bg,
            lightcolor=thumb_bg,
            troughcolor=trough_bg,
            bordercolor=thumb_bg,
            arrowsize=0,
            width=14,
            relief="flat",
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("pressed", thumb_pressed), ("active", thumb_hover)],
            darkcolor=[("pressed", thumb_pressed), ("active", thumb_hover)],
            lightcolor=[("pressed", thumb_pressed), ("active", thumb_hover)],
        )

        style.configure(
            "Card.Vertical.TScrollbar",
            gripcount=0,
            background=thumb_bg,
            darkcolor=thumb_bg,
            lightcolor=thumb_bg,
            troughcolor=trough_card,
            bordercolor=thumb_bg,
            arrowsize=0,
            width=14,
            relief="flat",
        )
        style.map(
            "Card.Vertical.TScrollbar",
            background=[("pressed", thumb_pressed), ("active", thumb_hover)],
            darkcolor=[("pressed", thumb_pressed), ("active", thumb_hover)],
            lightcolor=[("pressed", thumb_pressed), ("active", thumb_hover)],
        )

        # Template preset buttons & Mode switcher button styles
        style.configure("PresetActiveLow.TButton", background=c["green"], foreground="#FFFFFF", font=("Segoe UI Semibold", 9))
        style.map("PresetActiveLow.TButton", background=[("active", c["green"])])

        style.configure("PresetActiveMed.TButton", background=c["violet"], foreground="#FFFFFF", font=("Segoe UI Semibold", 9))
        style.map("PresetActiveMed.TButton", background=[("active", c["violet_hover"])])

        style.configure("PresetActiveHigh.TButton", background="#9333EA", foreground="#FFFFFF", font=("Segoe UI Semibold", 9))
        style.map("PresetActiveHigh.TButton", background=[("active", "#A855F7")])

        style.configure("ModeActive.TButton", background=c["violet"], foreground="#FFFFFF", font=("Segoe UI Semibold", 9))
        style.map("ModeActive.TButton", background=[("active", c["violet_hover"])])

        if hasattr(self, "theme_btn") and self.theme_btn:
            self.theme_btn.configure(text="Mode Terang" if mode == "dark" else "Mode Gelap")

        if hasattr(self, "models_canvas") and self.models_canvas:
            self.models_canvas.configure(bg=c["bg"])

        if hasattr(self, "engine_canvas") and self.engine_canvas:
            self.engine_canvas.configure(bg=c["bg"])

        if hasattr(self, "vocab_tree") and self.vocab_tree:
            self.vocab_tree.tag_configure("even", background=c["table_even"], foreground=c["text"])
            self.vocab_tree.tag_configure("odd", background=c["table_odd"], foreground=c["text"])

        if hasattr(self, "res_tree") and self.res_tree:
            self.res_tree.tag_configure("even", background=c["table_even"], foreground=c["text"])
            self.res_tree.tag_configure("odd", background=c["table_odd"], foreground=c["text"])

        if hasattr(self, "_refresh_dtln_status"):
            self._refresh_dtln_status()

    def t(self, key: str, **kwargs: Any) -> str:
        lang = getattr(self.config, "ui_language", DEFAULT_LANGUAGE)
        return t(key, lang=lang, **kwargs)

    def toggle_ui_language(self) -> None:
        curr = getattr(self.config, "ui_language", "id")
        nxt = "en" if curr == "id" else "id"
        self.set_ui_language(nxt)

    def set_ui_language(self, lang: str) -> None:
        if lang not in SUPPORTED_LANGUAGES:
            return
        self.config.ui_language = lang
        try:
            self.config_store.save(self.config)
        except Exception:
            pass
        self._reload_ui()

    def _reload_ui(self) -> None:
        selected_idx = 0
        if hasattr(self, "notebook") and self.notebook.winfo_exists():
            try:
                selected_idx = self.notebook.index(self.notebook.select())
            except Exception:
                selected_idx = 0

        self._wide_layout = None
        self._wide_engine_layout = None
        self._build_ui()
        self._load(self.config)
        self.refresh_microphones()
        if hasattr(self, "notebook") and 0 <= selected_idx < len(self.notebook.tabs()):
            try:
                self.notebook.select(selected_idx)
            except Exception:
                pass
        self._sync_mode_ui()
        if hasattr(self, "_refresh_dtln_status"):
            self._refresh_dtln_status()
        if hasattr(self, "_on_model_card_selected"):
            self._on_model_card_selected()
        if hasattr(self, "_update_vocab_status"):
            self._update_vocab_status()

        self.root.update_idletasks()
        self._apply_layout()
        self._apply_engine_layout()

    def toggle_theme(self) -> None:
        new_mode = "light" if self._theme_name == "dark" else "dark"
        self.config.ui_theme = new_mode
        self._apply_theme(new_mode)
        try:
            self.config_store.save(self.config)
        except Exception:
            pass

    def _build_ui(self) -> None:
        c = self.colors
        if hasattr(self, "shell") and self.shell and self.shell.winfo_exists():
            self.shell.destroy()
        self.shell = shell = ttk.Frame(self.root, padding=(20, 16))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)

        brand = ttk.Frame(header)
        brand.grid(row=0, column=0, sticky="w")
        ttk.Label(brand, text=self.t("app_kicker"), style="Kicker.TLabel").pack(anchor="w")
        ttk.Label(brand, text="KizCaption", style="Title.TLabel").pack(anchor="w")
        self._wrapped_label(
            brand,
            text=self.t("app_subtitle"),
            style="PageMuted.TLabel",
        ).pack(fill="x", pady=(2, 0))

        header_actions = ttk.Frame(header)
        header_actions.grid(row=0, column=1, sticky="ne", padx=(12, 0), pady=6)

        lang_code = getattr(self.config, "ui_language", "id")
        self.lang_btn = ttk.Button(
            header_actions,
            text="🌐 English" if lang_code == "id" else "🌐 Bahasa Indonesia",
            command=self.toggle_ui_language,
        )
        self.lang_btn.pack(side="left", padx=(0, 8))

        self.theme_btn = ttk.Button(
            header_actions,
            text=self.t("btn_light_mode") if self._theme_name == "dark" else self.t("btn_dark_mode"),
            command=self.toggle_theme,
        )
        self.theme_btn.pack(side="left", padx=(0, 8))

        self.status_var = tk.StringVar(value=self.t("status_ready"))
        self.status_label = ttk.Label(header_actions, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side="left")

        viewport = ttk.Frame(shell)
        viewport.grid(row=1, column=0, sticky="nsew")
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(viewport)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tab 1: Dashboard / Monitor
        tab_monitor = ttk.Frame(self.notebook)
        self.notebook.add(tab_monitor, text=self.t("tab_monitor"))
        tab_monitor.columnconfigure(0, weight=1)
        tab_monitor.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(tab_monitor, bg=c["bg"], highlightthickness=0, yscrollincrement=20)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.main_scrollbar = ttk.Scrollbar(tab_monitor, orient="vertical", command=self.canvas.yview, style="Vertical.TScrollbar")
        self.main_scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))

        def _on_canvas_yview(first, last):
            self.main_scrollbar.set(first, last)
            self._update_scroll_hint(float(first), float(last))

        self.canvas.configure(yscrollcommand=_on_canvas_yview)

        # Dynamic scroll hint bar at bottom of tab_monitor
        self.scroll_hint_frame = ttk.Frame(tab_monitor, style="Soft.TFrame", padding=(8, 5))
        self.scroll_hint_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.scroll_hint_frame.columnconfigure(0, weight=1)
        self.scroll_hint_label = ttk.Label(
            self.scroll_hint_frame,
            text=self.t("scroll_hint_more"),
            style="Soft.TLabel",
            foreground=c["cyan"],
            font=("Segoe UI Semibold", 9),
            anchor="center",
        )
        self.scroll_hint_label.grid(row=0, column=0, sticky="ew")

        self.body = ttk.Frame(self.canvas)
        self._body_window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.left = left = ttk.Frame(self.body)
        self.right = right = ttk.Frame(self.body)
        self._apply_layout()
        self.body.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._resize_layout)
        self.root.bind("<MouseWheel>", self._scroll_wheel, add=True)
        self.root.bind("<Key-Tab>", self._on_key_tab, add=True)
        self.root.bind("<Shift-Key-Tab>", self._on_key_tab, add=True)

        self.ui_mode_var = tk.StringVar(value=getattr(self.config, "ui_mode", "ez"))
        self.resource_preset_var = tk.StringVar(value=getattr(self.config, "resource_preset", "medium"))
        self.preset_badge_var = tk.StringVar(value=self.t("preset_badge_med"))
        self.preset_desc_var = tk.StringVar(value=self.t("preset_desc_med"))
        self._preset_internal_change = False

        self.mic_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.beam_var = tk.StringVar()
        self.hotwords_var = tk.StringVar()
        self.duration_var = tk.StringVar()
        self.runtime_var = tk.StringVar(value=self.t("runtime_not_ready"))
        self.metrics_var = tk.StringVar(value=self.t("metrics_placeholder"))
        self.stt_device_var = tk.StringVar()
        self.stt_compute_type_var = tk.StringVar(value=getattr(self.config, "stt_compute_type", "auto"))
        self.mt_device_var = tk.StringVar()
        self.mt_compute_type_var = tk.StringVar(value=getattr(self.config, "mt_compute_type", "auto"))
        self.mt_beam_size_var = tk.StringVar(value=str(self.config.mt_beam_size))
        self.cpu_threads_var = tk.StringVar(value=str(self.config.cpu_threads))
        self.audio_channel_var = tk.StringVar(value=self.config.audio_channel)
        self.audio_gain_var = tk.DoubleVar(value=self.config.audio_gain_db)
        self.normalize_audio_var = tk.BooleanVar(value=self.config.normalize_audio)
        self.audio_clarity_var = tk.BooleanVar(value=getattr(self.config, "audio_clarity", True))
        self.slang_normalization_var = tk.BooleanVar(value=getattr(self.config, "slang_normalization", True))
        self.profanity_filter_var = tk.BooleanVar(value=getattr(self.config, "profanity_filter", False))
        self.denoise_engine_var = tk.StringVar(value=getattr(self.config, "denoise_engine", "clarity"))
        self.target_vars = [tk.StringVar() for _ in range(3)]
        self.overlay_port_var = tk.StringVar()
        self.overlay_state_var = tk.StringVar(value=self.t("obs_server_preparing"))
        self.overlay_clients_var = tk.StringVar(value=self.t("obs_clients_count_fmt", count=0))
        self.overlay_activity_var = tk.StringVar(value=self.t("obs_no_caption_sent"))
        self.theme_var = tk.StringVar()
        self.font_var = tk.StringVar()
        self.font_size_var = tk.StringVar()
        self.text_color_var = tk.StringVar()
        self.outline_color_var = tk.StringVar()
        self.vad_var = tk.DoubleVar()
        self.silence_var = tk.IntVar()
        self.timeout_var = tk.IntVar()
        self.audio_device_var = tk.StringVar(value=self.t("audio_capture_inactive"))
        self.audio_hint_var = tk.StringVar(value=self.t("audio_hint_select_mic"))
        self.rms_var = tk.StringVar(value="— dBFS")
        self.peak_var = tk.StringVar(value="— dBFS")
        self.vad_status_var = tk.StringVar(value="— %")
        self.activity_var = tk.StringVar(value=self.t("activity_ready_session"))

        # Mode Switcher Bar (EZ Mode vs Advanced Mode)
        mode_card = ttk.Frame(left, style="Card.TFrame", padding=(10, 8))
        mode_card.pack(fill="x", pady=(0, 6))
        mode_card.columnconfigure(0, weight=1)
        mode_card.columnconfigure(1, weight=1)

        self.btn_mode_ez = ttk.Button(
            mode_card,
            text=f"🚀 {self.t('mode_ez')}",
            command=lambda: self.set_ui_mode("ez"),
        )
        self.btn_mode_ez.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_mode_adv = ttk.Button(
            mode_card,
            text=f"⚙️ {self.t('mode_adv')}",
            command=lambda: self.set_ui_mode("advanced"),
        )
        self.btn_mode_adv.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # EZ Template Setting Card
        self.ez_template_card = self._card(
            left,
            self.t("card_template_title"),
            self.t("card_template_sub"),
        )

        source_row = self._field_row(self.ez_template_card, self.t("label_speech_lang"))
        self.source_box = FilterCombobox(source_row, choices=SOURCE_CHOICES, textvariable=self.source_var)
        self.source_box.grid(row=0, column=1, sticky="ew")
        self._track(self.source_box)

        # Performance template buttons
        perf_label_row = ttk.Frame(self.ez_template_card, style="Inner.TFrame")
        perf_label_row.pack(fill="x", pady=(8, 4))
        ttk.Label(perf_label_row, text=self.t("label_pc_load"), style="Card.TLabel", font=("Segoe UI Semibold", 9)).pack(anchor="w")

        preset_btn_grid = ttk.Frame(self.ez_template_card, style="Inner.TFrame")
        preset_btn_grid.pack(fill="x", pady=(0, 6))
        preset_btn_grid.columnconfigure(0, weight=1)
        preset_btn_grid.columnconfigure(1, weight=1)
        preset_btn_grid.columnconfigure(2, weight=1)

        self.btn_preset_low = ttk.Button(
            preset_btn_grid,
            text=f"⚡ {self.t('preset_low')}",
            command=lambda: self.apply_resource_preset("low"),
        )
        self.btn_preset_low.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.btn_preset_med = ttk.Button(
            preset_btn_grid,
            text=f"⭐ {self.t('preset_med')}",
            command=lambda: self.apply_resource_preset("medium"),
        )
        self.btn_preset_med.grid(row=0, column=1, sticky="ew", padx=(3, 3))

        self.btn_preset_high = ttk.Button(
            preset_btn_grid,
            text=f"🚀 {self.t('preset_high')}",
            command=lambda: self.apply_resource_preset("high"),
        )
        self.btn_preset_high.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        # Status badge frame
        badge_frame = ttk.Frame(self.ez_template_card, style="Soft.TFrame", padding=(10, 8))
        badge_frame.pack(fill="x", pady=(4, 0))
        self.preset_badge_label = ttk.Label(
            badge_frame,
            textvariable=self.preset_badge_var,
            style="Soft.TLabel",
            foreground=c["cyan"],
            font=("Segoe UI Semibold", 10),
        )
        self.preset_badge_label.pack(anchor="w")
        self.preset_desc_label = self._wrapped_label(
            badge_frame,
            textvariable=self.preset_desc_var,
            style="Soft.TLabel",
            foreground=c["muted"],
            font=("Segoe UI", 9),
        )
        self.preset_desc_label.pack(fill="x", pady=(2, 0))

        # Audio inputs in left card
        mic_card = self._card(left, self.t("card_input_voice"), self.t("card_input_voice_sub"))
        mic_row = self._field_row(mic_card, self.t("field_mic"))
        self.mic_box = ttk.Combobox(mic_row, textvariable=self.mic_var, state="readonly")
        self.mic_box.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self._track(self.mic_box, "readonly")
        self.refresh_button = ttk.Button(mic_row, text=self.t("btn_refresh_mic"), command=self.refresh_microphones)
        self.refresh_button.grid(row=0, column=2)

        self.audio_meter = ttk.Progressbar(
            mic_card,
            mode="determinate",
            maximum=100,
            style="Audio.Horizontal.TProgressbar",
        )
        self.audio_meter.pack(fill="x", pady=(6, 4))
        meters = ttk.Frame(mic_card, style="Soft.TFrame", padding=8)
        meters.pack(fill="x", pady=(0, 6))
        for index, (title, variable) in enumerate((
            (self.t("meter_rms"), self.rms_var),
            (self.t("meter_peak"), self.peak_var),
            (self.t("meter_vad"), self.vad_status_var),
        )):
            meters.columnconfigure(index, weight=1, uniform="meters")
            ttk.Label(meters, text=title, style="Soft.TLabel", foreground=c["muted"],
                      font=("Segoe UI", 8)).grid(row=0, column=index, sticky="w")
            ttk.Label(meters, textvariable=variable, style="Soft.TLabel", foreground=c["cyan"],
                      font=("Segoe UI Semibold", 12)).grid(row=1, column=index, sticky="w")
        self._wrapped_label(mic_card, textvariable=self.audio_device_var, style="Muted.TLabel").pack(fill="x")
        self._wrapped_label(mic_card, textvariable=self.audio_hint_var, style="Card.TLabel").pack(fill="x", pady=(4, 0))

        # Targets in left card
        lang_card = self._card(left, self.t("card_caption_lang"), self.t("card_caption_lang_sub"))
        self.target_boxes: list[ttk.Combobox] = []
        for index, variable in enumerate(self.target_vars, 1):
            row = self._field_row(lang_card, f"Target {index}")
            box = FilterCombobox(row, choices=TARGET_CHOICES, textvariable=variable)
            box.grid(row=0, column=1, sticky="ew")
            self._track(box, "readonly")
            self.target_boxes.append(box)

        # Tab 1 Right side: Status & Live caption monitor
        status_card = self._card(right, self.t("card_status_inference"), self.t("card_status_inference_sub"))
        self._wrapped_label(status_card, textvariable=self.activity_var, style="Soft.TLabel").pack(fill="x", pady=2)
        self._wrapped_label(status_card, textvariable=self.runtime_var, style="Soft.TLabel").pack(fill="x", pady=2)
        self._wrapped_label(status_card, textvariable=self.metrics_var, style="Muted.TLabel").pack(fill="x", pady=2)

        live_card = self._card(right, self.t("card_monitor_caption"), self.t("card_monitor_caption_sub"))
        self.transcript_var = tk.StringVar(value=self.t("transcript_placeholder"))
        self._wrapped_label(live_card, textvariable=self.transcript_var, style="Card.TLabel").pack(fill="x", pady=(0, 8))

        self.monitor_output_container = ttk.Frame(live_card, style="Inner.TFrame")
        self.monitor_output_container.pack(fill="x")
        self.monitor_language_labels: list[ttk.Label] = []
        self.monitor_text_vars: list[tk.StringVar] = []
        for index in range(3):
            row = ttk.Frame(self.monitor_output_container, style="Soft.TFrame", padding=(10, 6))
            row.pack(fill="x", pady=2)
            row.columnconfigure(1, weight=1)
            language_label = ttk.Label(row, text=f"Target {index + 1}", style="Soft.TLabel", anchor="w")
            language_label.grid(row=0, column=0, sticky="nw", padx=(0, 8))
            text_var = tk.StringVar(value="—")
            text_label = self._wrapped_label(row, textvariable=text_var, style="Soft.TLabel")
            text_label.grid(row=0, column=1, sticky="ew")
            self.monitor_text_vars.append(text_var)
            self.monitor_language_labels.append(language_label)
        self._update_monitor_languages()

        self.overlay_rows_container = None

        # Tab 2: Mesin & VAD (Dedicated Tab with Dynamic Scroll & Responsive Grid)
        self.tab_engine = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_engine, text=self.t("tab_engine"))
        self._build_engine_tab(self.tab_engine)
        self._advanced_body = self.tab_engine

        # Tab 3: Gaya Caption (3 Profiles & Editor with In-App Preview)
        tab_styles = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_styles, text=self.t("tab_styles"))
        self._build_styles_tab(tab_styles)

        # Tab 4: Model & Resource
        self.tab_models = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_models, text=self.t("tab_models"))
        self._build_models_tab(self.tab_models)

        # Tab 5: OBS Studio Guide
        tab_obs = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_obs, text=self.t("tab_obs"))
        self._build_obs_tab(tab_obs)

        # Tab 6: About
        tab_about = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab_about, text=self.t("tab_about"))
        self._build_about_tab(tab_about)

        # Footer: Always visible actions and credit at bottom right
        self.footer = ttk.Frame(shell)
        self.footer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.save_button = ttk.Button(self.footer, text=self.t("btn_save"), command=self.save)
        self.test_button = ttk.Button(self.footer, text=self.t("btn_test_overlay"), command=self.test_overlay)
        self.start_button = ttk.Button(
            self.footer,
            text=self.t("btn_start_caption_caps"),
            style="Accent.TButton",
            command=self.toggle,
        )
        self.credit_label = ttk.Label(
            self.footer,
            text="by kenewjr 2026",
            style="Muted.TLabel",
            font=("Segoe UI", 9, "bold"),
            foreground=c["subtle"],
        )
        self.footer.bind("<Configure>", self._resize_footer)

        # Wire traces for template setting dynamics & source language selection
        self.source_var.trace_add("write", self._on_source_language_changed)
        for var in (
            self.model_var,
            self.beam_var,
            self.stt_device_var,
            self.stt_compute_type_var,
            self.mt_device_var,
            self.mt_compute_type_var,
            self.mt_beam_size_var,
            self.denoise_engine_var,
            self.slang_normalization_var,
            self.profanity_filter_var,
            self.audio_clarity_var,
        ):
            var.trace_add("write", self._on_advanced_setting_modified)

    def _build_engine_tab(self, parent: ttk.Frame) -> None:
        """Dedicated tab for Engine & VAD configuration with dynamic scroll and responsive grid."""
        c = self.colors
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.engine_canvas = tk.Canvas(parent, bg=c["bg"], highlightthickness=0, yscrollincrement=20)
        self.engine_canvas.grid(row=0, column=0, sticky="nsew")
        self.engine_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.engine_canvas.yview, style="Vertical.TScrollbar")
        self.engine_scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))

        def _on_engine_yview(first, last):
            self.engine_scrollbar.set(first, last)
            self._update_engine_scroll_hint(float(first), float(last))

        self.engine_canvas.configure(yscrollcommand=_on_engine_yview)

        # Dynamic scroll hint bar at bottom of tab_engine
        self.engine_scroll_hint_frame = ttk.Frame(parent, style="Soft.TFrame", padding=(8, 5))
        self.engine_scroll_hint_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.engine_scroll_hint_frame.columnconfigure(0, weight=1)
        self.engine_scroll_hint_label = ttk.Label(
            self.engine_scroll_hint_frame,
            text="⬇️ Ada pengaturan VRAM, VAD & Kosakata di bawah • Gulir mouse untuk melihat",
            style="Soft.TLabel",
            foreground=c["cyan"],
            font=("Segoe UI Semibold", 9),
            anchor="center",
        )
        self.engine_scroll_hint_label.grid(row=0, column=0, sticky="ew")

        self.engine_body = ttk.Frame(self.engine_canvas, padding=(10, 8))
        self._engine_body_window = self.engine_canvas.create_window(0, 0, window=self.engine_body, anchor="nw")

        self.engine_left = ttk.Frame(self.engine_body)
        self.engine_right = ttk.Frame(self.engine_body)
        self._apply_engine_layout()

        self.engine_body.bind("<Configure>", lambda _event: self.engine_canvas.configure(scrollregion=self.engine_canvas.bbox("all")))
        self.engine_canvas.bind("<Configure>", self._resize_engine_layout)

        # Card 1: Whisper STT Engine
        stt_card = self._card(self.engine_left, self.t("card_stt_title"), self.t("card_stt_sub"))
        self._compact_combo_field(stt_card, self.t("field_whisper_model"), MODEL_SIZES, self.model_var)
        self.whisper_lang_hint_label = self._wrapped_label(
            stt_card,
            text="",
            style="Muted.TLabel",
        )
        self.whisper_lang_hint_label.pack(fill="x", pady=(1, 6))
        self._compact_combo_field(stt_card, self.t("field_stt_device"), ("auto", "cuda", "cpu"), self.stt_device_var)
        self._compact_combo_field(stt_card, self.t("field_stt_precision"), COMPUTE_TYPES, self.stt_compute_type_var)
        self._compact_combo_field(stt_card, self.t("field_beam_size"), ("1", "3", "5"), self.beam_var)
        self._wrapped_label(
            stt_card,
            text=self.t("card_stt_sub"),
            style="Muted.TLabel",
        ).pack(fill="x", pady=(6, 0))

        # Card 2: NLLB MT Engine
        mt_card = self._card(self.engine_left, self.t("card_mt_title"), self.t("card_mt_sub"))
        self._compact_combo_field(mt_card, self.t("field_mt_device"), ("auto", "cuda", "cpu"), self.mt_device_var)
        self._compact_combo_field(mt_card, self.t("field_mt_precision"), COMPUTE_TYPES, self.mt_compute_type_var)
        self._compact_combo_field(mt_card, self.t("field_mt_beam"), ("1", "3"), self.mt_beam_size_var)
        cpu_r = self._field_row(mt_card, self.t("field_cpu_threads"), compact=True)
        cpu_sp = ttk.Spinbox(cpu_r, from_=1, to=16, textvariable=self.cpu_threads_var)
        cpu_sp.grid(row=0, column=1, sticky="ew")
        self._track(cpu_sp)
        ttk.Checkbutton(mt_card, text=self.t("chk_slang_norm"), variable=self.slang_normalization_var).pack(anchor="w", pady=(4, 2))
        is_en = (getattr(self.config, "ui_language", "id") == "en")
        mt_active_desc = (
            f"Active model: {self.config.nllb_model}\n🌐 Supports 200 languages (English, Japanese, Korean, Chinese, Arabic, Spanish, etc.)"
            if is_en else
            f"Model aktif: {self.config.nllb_model}\n🌐 Mendukung 200 bahasa (Inggris, Jepang, Korea, Mandarin, Arab, Spanyol, dll.)"
        )
        self.mt_active_label = self._wrapped_label(
            mt_card,
            text=mt_active_desc,
            style="Muted.TLabel",
        )
        self.mt_active_label.pack(fill="x", pady=(4, 0))

        # Card 2.5: Hardware VRAM Safety Indicator
        vram_card = self._card(self.engine_left, self.t("card_vram_title"), self.t("card_vram_sub"))
        self.vram_badge_label = ttk.Label(
            vram_card,
            text="Checking VRAM…" if is_en else "Memeriksa VRAM…",
            font=("Segoe UI Semibold", 9),
            style="Soft.TLabel",
            padding=(8, 6),
        )
        self.vram_badge_label.pack(anchor="w", pady=(4, 6))
        self.vram_detail_label = self._wrapped_label(
            vram_card,
            text="",
            style="Muted.TLabel",
        )
        self.vram_detail_label.pack(fill="x", pady=(2, 4))

        self.model_var.trace_add("write", lambda *_: (self._update_vram_status(), self._update_model_lang_hints()))
        self.stt_device_var.trace_add("write", lambda *_: self._update_vram_status())
        self.mt_device_var.trace_add("write", lambda *_: self._update_vram_status())
        self.stt_compute_type_var.trace_add("write", lambda *_: self._update_vram_status())
        self.mt_compute_type_var.trace_add("write", lambda *_: self._update_vram_status())

        # Card 3: Silero VAD
        vad_card = self._card(self.engine_right, self.t("card_vad_title"), self.t("card_vad_sub"))
        sens_r = self._field_row(vad_card, self.t("field_vad_threshold"), compact=True)
        sens_sp = ttk.Spinbox(sens_r, from_=0.05, to=0.95, increment=0.05, textvariable=self.vad_var)
        sens_sp.grid(row=0, column=1, sticky="ew")
        self._track(sens_sp)

        sil_r = self._field_row(vad_card, self.t("field_silence_gap"), compact=True)
        sil_sp = ttk.Spinbox(sil_r, from_=100, to=3000, increment=50, textvariable=self.silence_var)
        sil_sp.grid(row=0, column=1, sticky="ew")
        self._track(sil_sp)

        dur_r = self._field_row(vad_card, self.t("field_max_duration"), compact=True)
        dur_sp = ttk.Spinbox(dur_r, from_=3, to=60, textvariable=self.duration_var)
        dur_sp.grid(row=0, column=1, sticky="ew")
        self._track(dur_sp)

        tout_r = self._field_row(vad_card, self.t("field_clear_timeout"), compact=True)
        tout_sp = ttk.Spinbox(tout_r, from_=0, to=60, textvariable=self.timeout_var)
        tout_sp.grid(row=0, column=1, sticky="ew")
        self._track(tout_sp)

        self._wrapped_label(
            vad_card,
            text=self.t("vad_prob_note"),
            style="Muted.TLabel",
        ).pack(fill="x", pady=(4, 0))

        # Card 4: Audio Pre-processing & Port
        dsp_card = self._card(self.engine_right, self.t("card_dsp_title"), self.t("card_dsp_sub"))
        self._compact_combo_field(dsp_card, self.t("field_channel"), ("mix", "left", "right"), self.audio_channel_var)
        gain_r = self._field_row(dsp_card, self.t("field_mic_gain"), compact=True)
        gain_sp = ttk.Spinbox(gain_r, from_=-24.0, to=18.0, increment=1.0, textvariable=self.audio_gain_var)
        gain_sp.grid(row=0, column=1, sticky="ew")
        self._track(gain_sp)
        self._compact_combo_field(dsp_card, self.t("field_denoiser"), ("clarity", "dtln", "hybrid", "off"), self.denoise_engine_var)

        norm_r = ttk.Frame(dsp_card, style="Inner.TFrame")
        norm_r.pack(fill="x", pady=4)
        ttk.Checkbutton(norm_r, text=self.t("chk_auto_normalize"), variable=self.normalize_audio_var).pack(anchor="w")
        ttk.Checkbutton(norm_r, text=self.t("chk_clarity_boost"), variable=self.audio_clarity_var).pack(anchor="w")

        port_r = self._field_row(dsp_card, self.t("field_overlay_port"), compact=True)
        port_sp = ttk.Spinbox(port_r, from_=1024, to=65535, textvariable=self.overlay_port_var)
        port_sp.grid(row=0, column=1, sticky="ew")
        self._track(port_sp)

        # Card 5: Vocabulary & Adaptive Learning
        vocab_card = self._card(self.engine_right, self.t("card_vocab_hints_title"), self.t("card_vocab_hints_sub"))
        hw_r = self._field_row(vocab_card, self.t("field_whisper_hints"), compact=True)
        hw_ent = ttk.Entry(hw_r, textvariable=self.hotwords_var)
        hw_ent.grid(row=0, column=1, sticky="ew")
        self._track(hw_ent)

        lr_row = ttk.Frame(vocab_card, style="Inner.TFrame")
        lr_row.pack(fill="x", pady=(4, 6))
        ttk.Label(lr_row, text=self.t("label_learned_vocab"), style="Card.TLabel").pack(side="left")
        self.vocab_badge_label = ttk.Label(
            lr_row,
            textvariable=self.vocab_status_var,
            style="Soft.TLabel",
            foreground=c["cyan"],
            font=("Segoe UI Semibold", 9),
            padding=(6, 2),
        )
        self.vocab_badge_label.pack(side="left", padx=6)

        act_row = ttk.Frame(vocab_card, style="Inner.TFrame")
        act_row.pack(fill="x", pady=(2, 0))
        ttk.Button(act_row, text=self.t("btn_open_vocab_file"), command=self._open_vocab_file).pack(side="left", padx=(0, 6))
        ttk.Button(act_row, text=self.t("btn_reset_learned"), command=self._reset_learned_vocab).pack(side="left")

        ttk.Checkbutton(vocab_card, text=self.t("chk_profanity_filter"), variable=self.profanity_filter_var).pack(anchor="w", pady=(6, 2))

    def _build_styles_tab(self, parent: ttk.Frame) -> None:
        top_row = ttk.Frame(parent)
        top_row.pack(fill="x", pady=(0, 8))

        ttk.Label(top_row, text=self.t("label_stage_anchor")).pack(side="left", padx=(0, 6))
        self.stage_anchor_var = tk.StringVar(value=self.config.overlay.anchor)
        anchor_combo = ttk.Combobox(top_row, textvariable=self.stage_anchor_var, values=list(ANCHORS), state="readonly", width=14)
        anchor_combo.pack(side="left", padx=(0, 16))
        anchor_combo.bind("<<ComboboxSelected>>", self._on_stage_layout_changed)

        ttk.Label(top_row, text=self.t("label_stage_gap")).pack(side="left", padx=(0, 6))
        self.stage_gap_var = tk.StringVar(value=str(int(self.config.overlay.gap)))
        gap_spin = ttk.Spinbox(top_row, from_=0, to=80, textvariable=self.stage_gap_var, width=5, command=self._on_stage_layout_changed)
        gap_spin.pack(side="left", padx=(0, 16))
        gap_spin.bind("<FocusOut>", self._on_stage_layout_changed)
        gap_spin.bind("<Return>", self._on_stage_layout_changed)
        gap_spin.bind("<KeyRelease>", self._on_stage_layout_changed)

        ttk.Label(top_row, text=self.t("label_platform_preset")).pack(side="left", padx=(0, 6))
        self.stage_platform_var = tk.StringVar(value="Custom")
        platform_combo = ttk.Combobox(
            top_row,
            textvariable=self.stage_platform_var,
            values=["Custom", "Standard 16:9", "YouTube 1080p", "Twitch Stream", "TikTok Live 9:16"],
            state="readonly",
            width=16,
        )
        platform_combo.pack(side="left", padx=(0, 16))
        platform_combo.bind("<<ComboboxSelected>>", self._on_platform_preset_selected)

        styles_nb = ttk.Notebook(parent)
        styles_nb.pack(fill="both", expand=True)

        lang_code = getattr(self.config, "ui_language", "id")
        self.caption_editors = []
        for slot in (1, 2, 3):
            tab = ttk.Frame(styles_nb)
            styles_nb.add(tab, text=f"  {self.t(f'tab_slot_{slot}')}  ")
            editor = CaptionEditor(tab, slot=slot, get_style=self._get_profile, on_change=self._set_profile, lang=lang_code)
            editor.pack(fill="both", expand=True)
            self.caption_editors.append(editor)

    def _build_models_tab(self, parent: ttk.Frame) -> None:
        c = self.colors
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        models_canvas = tk.Canvas(parent, bg=c["bg"], highlightthickness=0, yscrollincrement=20)
        models_canvas.grid(row=0, column=0, sticky="nsew")
        models_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=models_canvas.yview, style="Vertical.TScrollbar")
        models_scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        models_canvas.configure(yscrollcommand=models_scrollbar.set)

        container = ttk.Frame(models_canvas)
        container_window = models_canvas.create_window(0, 0, window=container, anchor="nw")
        container.bind("<Configure>", lambda _event: models_canvas.configure(scrollregion=models_canvas.bbox("all")))
        models_canvas.bind("<Configure>", lambda e: models_canvas.itemconfig(container_window, width=e.width))
        self.models_canvas = models_canvas
        self.models_container = container

        lang_code = getattr(self.config, "ui_language", "id")
        hw_box = ttk.LabelFrame(container, text=self.t("sec_hw_box"), padding=8)
        hw_box.pack(fill="x", pady=(0, 8))
        self.hw_info_var = tk.StringVar(value=hardware_info(self.app_dir, lang_code))
        self._wrapped_label(hw_box, textvariable=self.hw_info_var, font=("Consolas", 9)).pack(fill="x")
        ttk.Button(hw_box, text=self.t("btn_refresh_hw"), command=lambda: self.hw_info_var.set(hardware_info(self.app_dir, getattr(self.config, "ui_language", "id")))).pack(anchor="e", pady=(4, 0))

        # 1. Estimasi Detail Resource per Model
        res_card = ttk.LabelFrame(container, text=self.t("sec_res_est_title"), padding=8)
        res_card.pack(fill="x", pady=(0, 8))

        self._wrapped_label(
            res_card,
            text=self.t("sec_res_est_desc"),
            style="Muted.TLabel",
        ).pack(fill="x", pady=(0, 6))

        tree_frame = ttk.Frame(res_card)
        tree_frame.pack(fill="x", expand=True)

        columns = ("model", "disk", "ram", "vram", "threads", "note")
        self.res_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=9,
            selectmode="browse",
        )
        self.res_tree.heading("model", text=self.t("tbl_col_component"), anchor="w")
        self.res_tree.heading("disk", text=self.t("tbl_col_disk"), anchor="center")
        self.res_tree.heading("ram", text=self.t("tbl_col_ram"), anchor="center")
        self.res_tree.heading("vram", text=self.t("tbl_col_vram"), anchor="center")
        self.res_tree.heading("threads", text=self.t("tbl_col_threads"), anchor="center")
        self.res_tree.heading("note", text=self.t("tbl_col_recommendation"), anchor="w")

        self.res_tree.column("model", width=190, minwidth=150, anchor="w")
        self.res_tree.column("disk", width=95, minwidth=80, anchor="center")
        self.res_tree.column("ram", width=135, minwidth=110, anchor="center")
        self.res_tree.column("vram", width=145, minwidth=120, anchor="center")
        self.res_tree.column("threads", width=105, minwidth=90, anchor="center")
        self.res_tree.column("note", width=340, minwidth=240, stretch=True, anchor="w")

        for idx, row in enumerate(model_resource_table_rows(lang_code)):
            tag = "even" if idx % 2 == 0 else "odd"
            self.res_tree.insert(
                "",
                "end",
                iid=row["key"],
                values=(row["model"], row["disk"], row["ram"], row["vram"], row["threads"], row["note"]),
                tags=(tag,),
            )

        self.res_tree.tag_configure("even", background=c["table_even"], foreground=c["text"])
        self.res_tree.tag_configure("odd", background=c["table_odd"], foreground=c["text"])

        ai_model_keys = tuple(
            k for k in (
                "whisper-small-id",
                "whisper-medium-id",
                "small",
                "base",
                "tiny",
                "medium",
                "large-v3-turbo",
                "large-v3",
                "distil-large-v3",
                "nllb",
                "nllb-1.3b",
            ) if k in CATALOG
        )
        self.ai_model_keys = ai_model_keys

        def _on_table_select(_event):
            sel = self.res_tree.selection()
            if not sel:
                return
            key = sel[0]
            if key in ai_model_keys:
                self.model_card_key.set(key)
                self._on_model_card_selected()
            elif key == "dtln":
                self._refresh_dtln_status()
                self._set_activity("Komponen Audio DTLN dipilih — kelola di kartu Komponen Audio & Neural Denoiser di bawah", "info")
            elif key == "silero":
                self._set_activity("Silero VAD v5 adalah detektor jeda suara bawaan kizcaption (models/silero_vad.onnx)", "info")

        self.res_tree.bind("<<TreeviewSelect>>", _on_table_select)
        self.res_tree.pack(fill="x", expand=True)

        mc_box = ttk.LabelFrame(container, text=self.t("sec_models_title"), padding=8)
        mc_box.pack(fill="x", pady=(0, 8))

        ctrl_row = ttk.Frame(mc_box)
        ctrl_row.pack(fill="x", pady=(0, 6))
        ttk.Label(ctrl_row, text=self.t("label_choose_ai_model")).pack(side="left", padx=(0, 6))
        self.model_card_key = tk.StringVar(value="whisper-small-id" if "whisper-small-id" in ai_model_keys else "small")
        combo = ttk.Combobox(ctrl_row, textvariable=self.model_card_key, values=list(ai_model_keys), state="readonly", width=28)
        combo.pack(side="left", padx=(0, 10))
        combo.bind("<<ComboboxSelected>>", self._on_model_card_selected)

        self.model_status_var = tk.StringVar()
        self.model_status_label = ttk.Label(ctrl_row, textvariable=self.model_status_var, font=("Segoe UI Semibold", 9))
        self.model_status_label.pack(side="left", padx=(0, 10))

        self.download_btn = ttk.Button(ctrl_row, text=self.t("btn_download_ai_model"), command=self._download_selected_model)
        self.download_btn.pack(side="right", padx=(6, 0))

        self.apply_model_btn = ttk.Button(ctrl_row, text=self.t("btn_use_ai_model"), command=self._apply_selected_model)
        self.apply_model_btn.pack(side="right", padx=(6, 0))

        self.scan_models_btn = ttk.Button(ctrl_row, text=self.t("btn_scan_models"), command=self._scan_and_detect_models)
        self.scan_models_btn.pack(side="right")

        self.model_progress_var = tk.StringVar()

        # Dedicated animated download card
        self.dl_card = ttk.Frame(mc_box, style="Soft.TFrame", padding=10)
        self.dl_card.pack(fill="x", pady=(4, 8))

        dl_top = ttk.Frame(self.dl_card, style="Soft.TFrame")
        dl_top.pack(fill="x", pady=(0, 4))
        self.dl_status_var = tk.StringVar(value=self.t("dl_status_ready"))
        ttk.Label(dl_top, textvariable=self.dl_status_var, style="Soft.TLabel", font=("Segoe UI Semibold", 9)).pack(side="left")
        self.dl_speed_var = tk.StringVar(value="")
        ttk.Label(dl_top, textvariable=self.dl_speed_var, style="Soft.TLabel", foreground=self.colors["cyan"], font=("Segoe UI Semibold", 9)).pack(side="right")

        self.download_pbar = ttk.Progressbar(
            self.dl_card,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            style="Audio.Horizontal.TProgressbar",
        )
        self.download_pbar.pack(fill="x", pady=(2, 4))

        dl_bot = ttk.Frame(self.dl_card, style="Soft.TFrame")
        dl_bot.pack(fill="x")
        self.dl_size_var = tk.StringVar(value="")
        ttk.Label(dl_bot, textvariable=self.dl_size_var, style="Soft.TLabel", foreground=self.colors["muted"], font=("Consolas", 8)).pack(side="left")
        self.dl_eta_var = tk.StringVar(value="")
        ttk.Label(dl_bot, textvariable=self.dl_eta_var, style="Soft.TLabel", foreground=self.colors["amber"], font=("Segoe UI Semibold", 8)).pack(side="right")

        desc_box = ttk.Frame(mc_box, style="Soft.TFrame", padding=10)
        desc_box.pack(fill="x")
        self.model_desc_var = tk.StringVar()
        self._wrapped_label(desc_box, textvariable=self.model_desc_var, style="Soft.TLabel").pack(fill="x")

        # 3. Dedicated Card for Audio Neural Denoiser (DTLN ONNX)
        dtln_box = ttk.LabelFrame(container, text=self.t("sec_denoiser_title"), padding=10)
        dtln_box.pack(fill="x", pady=(4, 12))

        self._wrapped_label(
            dtln_box,
            text=self.t("dtln_desc"),
            style="Muted.TLabel",
        ).pack(fill="x", pady=(0, 8))

        dtln_status_card = ttk.Frame(dtln_box, style="Soft.TFrame", padding=10)
        dtln_status_card.pack(fill="x", pady=(0, 8))
        dtln_status_card.columnconfigure(1, weight=1)

        ttk.Label(dtln_status_card, text=self.t("label_dtln_disk_status"), style="Soft.TLabel", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        is_en = getattr(self.config, "ui_language", "id") == "en"
        self.dtln_status_var = tk.StringVar(value="Checking…" if is_en else "Memeriksa…")
        self.dtln_status_label = ttk.Label(dtln_status_card, textvariable=self.dtln_status_var, font=("Segoe UI Semibold", 9))
        self.dtln_status_label.grid(row=0, column=1, sticky="w", pady=3)

        ttk.Label(dtln_status_card, text=self.t("label_dtln_usage_status"), style="Soft.TLabel", font=("Segoe UI Semibold", 9)).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        self.dtln_usage_var = tk.StringVar(value="")
        self.dtln_usage_label = ttk.Label(dtln_status_card, textvariable=self.dtln_usage_var, font=("Segoe UI Semibold", 9))
        self.dtln_usage_label.grid(row=1, column=1, sticky="w", pady=3)

        dtln_actions = ttk.Frame(dtln_box)
        dtln_actions.pack(fill="x")

        self.dtln_download_btn = ttk.Button(dtln_actions, text=self.t("btn_dl_dtln"), command=self._download_dtln_component)
        self.dtln_download_btn.pack(side="left", padx=(0, 8))

        self.dtln_apply_btn = ttk.Button(dtln_actions, text=self.t("btn_toggle_dtln"), command=self._toggle_dtln_denoiser)
        self.dtln_apply_btn.pack(side="left")

        self._refresh_dtln_status()
        self.denoise_engine_var.trace_add("write", lambda *_: self._refresh_dtln_status())

        self._on_model_card_selected()

    def _build_obs_tab(self, parent: ttk.Frame) -> None:
        left_card = self._card(parent, self.t("sec_obs_source_title"), self.t("sec_obs_source_sub"))

        status_r = ttk.Frame(left_card, style="Soft.TFrame", padding=10)
        status_r.pack(fill="x", pady=(0, 6))
        status_r.columnconfigure(1, weight=1)

        ttk.Label(status_r, text=self.t("obs_server_status_label"), style="Soft.TLabel", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(status_r, textvariable=self.overlay_state_var, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(status_r, text=self.t("obs_connected_clients_label"), style="Soft.TLabel", font=("Segoe UI Semibold", 9)).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(status_r, textvariable=self.overlay_clients_var, font=("Segoe UI", 9), foreground=self.colors["cyan"]).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(status_r, text=self.t("obs_last_activity_label"), style="Soft.TLabel", font=("Segoe UI Semibold", 9)).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(status_r, textvariable=self.overlay_activity_var, font=("Segoe UI", 9)).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(
            status_r,
            text=self.t("obs_client_count_desc"),
            style="Muted.TLabel",
            font=("Segoe UI", 8),
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        url_box = ttk.Frame(left_card, style="Soft.TFrame", padding=12)
        url_box.pack(fill="x", pady=4)
        url_box.columnconfigure(0, weight=1)

        ttk.Label(
            url_box,
            text=self.t("obs_all_in_one_title"),
            style="Soft.TLabel",
            foreground=self.colors["cyan"],
            font=("Segoe UI Semibold", 10),
        ).grid(row=0, column=0, sticky="w", columnspan=2, pady=(0, 4))

        ttk.Label(
            url_box,
            text=self.t("obs_all_in_one_desc"),
            style="Muted.TLabel",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, sticky="w", columnspan=2, pady=(0, 8))

        url_row = ttk.Frame(url_box, style="Soft.TFrame")
        url_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        url_row.columnconfigure(0, weight=1)

        self.obs_url_var = tk.StringVar(value=self.static_overlay_url())
        url_entry = ttk.Entry(url_row, textvariable=self.obs_url_var, font=("Consolas", 9))
        url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        url_entry["state"] = "readonly"

        btn_row = ttk.Frame(url_box, style="Soft.TFrame")
        btn_row.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(btn_row, text=self.t("btn_copy_link"), command=lambda: self.copy_url("all")).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text=self.t("btn_open_browser"), command=lambda: self.open_preview("all")).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text=self.t("btn_test_overlay"), command=self.test_overlay).pack(side="left")

        right_card = self._card(parent, self.t("obs_guide_title"), self.t("obs_guide_sub"))

        guide_text = "\n\n".join([self.t(f"obs_guide_step{i}") for i in range(1, 8)])
        self._wrapped_label(right_card, text=guide_text, style="Card.TLabel").pack(fill="both", expand=True)

    def _build_about_tab(self, parent: ttk.Frame) -> None:
        c = self.colors
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "kzp_icon.png"
        if icon_path.is_file():
            try:
                self._about_logo_img = tk.PhotoImage(file=str(icon_path))
                ttk.Label(parent, image=self._about_logo_img).pack(anchor="w", pady=(0, 6))
            except Exception:
                pass

        ttk.Label(parent, text=f"KizCaption v{__version__}", font=("Segoe UI Semibold", 15), foreground=c["cyan"]).pack(anchor="w", pady=(0, 4))
        ttk.Label(parent, text=f"Credit: {CREDIT}", font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(0, 8))

        is_en = (getattr(self.config, "ui_language", "id") == "en")
        about_text = (
            f"{self.t('about_desc')}\n\n"
            f"{self.t('about_components')}\n"
            "• Faster-Whisper (Systran) — Apache 2.0\n"
            "• CTranslate2 — MIT License\n"
            "• Silero VAD — MIT License\n"
            "• NLLB-200 (Meta) — CC-BY-NC-4.0 (Non-Commercial)\n"
            f"• {'KZP Logo — Original & Copyright-Free' if is_en else 'Logo KZP — Orisinal & Bebas Hak Cipta'}"
        )
        self._wrapped_label(
            parent,
            text=about_text,
            style="Muted.TLabel",
        ).pack(fill="x", pady=(0, 16))

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", pady=4)
        ttk.Button(btn_row, text=self.t("btn_open_config_folder"), command=lambda: self._open_dir(self.app_dir)).pack(side="left", padx=4)
        ttk.Button(btn_row, text=self.t("btn_open_models_folder"), command=lambda: self._open_dir(self.app_dir / "models")).pack(side="left", padx=4)
        ttk.Button(btn_row, text=self.t("btn_open_logs_folder"), command=lambda: self._open_dir(self.app_dir / "logs")).pack(side="left", padx=4)
        ttk.Button(btn_row, text="📋 Salin Log Error" if not is_en else "📋 Copy Error Log", command=self._copy_recent_logs).pack(side="left", padx=4)
        ttk.Button(btn_row, text=self.t("btn_check_updates"), command=self.check_for_updates_ui).pack(side="left", padx=4)

    def _copy_recent_logs(self) -> None:
        log_files = [self.app_dir / "logs" / "kizcaption.log", self.app_dir / "logs" / "diagnostic.log"]
        content = ""
        for lf in log_files:
            if lf.is_file():
                try:
                    with open(lf, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        content = "".join(lines[-100:])
                        if content.strip():
                            break
                except Exception:
                    pass
        is_en = (getattr(self.config, "ui_language", "id") == "en")
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo(
                "Log Disalin" if not is_en else "Logs Copied",
                "Log sistem terbaru berhasil disalin ke clipboard!\nAnda dapat menempelkan (paste) log ini untuk dikirim ke developer (kenewjr)."
                if not is_en else
                "Recent system logs copied to clipboard!\nYou can paste and send this log to developer (kenewjr).",
                parent=self.root,
            )
        else:
            self._open_dir(self.app_dir / "logs")

    def check_for_updates_ui(self) -> None:
        self._set_activity("Memeriksa pembaruan rilis GitHub…", "working")

        def run_check():
            from lumacaption.updater import check_for_updates
            res = check_for_updates(__version__)

            def on_done():
                self._set_activity("Pemeriksaan pembaruan selesai", "idle")
                if res["has_update"]:
                    msg = (
                        f"{self.t('update_found_msg', latest=res['latest_version'], current=__version__)}\n\n"
                        f"{self.t('update_ask_open')}"
                    )
                    if messagebox.askyesno(self.t("update_title"), msg):
                        webbrowser.open(res["release_url"])
                elif res["error"]:
                    messagebox.showwarning(
                        self.t("update_title"),
                        f"{self.t('update_err_msg')}:\n{res['error']}",
                    )
                else:
                    messagebox.showinfo(
                        self.t("update_title"),
                        self.t("update_none_msg", version=__version__),
                    )

            self.root.after(0, on_done)

        threading.Thread(target=run_check, daemon=True).start()

    def _open_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(p))
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _open_vocab_file(self) -> None:
        vf = self.app_dir / "vocabulary.json"
        if not vf.is_file():
            self.vocab_manager.save()
        if os.name == "nt":
            os.startfile(str(vf))
        else:
            subprocess.Popen(["xdg-open", str(vf)])

    def _reset_learned_vocab(self) -> None:
        is_en = getattr(self.config, "ui_language", "id") == "en"
        count = self.vocab_manager.learned_count
        if count == 0:
            msg = "No automatically learned terms yet." if is_en else "Belum ada kosakata yang dipelajari otomatis."
            messagebox.showinfo("KizCaption", msg)
            return
        prompt = f"Remove {count} automatically learned terms from speech history?" if is_en else f"Hapus {count} kata yang dipelajari otomatis dari ucapan?"
        if messagebox.askyesno("KizCaption", prompt):
            removed = self.vocab_manager.reset_learned()
            self._update_vocab_status()
            succ = f"{removed} terms cleared from dictionary." if is_en else f"{removed} kata berhasil dibersihkan dari kamus."
            messagebox.showinfo("KizCaption", succ)

    def _update_vocab_status(self) -> None:
        if hasattr(self, "vocab_status_var"):
            self.vocab_status_var.set(self.t("vocab_learned_status", count=self.vocab_manager.learned_count))

    def _get_profile(self, slot: int) -> CaptionStyle:
        if 1 <= slot <= len(self.config.overlay.profiles):
            return self.config.overlay.profiles[slot - 1]
        return CaptionStyle(name=f"Output {slot}")

    def _set_profile(self, slot: int, style: CaptionStyle) -> None:
        if 1 <= slot <= len(self.config.overlay.profiles):
            self.config.overlay.profiles[slot - 1] = style
            if slot == 1:
                self.font_var.set(style.font_family)
                self.font_size_var.set(str(int(style.font_size)))
                self.text_color_var.set(style.text_color)
                self.outline_color_var.set(style.outline_color)
            if self.overlay_service and self.overlay_service.running:
                self.overlay_service.update_styles(self.config.overlay)

    def _on_stage_layout_changed(self, _event=None) -> None:
        self.config.overlay.anchor = self.stage_anchor_var.get()
        try:
            val = float(self.stage_gap_var.get())
            if 0 <= val <= 80:
                self.config.overlay.gap = val
        except ValueError:
            pass
        if self.overlay_service and self.overlay_service.running:
            self.overlay_service.update_styles(self.config.overlay)

    def _on_platform_preset_selected(self, _event=None) -> None:
        p_name = self.stage_platform_var.get()
        from lumacaption.output.styles import PLATFORM_LAYOUTS
        if p_name in PLATFORM_LAYOUTS:
            layout = PLATFORM_LAYOUTS[p_name]
            for slot in (1, 2, 3):
                prof = self._get_profile(slot)
                prof.max_width = layout.get("max_width", prof.max_width)
                prof.margin_y = layout.get("margin_y", prof.margin_y)
                prof.align = layout.get("align", prof.align)
                if "font_size" in layout:
                    prof.font_size = layout["font_size"]
                self._set_profile(slot, prof)
            for editor in getattr(self, "caption_editors", []):
                editor.load_from_style(self._get_profile(editor.slot))

    def _on_model_card_selected(self, _event=None) -> None:
        key = self.model_card_key.get()
        cache = cache_for(self.app_dir, key)
        state, _ = inspect_model(key, cache)
        if state != "Tersedia lokal":
            alt_state, alt_source, _ = find_model_locally(key, self.app_dir)
            if alt_state == "Tersedia lokal" and alt_source is not None:
                adopt_model(key, alt_source, cache)
                state, _ = inspect_model(key, cache)
        lang = getattr(self.config, "ui_language", "id")
        is_en = (lang == "en")
        state_display = {
            "Tersedia lokal": "Available locally" if is_en else "Tersedia lokal",
            "Belum lengkap": "Incomplete cache" if is_en else "Belum lengkap",
            "Belum ada": "Not downloaded" if is_en else "Belum ada",
        }.get(state, state)
        self.model_status_var.set(f"Status: {state_display}")
        if hasattr(self, "res_tree") and self.res_tree and self.res_tree.exists(key):
            curr = self.res_tree.selection()
            if not curr or curr[0] != key:
                self.res_tree.selection_set(key)
                self.res_tree.see(key)
        details = resource_details(
            key,
            self.config.stt_device,
            threads=self.config.cpu_threads,
            beam=self.config.whisper_beam_size,
            lang=lang,
        )
        self.model_desc_var.set(details)
        if hasattr(self, "download_pbar"):
            if state == "Tersedia lokal":
                self.download_pbar.configure(mode="determinate", value=100)
                if is_en:
                    self.dl_status_var.set(f"✔ Model '{key}' is already available locally")
                    self.dl_speed_var.set("Ready")
                    self.dl_size_var.set(f"Disk size: ~{CATALOG[key].disk_mib:g} MiB")
                else:
                    self.dl_status_var.set(f"✔ Model '{key}' sudah tersedia lokal")
                    self.dl_speed_var.set("Siap")
                    self.dl_size_var.set(f"Kapasitas disk: ~{CATALOG[key].disk_mib:g} MiB")
                self.dl_eta_var.set("")
            elif state == "Belum lengkap":
                self.download_pbar.configure(mode="determinate", value=30)
                if is_en:
                    self.dl_status_var.set(f"⚠ Cache for model '{key}' is incomplete — click download to resume")
                    self.dl_speed_var.set("Incomplete")
                    self.dl_size_var.set(f"Estimated total: {CATALOG[key].disk_mib:g} MiB")
                else:
                    self.dl_status_var.set(f"⚠ Cache model '{key}' belum lengkap — klik unduh untuk melanjutkan")
                    self.dl_speed_var.set("Perlu Dilengkapi")
                    self.dl_size_var.set(f"Estimasi total: {CATALOG[key].disk_mib:g} MiB")
                self.dl_eta_var.set("")
            else:
                self.download_pbar.configure(mode="determinate", value=0)
                if is_en:
                    self.dl_status_var.set(f"Model '{key}' is not downloaded to disk")
                    self.dl_speed_var.set("Not Downloaded")
                    self.dl_size_var.set(f"Download size: ~{CATALOG[key].disk_mib:g} MiB")
                else:
                    self.dl_status_var.set(f"Model '{key}' belum diunduh ke disk")
                    self.dl_speed_var.set("Belum Ada")
                    self.dl_size_var.set(f"Ukuran unduhan: ~{CATALOG[key].disk_mib:g} MiB")
                self.dl_eta_var.set("")

    def _apply_selected_model(self) -> None:
        if self.pipeline and (self.pipeline.running or self.pipeline.inference_busy):
            is_en = getattr(self.config, "ui_language", "id") == "en"
            messagebox.showwarning(
                "Cannot Change Model" if is_en else "Tidak Dapat Mengganti Model",
                "Stop caption before changing the active model." if is_en else "Hentikan caption terlebih dahulu sebelum mengganti model aktif.",
                parent=self.root,
            )
            return

        key = self.model_card_key.get()
        if key in ("nllb", "nllb-1.3b"):
            nllb_repo = CATALOG[key].repo
            self.config.nllb_model = nllb_repo
            if hasattr(self, "mt_active_label"):
                self.mt_active_label.configure(text=f"Model aktif: {nllb_repo}")
            self._set_activity(f"Model NLLB aktif disetel ke: {nllb_repo}", "ok")
        elif key in MODEL_SIZES:
            self.model_var.set(key)
            self.config.whisper_model = key
            self._set_activity(f"Model Whisper aktif disetel ke: {key}", "ok")
        else:
            self._set_activity(f"Model {key} tidak dapat diaktifkan sebagai model utama", "warning")
            return

        try:
            self.config_store.save(self.config)
        except Exception as exc:
            self._set_activity(f"Gagal menyimpan model: {exc}", "error")

    def _scan_and_detect_models(self) -> None:
        from tkinter import filedialog
        adopted = detect_and_link_models(self.app_dir)
        if adopted:
            msg_items = "\n".join(f"• {m['key']} ({m['name']})" for m in adopted)
            self._set_activity(f"Ditemukan {len(adopted)} model dan siap digunakan.", "ok")
            messagebox.showinfo(
                self.t("scan_models_title"),
                self.t("scan_models_found", count=len(adopted), models=msg_items),
                parent=self.root,
            )
            self._on_model_card_selected()
            self._refresh_dtln_status()
            return

        ask_manual = messagebox.askyesno(
            self.t("scan_models_title"),
            self.t("scan_models_prompt_manual"),
            parent=self.root,
        )
        if ask_manual:
            chosen = filedialog.askdirectory(title=self.t("scan_models_title"), parent=self.root)
            if chosen:
                manual_adopted = detect_and_link_models(self.app_dir, custom_source_dir=Path(chosen))
                if manual_adopted:
                    msg_items = "\n".join(f"• {m['key']} ({m['name']})" for m in manual_adopted)
                    self._set_activity(f"Berhasil mengimpor {len(manual_adopted)} model dari {chosen}.", "ok")
                    messagebox.showinfo(
                        self.t("scan_models_title"),
                        self.t("scan_models_found", count=len(manual_adopted), models=msg_items),
                        parent=self.root,
                    )
                    self._on_model_card_selected()
                    self._refresh_dtln_status()
                else:
                    messagebox.showinfo(
                        self.t("scan_models_title"),
                        self.t("scan_models_none_found"),
                        parent=self.root,
                    )

    def _download_selected_model(self) -> None:
        key = self.model_card_key.get()
        cache = cache_for(self.app_dir, key)
        self.download_btn["state"] = "disabled"

        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spin_state = {"idx": 0, "active": True}

        self.download_pbar.configure(mode="indeterminate", value=0)
        self.download_pbar.start(12)
        self.dl_status_var.set(f"Menghubungkan ke server untuk model '{key}'…")
        self.dl_speed_var.set("Memulai…")
        self.dl_size_var.set("Menyiapkan alokasi disk…")
        self.dl_eta_var.set("")

        def spin_ticker():
            if spin_state["active"]:
                char = spinner[spin_state["idx"] % len(spinner)]
                spin_state["idx"] += 1
                cur = self.dl_status_var.get()
                if "Mengunduh" in cur:
                    clean = cur.lstrip("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⬇ ")
                    self.dl_status_var.set(f"⬇ {char} {clean}")
                self.root.after(120, spin_ticker)

        self.root.after(120, spin_ticker)

        def task():
            def cb(p):
                state = p.get("state") if isinstance(p, dict) else getattr(p, "state", "")
                pct = (p.get("percent") if isinstance(p, dict) else getattr(p, "percent", 0.0)) or 0.0
                speed_str = (p.get("speed_str") if isinstance(p, dict) else getattr(p, "speed_str", "")) or ""
                size_str = (p.get("size_str") if isinstance(p, dict) else getattr(p, "size_str", "")) or ""
                eta_str = (p.get("eta_str") if isinstance(p, dict) else getattr(p, "eta_str", "")) or ""
                filename = (p.get("filename") if isinstance(p, dict) else getattr(p, "filename", "")) or ""
                detail = (p.get("detail") if isinstance(p, dict) else getattr(p, "detail", "")) or ""

                def update_ui():
                    if state == "Mengunduh":
                        if self.download_pbar["mode"] == "indeterminate":
                            self.download_pbar.stop()
                            self.download_pbar.configure(mode="determinate")
                        self.download_pbar.configure(value=max(1.0, min(100.0, pct)))
                        name_display = filename or detail or key
                        self.dl_status_var.set(f"Mengunduh {name_display} • {pct:.1f}%")
                        self.dl_speed_var.set(f"⚡ {speed_str}")
                        self.dl_size_var.set(f"📦 {size_str}")
                        self.dl_eta_var.set(f"⏳ {eta_str}" if eta_str else "")
                    elif state == "Memverifikasi":
                        self.download_pbar.configure(mode="indeterminate")
                        self.download_pbar.start(15)
                        self.dl_status_var.set("Memverifikasi integritas file lokal…")
                        self.dl_speed_var.set("Verifikasi SHA/File")
                    elif state == "Tersedia lokal":
                        self.download_pbar.stop()
                        self.download_pbar.configure(mode="determinate", value=100)
                        self.dl_status_var.set(f"✔ Model '{key}' selesai diunduh & siap digunakan!")
                        self.dl_speed_var.set("Selesai (100%)")
                        self.dl_size_var.set("Penyimpanan lokal diverifikasi")
                        self.dl_eta_var.set("")

                self.root.after(0, update_ui)

            try:
                ensure_model(key, cache, cb)
                spin_state["active"] = False
                self.root.after(0, lambda: self._set_activity(f"Model {key} siap digunakan", "ok"))
                self.root.after(0, self._on_model_card_selected)
                self.root.after(0, self._update_model_lang_hints)
            except Exception as e:
                spin_state["active"] = False
                err_msg = str(e)
                def on_err(msg=err_msg):
                    self.download_pbar.stop()
                    self.download_pbar.configure(mode="determinate", value=0)
                    self.dl_status_var.set(f"❌ Gagal mengunduh: {msg}")
                    self.dl_speed_var.set("Terputus")
                    self.dl_size_var.set("")
                    self.dl_eta_var.set("")
                self.root.after(0, on_err)
            finally:
                spin_state["active"] = False
                self.root.after(0, lambda: self.download_btn.__setitem__("state", "normal"))

        threading.Thread(target=task, daemon=True).start()

    def _refresh_dtln_status(self) -> None:
        if not hasattr(self, "dtln_status_var") or not self.dtln_status_var:
            return
        cache = cache_for(self.app_dir, "dtln")
        state, _ = inspect_model("dtln", cache)
        c = self.colors
        is_en = getattr(self.config, "ui_language", "id") == "en"
        if state == "Tersedia lokal":
            self.dtln_status_var.set("✔ Available locally (~3.96 MiB)" if is_en else "✔ Tersedia lokal (~3.96 MiB)")
            if hasattr(self, "dtln_status_label"):
                self.dtln_status_label.configure(foreground=c["green"])
            if hasattr(self, "dtln_download_btn"):
                self.dtln_download_btn.configure(
                    text="✔ DTLN Installed (Re-download)" if is_en else "✔ DTLN Sudah Terpasang (Unduh Ulang)"
                )
        else:
            self.dtln_status_var.set("⚠ Not downloaded to disk (~3.96 MiB)" if is_en else "⚠ Belum diunduh ke disk (~3.96 MiB)")
            if hasattr(self, "dtln_status_label"):
                self.dtln_status_label.configure(foreground=c["amber"])
            if hasattr(self, "dtln_download_btn"):
                self.dtln_download_btn.configure(
                    text="Download DTLN Component (~3.96 MB)" if is_en else "Unduh Komponen DTLN (~3.96 MB)"
                )

        active_engine = self.denoise_engine_var.get()
        if active_engine in ("dtln", "hybrid"):
            self.dtln_usage_var.set(
                f"🟢 ACTIVE as Mic Noise Suppressor (Mode: {active_engine.upper()})"
                if is_en else
                f"🟢 AKTIF sebagai Peredam Bising Mic (Mode: {active_engine.upper()})"
            )
            if hasattr(self, "dtln_usage_label"):
                self.dtln_usage_label.configure(foreground=c["cyan"])
            if hasattr(self, "dtln_apply_btn"):
                self.dtln_apply_btn.configure(
                    text="Disable DTLN (Use Clarity Boost)" if is_en else "Matikan DTLN (Gunakan Clarity Boost)"
                )
        else:
            self.dtln_usage_var.set(
                f"⚪ Inactive (Current noise suppressor: {active_engine.upper()})"
                if is_en else
                f"⚪ Tidak Aktif (Peredam bising saat ini: {active_engine.upper()})"
            )
            if hasattr(self, "dtln_usage_label"):
                self.dtln_usage_label.configure(foreground=c["muted"])
            if hasattr(self, "dtln_apply_btn"):
                self.dtln_apply_btn.configure(
                    text="Activate DTLN as Noise Suppressor" if is_en else "Aktifkan DTLN Sebagai Peredam Bising"
                )

    def _toggle_dtln_denoiser(self) -> None:
        cache = cache_for(self.app_dir, "dtln")
        state, _ = inspect_model("dtln", cache)
        is_en = getattr(self.config, "ui_language", "id") == "en"
        if state != "Tersedia lokal":
            if is_en:
                ans = messagebox.askyesno(
                    "DTLN Component Missing",
                    "DTLN model file (~3.96 MB) not found on disk.\nDo you want to download it now?",
                    parent=self.root,
                )
            else:
                ans = messagebox.askyesno(
                    "Komponen DTLN Belum Ada",
                    "File model DTLN (~3.96 MB) belum ada di disk.\nApakah Anda ingin mengunduhnya sekarang?",
                    parent=self.root,
                )
            if ans:
                self._download_dtln_component()
            return

        current = self.denoise_engine_var.get()
        if current in ("dtln", "hybrid"):
            new_mode = "clarity"
            self.denoise_engine_var.set(new_mode)
            self._set_activity("DTLN noise suppressor disabled (switched to Clarity Boost)" if is_en else "Peredam bising DTLN dimatikan (beralih ke Clarity Boost)", "ok")
        else:
            new_mode = "dtln"
            self.denoise_engine_var.set(new_mode)
            self._set_activity("DTLN noise suppressor activated!" if is_en else "Peredam bising DTLN diaktifkan!", "ok")

        self.config.denoise_engine = new_mode
        try:
            self.config_store.save(self.config)
        except Exception:
            pass
        self._refresh_dtln_status()

    def _download_dtln_component(self) -> None:
        cache = cache_for(self.app_dir, "dtln")
        if hasattr(self, "dtln_download_btn"):
            self.dtln_download_btn["state"] = "disabled"

        self.download_pbar.configure(mode="indeterminate", value=0)
        self.download_pbar.start(12)
        self.dl_status_var.set("Menghubungkan untuk unduhan DTLN Neural Denoiser…")
        self.dl_speed_var.set("Memulai…")
        self.dl_size_var.set("Alokasi ~3.96 MiB…")
        self.dl_eta_var.set("")

        def task():
            def cb(p):
                state = p.get("state") if isinstance(p, dict) else getattr(p, "state", "")
                pct = (p.get("percent") if isinstance(p, dict) else getattr(p, "percent", 0.0)) or 0.0
                speed_str = (p.get("speed_str") if isinstance(p, dict) else getattr(p, "speed_str", "")) or ""
                size_str = (p.get("size_str") if isinstance(p, dict) else getattr(p, "size_str", "")) or ""
                eta_str = (p.get("eta_str") if isinstance(p, dict) else getattr(p, "eta_str", "")) or ""

                def update_ui():
                    if state == "Mengunduh":
                        if self.download_pbar["mode"] == "indeterminate":
                            self.download_pbar.stop()
                            self.download_pbar.configure(mode="determinate")
                        self.download_pbar.configure(value=max(1.0, min(100.0, pct)))
                        self.dl_status_var.set(f"Mengunduh DTLN ONNX • {pct:.1f}%")
                        self.dl_speed_var.set(f"⚡ {speed_str}")
                        self.dl_size_var.set(f"📦 {size_str}")
                        self.dl_eta_var.set(f"⏳ {eta_str}" if eta_str else "")
                    elif state == "Memverifikasi":
                        self.download_pbar.configure(mode="indeterminate")
                        self.download_pbar.start(15)
                        self.dl_status_var.set("Memverifikasi file DTLN…")
                        self.dl_speed_var.set("Verifikasi ONNX")
                    elif state == "Tersedia lokal":
                        self.download_pbar.stop()
                        self.download_pbar.configure(mode="determinate", value=100)
                        self.dl_status_var.set("✔ Komponen DTLN selesai diunduh & siap digunakan!")
                        self.dl_speed_var.set("Selesai (100%)")
                        self.dl_size_var.set("3.96 MiB siap")
                        self.dl_eta_var.set("")

                self.root.after(0, update_ui)

            try:
                ensure_model("dtln", cache, cb)
                self.root.after(0, lambda: self._set_activity("DTLN Neural Denoiser siap digunakan", "ok"))
                self.root.after(0, self._refresh_dtln_status)
            except Exception as e:
                err_msg = str(e)
                def on_err(msg=err_msg):
                    self.download_pbar.stop()
                    self.download_pbar.configure(mode="determinate", value=0)
                    self.dl_status_var.set(f"❌ Gagal mengunduh DTLN: {msg}")
                self.root.after(0, on_err)
            finally:
                self.root.after(0, lambda: getattr(self, "dtln_download_btn", None) and self.dtln_download_btn.__setitem__("state", "normal"))

        threading.Thread(target=task, daemon=True).start()

    def _load(self, config: AppConfig) -> None:
        self._preset_internal_change = True
        try:
            self.source_var.set(source_display_name(config.source_language))
            for index, variable in enumerate(self.target_vars):
                variable.set(config.targets[index].language if index < len(config.targets) else UNUSED_TARGET)
            self.model_var.set(config.whisper_model)
            self.beam_var.set(str(config.whisper_beam_size))
            self.hotwords_var.set(config.whisper_hotwords)
            self.stt_device_var.set(config.stt_device)
            self.stt_compute_type_var.set(getattr(config, "stt_compute_type", "auto"))
            self.mt_device_var.set(config.mt_device)
            self.mt_compute_type_var.set(getattr(config, "mt_compute_type", "auto"))
            self.mt_beam_size_var.set(str(config.mt_beam_size))
            self.cpu_threads_var.set(str(config.cpu_threads))
            self.audio_channel_var.set(config.audio_channel)
            self.audio_gain_var.set(config.audio_gain_db)
            self.normalize_audio_var.set(config.normalize_audio)
            self.audio_clarity_var.set(getattr(config, "audio_clarity", True))
            self.slang_normalization_var.set(getattr(config, "slang_normalization", True))
            self.profanity_filter_var.set(getattr(config, "profanity_filter", False))
            self.denoise_engine_var.set(getattr(config, "denoise_engine", "clarity"))
            self.overlay_port_var.set(str(config.overlay.port))
            self.theme_var.set(config.overlay.theme)
            self.font_var.set(config.overlay.font_family)
            self.font_size_var.set(str(config.overlay.font_size))
            self.text_color_var.set(config.overlay.text_color)
            self.outline_color_var.set(config.overlay.outline_color)
            self.vad_var.set(config.vad_threshold)
            self.silence_var.set(config.min_silence_ms)
            self.duration_var.set(config.max_utterance_seconds)
            self.timeout_var.set(config.caption_timeout_seconds)

            preset = getattr(config, "resource_preset", "medium")
            self.resource_preset_var.set(preset)
            self.ui_mode_var.set(getattr(config, "ui_mode", "ez"))
            self._sync_mode_ui()
            self._sync_preset_badge(preset)
            self._update_preset_button_styles()

            self._update_monitor_languages()
            self._rebuild_overlay_rows()
            self._update_vram_status()
            self._update_model_lang_hints()
        finally:
            self._preset_internal_change = False

    def _update_vram_status(self) -> None:
        if not hasattr(self, "vram_badge_label"):
            return
        lang = getattr(self.config, "ui_language", "id")
        status, badge, detail = evaluate_vram_safety(
            self.model_var.get() or self.config.whisper_model,
            self.config.nllb_model,
            self.stt_device_var.get() or self.config.stt_device,
            self.mt_device_var.get() or self.config.mt_device,
            self.stt_compute_type_var.get() or getattr(self.config, "stt_compute_type", "auto"),
            self.mt_compute_type_var.get() or getattr(self.config, "mt_compute_type", "auto"),
            lang=lang,
        )
        fg_map = {
            "safe": self.colors.get("green", "#58D6A8"),
            "caution": self.colors.get("amber", "#F0BA66"),
            "danger": self.colors.get("red", "#FF718D"),
            "cpu": self.colors.get("cyan", "#4DD8E7"),
        }
        self.vram_badge_label.configure(text=badge, foreground=fg_map.get(status, self.colors["text"]))
        self.vram_detail_label.configure(text=detail)

    def _update_model_lang_hints(self) -> None:
        if not hasattr(self, "whisper_lang_hint_label"):
            return
        model = self.model_var.get() or self.config.whisper_model
        hint_keys = {
            "whisper-small-id": ("hint_model_whisper_small_id", self.colors.get("green", "#58D6A8")),
            "whisper-medium-id": ("hint_model_whisper_medium_id", self.colors.get("cyan", "#4DD8E7")),
            "small": ("hint_model_small", self.colors.get("green", "#58D6A8")),
            "base": ("hint_model_base", self.colors.get("text", "#F4F6FF")),
            "tiny": ("hint_model_tiny", self.colors.get("muted", "#8C96B4")),
            "medium": ("hint_model_medium", self.colors.get("cyan", "#4DD8E7")),
            "large-v3-turbo": ("hint_model_large_v3_turbo", self.colors.get("cyan", "#4DD8E7")),
            "distil-large-v3": ("hint_model_distil_large_v3", self.colors.get("amber", "#F0BA66")),
            "large-v3": ("hint_model_large_v3", self.colors.get("text", "#F4F6FF")),
        }
        hint_key, fg = hint_keys.get(model, ("hint_model_default", self.colors.get("muted", "#8C96B4")))
        text = self.t(hint_key)
        if model in CATALOG:
            cache = cache_for(self.app_dir, model)
            status, _ = inspect_model(model, cache)
            if status == "Belum ada":
                disk_prefix = self.t("disk_status_missing_msg", model=model, size=CATALOG[model].disk_mib)
                text = disk_prefix + text
                fg = self.colors.get("red", "#FF718D")
            elif status == "Belum lengkap":
                disk_prefix = self.t("disk_status_incomplete_msg", model=model)
                text = disk_prefix + text
                fg = self.colors.get("amber", "#F0BA66")
            else:
                text = f"{self.t('disk_status_ready_prefix')}{text}"
                fg = self.colors.get("green", "#58D6A8")

        self.whisper_lang_hint_label.configure(text=text, foreground=fg)

    def _update_scroll_hint(self, first: float, last: float) -> None:
        if not hasattr(self, "scroll_hint_label"):
            return
        if (last - first) >= 0.98:
            self.scroll_hint_label.configure(
                text=self.t("scroll_hint_all"),
                foreground=self.colors.get("muted", "#8C96B4"),
            )
        elif first <= 0.02:
            self.scroll_hint_label.configure(
                text=self.t("scroll_hint_more"),
                foreground=self.colors.get("cyan", "#4DD8E7"),
            )
        elif last >= 0.98:
            self.scroll_hint_label.configure(
                text=self.t("scroll_hint_bottom"),
                foreground=self.colors.get("muted", "#8C96B4"),
            )
        else:
            pct = int((first + last) / 2 * 100)
            self.scroll_hint_label.configure(
                text=self.t("scroll_hint_pos", pct=pct),
                foreground=self.colors.get("violet", "#806CFF"),
            )

    def set_ui_mode(self, mode: str) -> None:
        self.ui_mode_var.set(mode)
        self._sync_mode_ui()
        if mode == "advanced":
            self.notebook.select(self.tab_engine)
        else:
            self.notebook.select(0)

    def _sync_mode_ui(self) -> None:
        mode = self.ui_mode_var.get()
        if not hasattr(self, "btn_mode_ez") or not hasattr(self, "btn_mode_adv"):
            return
        if mode == "ez":
            self.btn_mode_ez.configure(style="ModeActive.TButton")
            self.btn_mode_adv.configure(style="TButton")
        else:
            self.btn_mode_ez.configure(style="TButton")
            self.btn_mode_adv.configure(style="ModeActive.TButton")

    def apply_resource_preset(self, preset_key: str) -> None:
        self._preset_internal_change = True
        try:
            source_lang = self.source_var.get().strip().casefold()
            is_indo = ("indonesia" in source_lang)
            key = preset_key.strip().casefold()
            is_en = getattr(self.config, "ui_language", "id") == "en"

            if key in ("low", "hemat"):
                self.resource_preset_var.set("Hemat")
                self.model_var.set("base")
                self.beam_var.set("1")
                self.stt_compute_type_var.set("int8")
                self.stt_device_var.set("auto")
                self.denoise_engine_var.set("clarity")
                self.slang_normalization_var.set(True)
                self.audio_clarity_var.set(True)
                if is_en:
                    self.preset_badge_var.set("🟢 LOW SPEC (SAVER)")
                    self.preset_desc_var.set("⚡ Ultra Light • Base Model • Int8 • 1 Beam • Low CPU & RAM")
                else:
                    self.preset_badge_var.set("🟢 HEMAT (LOW SPEC)")
                    self.preset_desc_var.set("⚡ Sangat Ringan • Model Base • Int8 • 1 Beam • Hemat CPU & RAM")

            elif key in ("medium", "balanced", "seimbang"):
                self.resource_preset_var.set("Seimbang")
                if is_indo:
                    self.model_var.set("whisper-small-id")
                    desc = "⭐ Recommended for ID • whisper-small-id • Int8_F16 • DTLN Neural Denoise" if is_en else "⭐ Rekomendasi Utama Indo • whisper-small-id • Int8_F16 • DTLN Neural Denoise"
                else:
                    self.model_var.set("small")
                    desc = "⭐ Recommended Balanced • Small Model • Int8_F16 • DTLN Neural Denoise" if is_en else "⭐ Rekomendasi Seimbang • Model Small • Int8_F16 • DTLN Neural Denoise"
                self.beam_var.set("3")
                self.stt_compute_type_var.set("int8_float16")
                self.stt_device_var.set("auto")
                self.denoise_engine_var.set("dtln")
                self.slang_normalization_var.set(True)
                self.audio_clarity_var.set(True)
                self.preset_badge_var.set("🔵 BALANCED (RECOMMENDED) ⭐" if is_en else "🔵 SEIMBANG (BALANCED) ⭐")
                self.preset_desc_var.set(desc)

            elif key in ("high", "accuracy", "akurasi"):
                self.resource_preset_var.set("Akurasi")
                if is_indo:
                    self.model_var.set("whisper-medium-id")
                    desc = "🎯 Max ID Accuracy • whisper-medium-id • Float16 • Hybrid Denoise" if is_en else "🎯 Akurasi Maksimal Indo • whisper-medium-id • Float16 • Hybrid Denoise"
                else:
                    self.model_var.set("large-v3-turbo")
                    desc = "🎯 Global 99+ Languages • large-v3-turbo • Float16 • Hybrid Denoise" if is_en else "🎯 Akurasi Global 99+ Bahasa • large-v3-turbo • Float16 • Hybrid Denoise"
                self.beam_var.set("5")
                self.stt_compute_type_var.set("float16")
                self.stt_device_var.set("auto")
                self.denoise_engine_var.set("hybrid")
                self.slang_normalization_var.set(True)
                self.audio_clarity_var.set(True)
                self.preset_badge_var.set("🟣 HIGH ACCURACY (HIGH SPEC)" if is_en else "🟣 AKURASI TINGGI (HIGH SPEC)")
                self.preset_desc_var.set(desc)

            self._update_preset_button_styles()
            self._update_vram_status()
            self._update_model_lang_hints()
        finally:
            self._preset_internal_change = False

    def _update_preset_button_styles(self) -> None:
        if not hasattr(self, "btn_preset_low"):
            return
        preset = self.resource_preset_var.get().strip().casefold()
        is_en = getattr(self.config, "ui_language", "id") == "en"
        low_title = "⚡ Low Spec\nLightweight" if is_en else "⚡ Hemat\nLow Spec"
        med_title = "⭐ Recommended\nBalanced" if is_en else "⭐ Seimbang\nRecommended"
        high_title = "🎯 High Spec\nAccurate" if is_en else "🎯 Akurasi\nHigh Spec"

        if preset in ("low", "hemat"):
            self.btn_preset_low.configure(style="PresetActiveLow.TButton", text=f"[✔] {low_title}")
            self.btn_preset_med.configure(style="TButton", text=med_title)
            self.btn_preset_high.configure(style="TButton", text=high_title)
        elif preset in ("medium", "balanced", "seimbang"):
            self.btn_preset_low.configure(style="TButton", text=low_title)
            self.btn_preset_med.configure(style="PresetActiveMed.TButton", text=f"[✔] {med_title}")
            self.btn_preset_high.configure(style="TButton", text=high_title)
        elif preset in ("high", "accuracy", "akurasi"):
            self.btn_preset_low.configure(style="TButton", text=low_title)
            self.btn_preset_med.configure(style="TButton", text=med_title)
            self.btn_preset_high.configure(style="PresetActiveHigh.TButton", text=f"[✔] {high_title}")
        else:
            # Custom
            self.btn_preset_low.configure(style="TButton", text=low_title)
            self.btn_preset_med.configure(style="TButton", text=med_title)
            self.btn_preset_high.configure(style="TButton", text=high_title)

    def _sync_preset_badge(self, preset: str) -> None:
        source_lang = self.source_var.get().strip().casefold()
        is_indo = ("indonesia" in source_lang)
        key = preset.strip().casefold()
        is_en = getattr(self.config, "ui_language", "id") == "en"
        if key in ("low", "hemat"):
            self.preset_badge_var.set("🟢 LOW SPEC (SAVER)" if is_en else "🟢 HEMAT (LOW SPEC)")
            self.preset_desc_var.set("⚡ Ultra Light • Base Model • Int8 • 1 Beam • Low CPU & RAM" if is_en else "⚡ Sangat Ringan • Model Base • Int8 • 1 Beam • Hemat CPU & RAM")
        elif key in ("medium", "balanced", "seimbang"):
            self.preset_badge_var.set("🔵 BALANCED (RECOMMENDED) ⭐" if is_en else "🔵 SEIMBANG (BALANCED) ⭐")
            if is_indo:
                desc = "⭐ Recommended for ID • whisper-small-id • DTLN Neural Denoise" if is_en else "⭐ Rekomendasi Utama Indo • whisper-small-id • DTLN Neural Denoise"
            else:
                desc = "⭐ Recommended Balanced • Small Model • DTLN Neural Denoise" if is_en else "⭐ Rekomendasi Seimbang • Model Small • DTLN Neural Denoise"
            self.preset_desc_var.set(desc)
        elif key in ("high", "accuracy", "akurasi"):
            self.preset_badge_var.set("🟣 HIGH ACCURACY (HIGH SPEC)" if is_en else "🟣 AKURASI TINGGI (HIGH SPEC)")
            if is_indo:
                desc = "🎯 Max ID Accuracy • whisper-medium-id • Hybrid Denoise" if is_en else "🎯 Akurasi Maksimal Indo • whisper-medium-id • Hybrid Denoise"
            else:
                desc = "🎯 Global 99+ Languages • large-v3-turbo • Hybrid Denoise" if is_en else "🎯 Akurasi Global • large-v3-turbo • Hybrid Denoise"
            self.preset_desc_var.set(desc)
        else:
            self.preset_badge_var.set("⚡ CUSTOM" if is_en else "⚡ KUSTOM (CUSTOM)")
            self.preset_desc_var.set("⚙️ Settings manually configured in Engine & VAD Tab" if is_en else "⚙️ Pengaturan diatur secara manual melalui Tab Mesin & VAD")

    def _on_source_language_changed(self, *args) -> None:
        if self._preset_internal_change:
            return
        preset = self.resource_preset_var.get().strip().casefold()
        if preset in ("low", "medium", "high", "balanced", "accuracy", "hemat", "seimbang", "akurasi"):
            self.apply_resource_preset(preset)

    def _on_advanced_setting_modified(self, *args) -> None:
        if self._preset_internal_change:
            return
        is_en = getattr(self.config, "ui_language", "id") == "en"
        self.resource_preset_var.set("Custom")
        self.preset_badge_var.set("⚡ CUSTOM" if is_en else "⚡ KUSTOM (CUSTOM)")
        self.preset_desc_var.set("⚙️ Settings manually configured in Engine & VAD Tab" if is_en else "⚙️ Pengaturan diatur secara manual melalui Tab Mesin & VAD")
        self._update_preset_button_styles()
        self._update_vram_status()
        self._update_model_lang_hints()

    def _card(self, parent, title: str, subtitle: str, collapsible: bool = False) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=6)
        head = ttk.Frame(card, style="Inner.TFrame")
        head.pack(fill="x", pady=(0, 8))
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(head, text=subtitle, style="Muted.TLabel").grid(row=1, column=0, sticky="w")
        if collapsible:
            self.advanced_button = ttk.Button(head, text="Buka", style="Link.TButton")
            self.advanced_button.grid(row=0, column=1, rowspan=2, sticky="e")
        return card

    def _field_row(self, parent, label: str, *, compact: bool = False) -> ttk.Frame:
        row = ttk.Frame(parent, style="Inner.TFrame")
        row.pack(fill="x", pady=4)
        row.columnconfigure(1, weight=1)
        width = 17 if compact else 20
        ttk.Label(row, text=label, style="Card.TLabel", width=width, anchor="w").grid(
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
            self._advanced_body.pack(fill="x", pady=(8, 0))
            if hasattr(self, "advanced_button"):
                self.advanced_button.configure(text="Tutup")
        else:
            self._advanced_body.pack_forget()
            if hasattr(self, "advanced_button"):
                self.advanced_button.configure(text="Buka")
        self.body.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _apply_layout(self, width: int | None = None) -> None:
        if not hasattr(self, "canvas") or not hasattr(self, "body") or not self.canvas.winfo_exists():
            return
        w = width if width is not None else self.canvas.winfo_width()
        if w <= 10:
            w = max(400, self.root.winfo_width() - 40)
        self.canvas.itemconfigure(self._body_window, width=w)
        wide = w >= 1020
        self._wide_layout = wide
        self.body.columnconfigure(0, weight=1, uniform="body" if wide else "")
        self.body.columnconfigure(1, weight=1 if wide else 0, uniform="body" if wide else "")
        self.left.grid(row=0, column=0, sticky="new", padx=(0, 8) if wide else 0)
        self.right.grid(row=0 if wide else 1, column=1 if wide else 0,
                        sticky="new", padx=(8, 0) if wide else 0)
        self.body.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_layout(self, event) -> None:
        wide = event.width >= 1020
        if wide == self._wide_layout and hasattr(self, "left") and self.left.winfo_ismapped():
            self.canvas.itemconfigure(self._body_window, width=event.width)
            return
        self._apply_layout(event.width)

    def _apply_engine_layout(self, width: int | None = None) -> None:
        if not hasattr(self, "engine_canvas") or not hasattr(self, "engine_body") or not self.engine_canvas.winfo_exists():
            return
        w = width if width is not None else self.engine_canvas.winfo_width()
        if w <= 10:
            w = max(400, self.root.winfo_width() - 40)
        self.engine_canvas.itemconfigure(self._engine_body_window, width=w)
        wide = w >= 960
        self._wide_engine_layout = wide
        self.engine_body.columnconfigure(0, weight=1, uniform="eng" if wide else "")
        self.engine_body.columnconfigure(1, weight=1 if wide else 0, uniform="eng" if wide else "")
        self.engine_left.grid(row=0, column=0, sticky="new", padx=(0, 8) if wide else 0)
        self.engine_right.grid(
            row=0 if wide else 1,
            column=1 if wide else 0,
            sticky="new",
            padx=(8, 0) if wide else 0,
            pady=(0, 0) if wide else (8, 0),
        )
        self.engine_body.update_idletasks()
        self.engine_canvas.configure(scrollregion=self.engine_canvas.bbox("all"))

    def _resize_engine_layout(self, event) -> None:
        if not hasattr(self, "engine_canvas") or not hasattr(self, "engine_body"):
            return
        wide = event.width >= 960
        if getattr(self, "_wide_engine_layout", None) == wide and hasattr(self, "engine_left") and self.engine_left.winfo_ismapped():
            self.engine_canvas.itemconfigure(self._engine_body_window, width=event.width)
            return
        self._apply_engine_layout(event.width)

    def _update_engine_scroll_hint(self, first: float, last: float) -> None:
        if not hasattr(self, "engine_scroll_hint_label") or not self.engine_scroll_hint_label:
            return
        c = self.colors
        if first <= 0.01 and last >= 0.99:
            self.engine_scroll_hint_label.configure(
                text=self.t("engine_scroll_all"),
                foreground=c["muted"],
            )
        elif last >= 0.95:
            self.engine_scroll_hint_label.configure(
                text=self.t("engine_scroll_bottom"),
                foreground=c["green"],
            )
        else:
            pct_remaining = int((1.0 - last) * 100)
            self.engine_scroll_hint_label.configure(
                text=self.t("engine_scroll_more", pct=pct_remaining),
                foreground=c["cyan"],
            )

    def _is_widget_child_of(self, widget, ancestor) -> bool:
        curr = widget
        while curr:
            if curr == ancestor:
                return True
            curr = getattr(curr, "master", None)
        return False

    def _resize_footer(self, event) -> None:
        narrow = event.width < 650
        for column in range(4):
            self.footer.columnconfigure(column, weight=0 if column == 3 else 1, uniform="" if column == 3 else "actions")
        self.save_button.grid(row=1 if narrow else 0, column=0, sticky="ew", padx=(0, 4))
        self.test_button.grid(row=1 if narrow else 0, column=1, sticky="ew", padx=(4, 0) if narrow else 4)
        self.start_button.grid(row=0, column=0 if narrow else 2, columnspan=2 if narrow else 1,
                               sticky="ew", padx=0 if narrow else (4, 4), pady=(0, 8) if narrow else 0)
        self.credit_label.grid(row=2 if narrow else 0, column=0 if narrow else 3, sticky="e" if not narrow else "w", pady=(4, 0) if narrow else 0, padx=8)

    def _reveal_focus_idle(self, widget: Any) -> None:
        self._focus_idle_id = None
        if not getattr(self, "_closed", False) and hasattr(widget, "winfo_exists") and widget.winfo_exists():
            try:
                self._reveal_focus(SimpleNamespace(widget=widget))
            except Exception:
                pass

    def _on_key_tab(self, _event) -> None:
        focused = self.root.focus_get()
        if focused and isinstance(focused, tk.Widget) and not getattr(self, "_closed", False):
            if self._focus_idle_id is not None:
                try:
                    self.root.after_cancel(self._focus_idle_id)
                except Exception:
                    pass
            self._focus_idle_id = self.root.after_idle(self._reveal_focus_idle, focused)

    def _scroll_wheel(self, event):
        widget = getattr(event, "widget", None)
        if widget and any(isinstance(widget, cls) for cls in (ttk.Treeview, tk.Text, tk.Listbox)):
            return
        if getattr(event, "delta", 0):
            steps = max(1, abs(event.delta) // 120)
            direction = -steps * 2 if event.delta > 0 else steps * 2

            selected = ""
            try:
                selected = self.notebook.select()
            except Exception:
                pass

            if hasattr(self, "engine_canvas") and self.engine_canvas and self.engine_canvas.winfo_viewable():
                if selected == str(getattr(self, "tab_engine", None)) or (widget and self._is_widget_child_of(widget, getattr(self, "tab_engine", None))):
                    self.engine_canvas.yview_scroll(direction, "units")
                    return "break"

            if hasattr(self, "models_canvas") and self.models_canvas and self.models_canvas.winfo_viewable():
                if selected == str(getattr(self, "tab_models", None)) or (widget and self._is_widget_child_of(widget, getattr(self, "tab_models", None))):
                    self.models_canvas.yview_scroll(direction, "units")
                    return "break"

            if hasattr(self, "canvas") and self.canvas and self.canvas.winfo_viewable():
                if self.body.winfo_height() > self.canvas.winfo_height():
                    self.canvas.yview_scroll(direction, "units")
                    return "break"
                scroll_region = self.canvas.bbox("all")
                if scroll_region and (scroll_region[3] - scroll_region[1]) > self.canvas.winfo_height():
                    self.canvas.yview_scroll(direction, "units")
                    return "break"
        return "break"

    def _reveal_focus(self, event) -> None:
        widget = getattr(event, "widget", None)
        if not widget or not widget.winfo_viewable():
            return
        for cv, body in (
            (getattr(self, "canvas", None), getattr(self, "body", None)),
            (getattr(self, "engine_canvas", None), getattr(self, "engine_body", None)),
            (getattr(self, "models_canvas", None), getattr(self, "models_container", None)),
        ):
            if cv and cv.winfo_viewable() and body and self._is_widget_child_of(widget, body):
                widget_top = widget.winfo_rooty()
                widget_bottom = widget_top + widget.winfo_height()
                canvas_top = cv.winfo_rooty()
                canvas_bottom = canvas_top + cv.winfo_height()
                if widget_top < canvas_top or widget_bottom > canvas_bottom:
                    scrollregion = cv.bbox("all")
                    if scrollregion:
                        total_height = scrollregion[3] - scrollregion[1]
                        if total_height > 0:
                            body_y = widget.winfo_rooty() - body.winfo_rooty()
                            relative_y = body_y / total_height
                            cv.yview_moveto(max(0.0, min(1.0, relative_y - 0.05)))
                break

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
                profiles=self.config.overlay.profiles,
                anchor=self.config.overlay.anchor,
                gap=self.config.overlay.gap,
                order=self.config.overlay.order,
            ),
            whisper_model=self.model_var.get(),
            whisper_beam_size=int(self.beam_var.get()),
            whisper_hotwords=self.hotwords_var.get().strip(),
            stt_device=self.stt_device_var.get(),
            stt_compute_type=self.stt_compute_type_var.get(),
            mt_device=self.mt_device_var.get(),
            mt_compute_type=self.mt_compute_type_var.get(),
            nllb_model=self.config.nllb_model,
            vad_threshold=round(self.vad_var.get(), 2),
            min_silence_ms=int(self.silence_var.get()),
            min_speech_ms=self.config.min_speech_ms,
            max_utterance_seconds=int(self.duration_var.get()),
            caption_timeout_seconds=int(self.timeout_var.get()),
            audio_channel=self.audio_channel_var.get(),
            audio_gain_db=float(self.audio_gain_var.get()),
            normalize_audio=bool(self.normalize_audio_var.get()),
            audio_clarity=bool(self.audio_clarity_var.get()),
            slang_normalization=bool(self.slang_normalization_var.get()),
            profanity_filter=bool(self.profanity_filter_var.get()),
            denoise_engine=str(self.denoise_engine_var.get()),
            cpu_threads=int(self.cpu_threads_var.get()),
            mt_beam_size=int(self.mt_beam_size_var.get()),
            resource_preset=self.resource_preset_var.get(),
            ui_mode=self.ui_mode_var.get(),
            vocabulary_packages=self.config.vocabulary_packages,
            regional_assistance=self.config.regional_assistance,
            ui_theme=self._theme_name,
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
            if not getattr(self, "_closed", False):
                if self._scroll_idle_id is not None:
                    try:
                        self.root.after_cancel(self._scroll_idle_id)
                    except Exception:
                        pass
                self._scroll_idle_id = self.root.after_idle(self._restore_scroll, scroll_pos)
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
            tuple(target.language for target in config.targets),
            tuple(target.profile for target in config.targets),
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
            is_en = (getattr(self.config, "ui_language", "id") == "en")
            active_prefix = "Active local" if is_en else "Aktif lokal"
            self.overlay_state_var.set(f"{active_prefix} • 127.0.0.1:{self.config.overlay.port}")
            self._update_overlay_diagnostics()
            return True
        except Exception as exc:
            is_en = (getattr(self.config, "ui_language", "id") == "en")
            self.overlay_state_var.set(f"Server failed • {exc}" if is_en else f"Server gagal • {exc}")
            self.overlay_clients_var.set("0 connected (server inactive)" if is_en else "0 terhubung (server nonaktif)")
            self._overlay_fingerprint = None
            if show_error:
                messagebox.showerror("Overlay Error" if is_en else "Overlay tidak dapat dimulai", str(exc), parent=self.root)
            return False

    def _update_overlay_diagnostics(self, client_count: int | None = None) -> None:
        is_en = (getattr(self.config, "ui_language", "id") == "en")
        if client_count is not None:
            clients = client_count
        elif self.overlay_service and self.overlay_service.running:
            diag = self.overlay_service.get_diagnostics()
            clients = diag.get("clients", 0)
        else:
            self.overlay_clients_var.set("0 connected (server inactive)" if is_en else "0 terhubung (server nonaktif)")
            return

        if clients == 0:
            self.overlay_clients_var.set(self.t("obs_clients_count_fmt", count=0))
        elif clients == 1:
            self.overlay_clients_var.set("1 browser source connected" if is_en else "1 browser source terhubung")
        else:
            self.overlay_clients_var.set(f"{clients} browser sources connected" if is_en else f"{clients} browser source terhubung")

        if self.overlay_service and self.overlay_service.running:
            diag = self.overlay_service.get_diagnostics()
            last_time = diag.get("last_publish_time", 0.0)
            last_clients = diag.get("last_publish_clients", 0)
            if last_time > 0:
                import datetime
                time_str = datetime.datetime.fromtimestamp(last_time).strftime("%H:%M:%S")
                if is_en:
                    self.overlay_activity_var.set(f"Sent at {time_str} ({last_clients} receivers)")
                else:
                    self.overlay_activity_var.set(f"Terkirim pk {time_str} ({last_clients} penerima)")
            else:
                self.overlay_activity_var.set(self.t("obs_no_caption_sent"))

    def toggle(self) -> None:
        if self._stopping:
            if self._closing_since and time.monotonic() - self._closing_since > 1.0:
                if self.pipeline:
                    try:
                        self.pipeline.request_stop()
                    except Exception:
                        pass
                self.pipeline = None
                self._stopping = False
                self._set_live_controls(False)
                self._set_status("SIAP", GREEN)
                self._set_activity("Caption dihentikan", "ok")
                return
            else:
                return
        if self.pipeline and (self.pipeline.running or self.pipeline.inference_busy):
            self.pipeline.request_stop()
            self._stopping = True
            self._closing_since = time.monotonic()
            self._set_status("MENUTUP", AMBER)
            self._set_activity("Capture dihentikan; menutup mesin caption…", "warning")
            self.start_button["state"] = "disabled"
            self.start_button.configure(text="MENUTUP MESIN")
            return
        config = self.save(announce=False)
        if not config or not self._sync_overlay(show_error=True):
            return

        model_name = config.whisper_model
        if model_name in CATALOG:
            cache = cache_for(self.app_dir, model_name)
            state, _ = inspect_model(model_name, cache)
            if state != "Tersedia lokal":
                is_en = (getattr(config, "ui_language", "id") == "en")
                msg = (
                    f"Model '{model_name}' belum diunduh ke disk (~{CATALOG[model_name].disk_mib:.0f} MiB).\n\n"
                    f"Silakan unduh model ini terlebih dahulu melalui Tab 'Model & Resource'."
                    if not is_en else
                    f"Model '{model_name}' has not been downloaded yet (~{CATALOG[model_name].disk_mib:.0f} MiB).\n\n"
                    f"Please download it first in 'Model & Resource' tab."
                )
                messagebox.showwarning(
                    "Model Belum Tersedia" if not is_en else "Model Missing",
                    msg,
                    parent=self.root,
                )
                if hasattr(self, "notebook") and hasattr(self, "tab_models"):
                    self.notebook.select(self.tab_models)
                return

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
        if not self._closed:
            self._poll_timer = self.root.after(80, self._poll_events)

    def _finish_stop(self) -> None:
        if self.pipeline and (self.pipeline.running or self.pipeline.inference_busy):
            self.start_button["state"] = "disabled"
            self.start_button.configure(text="MENUTUP MESIN")
            if not self._closing_since:
                self._closing_since = time.monotonic()
            if time.monotonic() - self._closing_since >= 2.0:
                self.runtime_var.set("Shutdown selesai")
                self.pipeline = None
                self._stopping = False
                self._set_live_controls(False)
                self._set_status("SIAP", GREEN)
                self._set_activity("Caption dihentikan", "ok")
            return
        self.pipeline = None
        self._stopping = False
        self._set_live_controls(False)
        if self._pipeline_error:
            self._set_status("ERROR", RED)
            self._set_activity(self._pipeline_error, "error")
            self.runtime_var.set(f"❌ Error: {self._pipeline_error}")
        else:
            self._set_status("SIAP", GREEN)
            self._set_activity("Caption dihentikan", "ok")

    def _reset_audio(self) -> None:
        self._level_history.clear()
        self._smoothed_db = None
        self._last_speech_at = 0.0
        self._vad_probability = 0.0
        self._vad_speaking = False
        if hasattr(self, "audio_meter"):
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
        if hasattr(self, "audio_meter"):
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
                self.activity_var.set("Inisialisasi sistem & model AI…")
                self.runtime_var.set(event.message)
        elif event.kind == "model_ready":
            self.runtime_var.set(
                f"Whisper {data.get('model', '')} • STT {data.get('stt_device', '').upper()} • MT {data.get('mt_device', '').upper()}"
            )
        elif event.kind == "metrics":
            stt_ms = data.get("stt_ms", 0.0)
            mt_ms = data.get("mt_ms", 0.0)
            total_ms = data.get("after_vad_ms", 0.0)
            self.metrics_var.set(
                f"⚡ STT {stt_ms:.0f} ms • MT {mt_ms:.0f} ms | Respon: {total_ms:.0f} ms (di luar jeda hening)"
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
            self.runtime_var.set(event.message)
            self._set_activity(event.message, "working")
        elif event.kind == "transcript":
            self._processing = False
            text = str(data.get("text", event.message))
            lang = data.get("language", "id")
            self.transcript_var.set(f"[{lang}] {text}" if "language" in data else text)
            self._set_activity("Ucapan dikenali • menerjemahkan…", "working")
        elif event.kind == "translations":
            self._processing = False
            self._last_translations = {str(k): str(v) for k, v in data.items()}
            self._show_translations(self._last_translations)
            self._set_activity("Terjemahan siap • mengirim overlay…", "working")
        elif event.kind == "published":
            self._processing = False
            self._set_activity("Caption terkirim ke server overlay", "ok")
            self._update_overlay_diagnostics()
        elif event.kind == "overlay_ready":
            is_en = (getattr(self.config, "ui_language", "id") == "en")
            active_prefix = "Active local" if is_en else "Aktif lokal"
            self.overlay_state_var.set(f"{active_prefix} • 127.0.0.1:{self.config.overlay.port}")
            self._update_overlay_diagnostics()
        elif event.kind == "clients_changed":
            count = int(event.message) if event.message.isdigit() else None
            self._update_overlay_diagnostics(count)
        elif event.kind == "model_status":
            self.model_progress_var.set(event.message)
        elif event.kind == "learning":
            self.vocab_manager.load()
            self._update_vocab_status()
            self._set_activity(event.message, "ok")
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
            self.runtime_var.set(f"❌ Error: {event.message}")
            self._stopping = False
            self._set_live_controls(False)
            self.start_button.configure(
                text=self.t("btn_start_caption_caps"),
                style="Accent.TButton",
                state="normal",
            )
            self._reset_audio()
        elif event.kind in {"warning", "background"}:
            self._set_activity(event.message, "warning")
            if not self._stopping and self.status_var.get() == "MENYIAPKAN":
                self.runtime_var.set(f"Proses: {event.message}")

    def _set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    def _set_activity(self, message: str, kind: str = "idle") -> None:
        self.activity_var.set(message)

    def _show_translations(self, translations: dict[str, str]) -> None:
        self._last_translations = translations
        for index, target in enumerate(self.config.targets):
            if index < len(self.monitor_text_vars):
                self.monitor_text_vars[index].set(translations.get(target.language, "—"))

    def _update_monitor_languages(self) -> None:
        texts: list[str] = []
        for index in range(len(self.monitor_language_labels)):
            if index < len(self.config.targets):
                texts.append(f"Output {self.config.targets[index].profile} ({self.config.targets[index].language})")
            else:
                texts.append("—")
        max_len = max((len(t) for t in texts), default=12)
        for label, text in zip(self.monitor_language_labels, texts):
            label.configure(text=text, width=max(max_len, 12), anchor="w")

    def _set_live_controls(self, live: bool) -> None:
        for widget, idle_state in self._editable:
            widget["state"] = "disabled" if live else idle_state
        self.save_button["state"] = "disabled" if live else "normal"
        self.test_button["state"] = "disabled" if live else "normal"
        self.start_button["state"] = "normal"
        self.start_button.configure(
            text=self.t("btn_stop_caption_caps") if live else self.t("btn_start_caption_caps"),
            style="Danger.TButton" if live else "Accent.TButton",
        )

    def _rebuild_overlay_rows(self) -> None:
        if hasattr(self, "obs_url_var"):
            self.obs_url_var.set(self.static_overlay_url())
        if not getattr(self, "overlay_rows_container", None):
            return
        for child in self.overlay_rows_container.winfo_children():
            child.destroy()

        all_row = ttk.Frame(self.overlay_rows_container, style="Soft.TFrame", padding=(11, 8))
        all_row.pack(fill="x", pady=3)
        all_row.columnconfigure(0, weight=1)
        is_en = getattr(self.config, "ui_language", "id") == "en"
        all_title = "★ All-in-One Multilingual (All Slots)" if is_en else "★ Multi-Bahasa Sekaligus (All-in-One)"
        self._wrapped_label(
            all_row, text=all_title, style="Soft.TLabel",
            foreground=self.colors["cyan"], font=("Segoe UI Semibold", 9),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(all_row, text=self.t("btn_copy_link"), command=lambda: self.copy_url("all")).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(all_row, text=self.t("btn_open_browser"), command=lambda: self.open_preview("all")).grid(row=0, column=2)

    def static_overlay_url(self) -> str:
        return static_overlay_url(self.config.overlay)

    def overlay_url(self, language: str = "", profile: int | str = "") -> str:
        return overlay_url(self.config.overlay, language=language, profile=profile)

    def copy_url(self, language: str) -> None:
        self.root.clipboard_clear()
        url = self.static_overlay_url() if language in ("all", "static", "") else self.overlay_url(language)
        self.root.clipboard_append(url)
        self.root.update_idletasks()
        label = "Multi-Bahasa (All)" if language in ("all", "static", "") else language
        self._set_activity(f"URL {label} disalin", "ok")

    def copy_url_slot(self, slot: int) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.overlay_url(profile=slot))
        self.root.update_idletasks()
        self._set_activity(f"URL Output {slot} disalin", "ok")

    def open_preview(self, language: str) -> None:
        if self._sync_overlay(show_error=True) and self.overlay_service:
            if language in ("all", "static", ""):
                webbrowser.open(f"{self.static_overlay_url()}?preview=1")
            else:
                webbrowser.open(self.overlay_service.preview_url(language))

    def open_preview_slot(self, slot: int) -> None:
        if self._sync_overlay(show_error=True) and self.overlay_service:
            webbrowser.open(self.overlay_service.preview_url(profile=slot))

    def test_overlay(self) -> None:
        config = self.save(announce=False)
        if not config or not self._sync_overlay(show_error=True) or not self.overlay_service:
            return
        samples = {
            target.language: f"{target.language} • KizCaption siap tampil di OBS"
            for target in config.targets
        }
        try:
            self.overlay_service.publish_now(samples)
            self._last_translations = samples
            self._show_translations(samples)
            self._update_overlay_diagnostics()
            diag = self.overlay_service.get_diagnostics()
            connected = diag.get("clients", 0)
            if connected > 0:
                self._set_activity(f"Caption tes terkirim ke {connected} browser source aktif", "ok")
            else:
                self._set_activity("Caption tes dikirim (0 browser source terhubung, buka OBS)", "warning")
        except Exception as exc:
            messagebox.showerror("Tes overlay gagal", str(exc), parent=self.root)

    def _restore_scroll(self, pos: float) -> None:
        self._scroll_idle_id = None
        if not getattr(self, "_closed", False) and hasattr(self, "canvas") and self.canvas.winfo_exists():
            try:
                self.canvas.yview_moveto(pos)
            except Exception:
                pass

    def _deferred_sync_overlay(self) -> None:
        if not getattr(self, "_closed", False):
            self._sync_overlay(show_error=False)

    def _on_root_destroy(self, event: Any) -> None:
        if getattr(event, "widget", None) == self.root:
            self.close()

    def close(self) -> None:
        self._closed = True
        if self._poll_timer is not None:
            try:
                self.root.after_cancel(self._poll_timer)
            except Exception:
                pass
            self._poll_timer = None
        if getattr(self, "_overlay_timer", None) is not None:
            try:
                self.root.after_cancel(self._overlay_timer)
            except Exception:
                pass
            self._overlay_timer = None
        if getattr(self, "_scroll_idle_id", None) is not None:
            try:
                self.root.after_cancel(self._scroll_idle_id)
            except Exception:
                pass
            self._scroll_idle_id = None
        if getattr(self, "_focus_idle_id", None) is not None:
            try:
                self.root.after_cancel(self._focus_idle_id)
            except Exception:
                pass
            self._focus_idle_id = None
        if self.pipeline:
            try:
                self.pipeline.request_stop()
            except Exception:
                pass
            self.pipeline = None
        if self.overlay_service:
            try:
                self.overlay_service.stop(timeout=1.0)
            except Exception:
                pass
            self.overlay_service = None

    @staticmethod
    def _wrapped_label(parent, **kwargs) -> ttk.Label:
        label = ttk.Label(parent, width=1, anchor="w", justify="left", **kwargs)
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(40, event.width)))
        return label
