"""Caption Style Editor widget for LumaCaption profiles."""
from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from tkinter.font import Font
from typing import Callable

from lumacaption.output.styles import (
    ANCHORS,
    CaptionStyle,
    CHOICES,
    FONTS,
    NUMBERS,
    PLATFORM_LAYOUTS,
    PRESETS,
    preset_style,
)
from lumacaption.output.style_io import export_css, import_css
from lumacaption.i18n import t


def _draw_round_rect(canvas: tk.Canvas, x0: float, y0: float, x1: float, y1: float, radius: float, fill: str, outline: str = "", width: int = 1) -> None:
    if radius <= 3 or x1 <= x0 or y1 <= y0:
        canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=width)
        return
    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    points = [
        x0 + r, y0,
        x1 - r, y0,
        x1, y0,
        x1, y0 + r,
        x1, y1 - r,
        x1, y1,
        x1 - r, y1,
        x0 + r, y1,
        x0, y1,
        x0, y1 - r,
        x0, y0 + r,
        x0, y0,
    ]
    canvas.create_polygon(points, smooth=True, fill=fill, outline=outline, width=width)


class CaptionEditor(ttk.Frame):
    def __init__(
        self,
        master,
        slot: int,
        get_style: Callable[[int], CaptionStyle],
        on_change: Callable[[int, CaptionStyle], None],
        lang: str = "id",
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.slot = slot
        self.lang = lang
        self._get_style = get_style
        self._on_change = on_change
        self._updating = False

        self._build_ui()
        self.load_from_style(self._get_style(self.slot))

    def t(self, key: str, **kwargs) -> str:
        return t(key, self.lang, **kwargs)

    def _build_ui(self) -> None:
        # 1. Preset bar
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(top_bar, text=self.t("editor_slot_label", slot=self.slot), font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(top_bar, textvariable=self.name_var, width=15)
        self.name_entry.pack(side="left", padx=(0, 12))
        self.name_var.trace_add("write", self._on_field_changed)

        ttk.Label(top_bar, text=self.t("editor_preset_label")).pack(side="left", padx=(0, 4))
        self.preset_var = tk.StringVar()
        preset_names = list(PRESETS.keys())
        self.preset_combo = ttk.Combobox(top_bar, textvariable=self.preset_var, values=preset_names, state="readonly", width=18)
        self.preset_combo.pack(side="left", padx=(0, 12))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Button(top_bar, text=self.t("btn_export_css"), command=self._export_css).pack(side="right", padx=2)
        ttk.Button(top_bar, text=self.t("btn_import_css"), command=self._import_css).pack(side="right", padx=2)

        # 2. Controls Notebook or Cards
        controls = ttk.Notebook(self)
        controls.pack(fill="both", expand=True, padx=8, pady=4)

        # Tab: Tipografi
        tab_typo = ttk.Frame(controls, padding=8)
        controls.add(tab_typo, text=self.t("editor_tab_typography"))

        ttk.Label(tab_typo, text=self.t("editor_font_family")).grid(row=0, column=0, sticky="w", pady=3)
        self.font_var = tk.StringVar()
        self.font_combo = ttk.Combobox(tab_typo, textvariable=self.font_var, values=list(FONTS), width=20)
        self.font_combo.grid(row=0, column=1, sticky="w", pady=3)
        self.font_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_typo, text=self.t("editor_font_size")).grid(row=1, column=0, sticky="w", pady=3)
        self.size_var = tk.DoubleVar()
        self.size_spin = ttk.Spinbox(tab_typo, from_=12, to=192, textvariable=self.size_var, width=8)
        self.size_spin.grid(row=1, column=1, sticky="w", pady=3)
        self.size_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_typo, text=self.t("editor_font_weight")).grid(row=2, column=0, sticky="w", pady=3)
        self.weight_var = tk.IntVar()
        self.weight_combo = ttk.Combobox(tab_typo, textvariable=self.weight_var, values=[100, 300, 400, 600, 700, 800, 900], width=8, state="readonly")
        self.weight_combo.grid(row=2, column=1, sticky="w", pady=3)
        self.weight_var.trace_add("write", self._on_field_changed)

        self.italic_var = tk.BooleanVar()
        ttk.Checkbutton(tab_typo, text=self.t("editor_italic"), variable=self.italic_var, command=self._on_field_changed).grid(row=3, column=0, columnspan=2, sticky="w", pady=3)

        ttk.Label(tab_typo, text=self.t("editor_letter_spacing")).grid(row=4, column=0, sticky="w", pady=3)
        self.letter_spacing_var = tk.DoubleVar()
        ttk.Spinbox(tab_typo, from_=-2, to=12, textvariable=self.letter_spacing_var, width=8).grid(row=4, column=1, sticky="w", pady=3)
        self.letter_spacing_var.trace_add("write", self._on_field_changed)

        # Tab: Warna & Garis Tepi
        tab_colors = ttk.Frame(controls, padding=8)
        controls.add(tab_colors, text=self.t("editor_tab_colors"))

        ttk.Label(tab_colors, text=self.t("editor_text_color")).grid(row=0, column=0, sticky="w", pady=3)
        self.text_color_var = tk.StringVar()
        tc_frame = ttk.Frame(tab_colors)
        tc_frame.grid(row=0, column=1, sticky="w", pady=3)
        ttk.Entry(tc_frame, textvariable=self.text_color_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Button(tc_frame, text=self.t("editor_btn_choose"), command=lambda: self._choose_color(self.text_color_var)).pack(side="left")
        self.text_color_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_colors, text=self.t("editor_outline_color")).grid(row=1, column=0, sticky="w", pady=3)
        self.outline_color_var = tk.StringVar()
        oc_frame = ttk.Frame(tab_colors)
        oc_frame.grid(row=1, column=1, sticky="w", pady=3)
        ttk.Entry(oc_frame, textvariable=self.outline_color_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Button(oc_frame, text=self.t("editor_btn_choose"), command=lambda: self._choose_color(self.outline_color_var)).pack(side="left")
        self.outline_color_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_colors, text=self.t("editor_outline_width")).grid(row=2, column=0, sticky="w", pady=3)
        self.outline_width_var = tk.DoubleVar()
        ttk.Spinbox(tab_colors, from_=0, to=10, textvariable=self.outline_width_var, width=8).grid(row=2, column=1, sticky="w", pady=3)
        self.outline_width_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_colors, text=self.t("editor_shadow_color")).grid(row=3, column=0, sticky="w", pady=3)
        self.shadow_color_var = tk.StringVar()
        sc_frame = ttk.Frame(tab_colors)
        sc_frame.grid(row=3, column=1, sticky="w", pady=3)
        ttk.Entry(sc_frame, textvariable=self.shadow_color_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Button(sc_frame, text=self.t("editor_btn_choose"), command=lambda: self._choose_color(self.shadow_color_var)).pack(side="left")
        self.shadow_color_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_colors, text=self.t("editor_shadow_blur")).grid(row=4, column=0, sticky="w", pady=3)
        self.shadow_blur_var = tk.DoubleVar()
        ttk.Spinbox(tab_colors, from_=0, to=30, textvariable=self.shadow_blur_var, width=8).grid(row=4, column=1, sticky="w", pady=3)
        self.shadow_blur_var.trace_add("write", self._on_field_changed)

        # Tab: Latar & Kotak (Card)
        tab_bg = ttk.Frame(controls, padding=8)
        controls.add(tab_bg, text=self.t("editor_tab_background"))

        ttk.Label(tab_bg, text=self.t("editor_bg_type")).grid(row=0, column=0, sticky="w", pady=3)
        self.bg_type_var = tk.StringVar()
        ttk.Combobox(tab_bg, textvariable=self.bg_type_var, values=["transparent", "solid", "gradient"], state="readonly", width=12).grid(row=0, column=1, sticky="w", pady=3)
        self.bg_type_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_bg, text=self.t("editor_bg_color")).grid(row=1, column=0, sticky="w", pady=3)
        self.bg_color_var = tk.StringVar()
        bgc_frame = ttk.Frame(tab_bg)
        bgc_frame.grid(row=1, column=1, sticky="w", pady=3)
        ttk.Entry(bgc_frame, textvariable=self.bg_color_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Button(bgc_frame, text=self.t("editor_btn_choose"), command=lambda: self._choose_color(self.bg_color_var)).pack(side="left")
        self.bg_color_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_bg, text=self.t("editor_bg_opacity")).grid(row=2, column=0, sticky="w", pady=3)
        self.bg_opacity_var = tk.DoubleVar()
        ttk.Spinbox(tab_bg, from_=0.0, to=1.0, increment=0.05, textvariable=self.bg_opacity_var, width=8).grid(row=2, column=1, sticky="w", pady=3)
        self.bg_opacity_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_bg, text=self.t("editor_radius")).grid(row=3, column=0, sticky="w", pady=3)
        self.radius_var = tk.DoubleVar()
        ttk.Spinbox(tab_bg, from_=0, to=100, textvariable=self.radius_var, width=8).grid(row=3, column=1, sticky="w", pady=3)
        self.radius_var.trace_add("write", self._on_field_changed)

        # Tab: Posisi & Animasi
        tab_layout = ttk.Frame(controls, padding=8)
        controls.add(tab_layout, text=self.t("editor_tab_layout"))

        ttk.Label(tab_layout, text=self.t("editor_platform_preset")).grid(row=0, column=0, sticky="w", pady=3)
        self.platform_var = tk.StringVar(value="Custom")
        platform_combo = ttk.Combobox(
            tab_layout,
            textvariable=self.platform_var,
            values=["Custom", "Standard 16:9", "YouTube 1080p", "Twitch Stream", "TikTok Live 9:16"],
            state="readonly",
            width=16,
        )
        platform_combo.grid(row=0, column=1, sticky="w", pady=3)
        platform_combo.bind("<<ComboboxSelected>>", self._on_platform_selected)

        ttk.Label(tab_layout, text=self.t("editor_align")).grid(row=1, column=0, sticky="w", pady=3)
        self.align_var = tk.StringVar()
        ttk.Combobox(tab_layout, textvariable=self.align_var, values=["center", "left", "right"], state="readonly", width=10).grid(row=1, column=1, sticky="w", pady=3)
        self.align_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_layout, text=self.t("editor_max_width")).grid(row=2, column=0, sticky="w", pady=3)
        self.max_width_var = tk.DoubleVar()
        ttk.Spinbox(tab_layout, from_=20, to=100, textvariable=self.max_width_var, width=8).grid(row=2, column=1, sticky="w", pady=3)
        self.max_width_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_layout, text=self.t("editor_margin_y")).grid(row=3, column=0, sticky="w", pady=3)
        self.margin_y_var = tk.DoubleVar()
        ttk.Spinbox(tab_layout, from_=0, to=360, textvariable=self.margin_y_var, width=8).grid(row=3, column=1, sticky="w", pady=3)
        self.margin_y_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_layout, text=self.t("editor_animation")).grid(row=4, column=0, sticky="w", pady=3)
        self.anim_var = tk.StringVar()
        ttk.Combobox(tab_layout, textvariable=self.anim_var, values=["fade", "slide", "pop", "none"], state="readonly", width=10).grid(row=4, column=1, sticky="w", pady=3)
        self.anim_var.trace_add("write", self._on_field_changed)

        ttk.Label(tab_layout, text=self.t("editor_timeout")).grid(row=5, column=0, sticky="w", pady=3)
        self.timeout_var = tk.DoubleVar()
        ttk.Spinbox(tab_layout, from_=0, to=60, textvariable=self.timeout_var, width=8).grid(row=5, column=1, sticky="w", pady=3)
        self.timeout_var.trace_add("write", self._on_field_changed)

        self.label_var = tk.BooleanVar()
        ttk.Checkbutton(tab_layout, text=self.t("editor_show_label"), variable=self.label_var, command=self._on_field_changed).grid(row=6, column=0, columnspan=2, sticky="w", pady=3)

        # 3. In-App Live Preview Section
        preview_frame = ttk.LabelFrame(self, text=self.t("editor_preview_title"), padding=8)
        preview_frame.pack(fill="x", padx=8, pady=(4, 8))
        preview_frame.pack(fill="x", padx=8, pady=(4, 8))

        self.preview_canvas = tk.Canvas(
            preview_frame,
            height=85,
            bg="#0A0E1A",
            highlightthickness=1,
            highlightbackground="#1E293B",
        )
        self.preview_canvas.pack(fill="x", expand=True)
        self.preview_canvas.bind("<Configure>", lambda _e: self._render_preview())

    def _choose_color(self, var: tk.StringVar) -> None:
        initial = var.get().strip() or "#FFFFFF"
        color = colorchooser.askcolor(color=initial, parent=self)
        if color and color[1]:
            var.set(color[1].upper())

    def _on_preset_selected(self, _event=None) -> None:
        preset_name = self.preset_var.get()
        if preset_name in PRESETS:
            current_name = self.name_var.get() or f"Output {self.slot}"
            style = preset_style(preset_name, current_name)
            self._current_style = style
            self.load_from_style(style)
            self._notify_change()

    def _on_field_changed(self, *_) -> None:
        if self._updating:
            return
        self._notify_change()
        self._render_preview()

    def _render_preview(self) -> None:
        if not hasattr(self, "preview_canvas") or not self.preview_canvas.winfo_exists():
            return
        canvas = self.preview_canvas
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 10 or h <= 10:
            return

        canvas.delete("all")

        s = getattr(self, "_current_style", None) or self.to_style()

        font_family = self.font_var.get().strip() or "Segoe UI"
        try:
            raw_size = float(self.size_var.get())
        except Exception:
            raw_size = 48
        font_size = max(11, min(24, int(raw_size * 0.38)))
        weight = "bold" if int(self.weight_var.get() or 700) >= 600 else "normal"
        slant = "italic" if self.italic_var.get() else "roman"
        font_spec = (font_family, font_size, f"{weight} {slant}".strip())

        text_color = self.text_color_var.get().strip() or "#FFFFFF"
        outline_color = self.outline_color_var.get().strip() or "#000000"
        shadow_color = self.shadow_color_var.get().strip() or "#000000"
        has_outline = float(self.outline_width_var.get() or 0) > 0
        has_shadow = float(self.shadow_blur_var.get() or 0) > 0
        shadow_blur = float(self.shadow_blur_var.get() or 0)
        bg_type = self.bg_type_var.get() or "transparent"
        bg_color = self.bg_color_var.get().strip() or "#101625"
        show_label = self.label_var.get()
        align = self.align_var.get() or "center"

        border_color = getattr(s, "border_color", "#334155")
        border_width = int(float(getattr(s, "border_width", 0)))
        accent_color = getattr(s, "accent_color", "#38BDF8")
        decoration = getattr(s, "decoration", "none")

        sample_texts = {
            1: self.t("editor_sample_text_1"),
            2: self.t("editor_sample_text_2"),
            3: "皆さん、こんにちは！KizCaptionのライブへようこそ。" if self.lang == "id" else "皆さま、こんにちは！KizCaptionへようこそ。",
        }
        sample_text = sample_texts.get(self.slot, self.t("editor_sample_text_1"))

        # Exact text dimensions
        try:
            tk_font = Font(family=font_family, size=font_size, weight=weight, slant=slant)
            tw = tk_font.measure(sample_text)
            th = tk_font.metrics("linespace")
        except Exception:
            tw = max(100, int(len(sample_text) * font_size * 0.9))
            th = int(font_size * 1.5)

        pad_x = max(18, min(36, int(float(getattr(s, "padding_x", 24)) * 0.7)))
        pad_y = max(8, min(18, int(float(getattr(s, "padding_y", 10)) * 0.7)))
        card_w = min(w - 24, tw + pad_x * 2)
        card_h = th + pad_y * 2 + (14 if show_label else 0)

        if align == "left":
            card_x0 = 16
            card_x1 = card_x0 + card_w
            text_x = card_x0 + pad_x + (6 if decoration == "accent" else 0)
            anchor_tk = "w"
            label_x = text_x
            label_anchor = "w"
        elif align == "right":
            card_x1 = w - 16
            card_x0 = card_x1 - card_w
            text_x = card_x1 - pad_x
            anchor_tk = "e"
            label_x = text_x
            label_anchor = "e"
        else:
            card_x0 = (w - card_w) / 2
            card_x1 = card_x0 + card_w
            text_x = w / 2
            anchor_tk = "center"
            label_x = w / 2
            label_anchor = "center"

        card_y0 = (h - card_h) / 2
        card_y1 = card_y0 + card_h
        text_y = card_y0 + pad_y + (14 if show_label else 0) + th / 2

        # Draw card background if not transparent
        if bg_type != "transparent":
            radius = min(card_h / 2, float(self.radius_var.get() or 0) * 0.5)

            # Comic bubble / hard offset shadow
            if decoration == "bubble" or (s.shadow_blur == 0 and (s.shadow_x > 0 or s.shadow_y > 0)):
                _draw_round_rect(canvas, card_x0 + 3, card_y0 + 3, card_x1 + 3, card_y1 + 3, radius, fill="#0F172A", outline="")

            # Main card background
            _draw_round_rect(
                canvas, card_x0, card_y0, card_x1, card_y1, radius,
                fill=bg_color,
                outline=border_color if border_width > 0 else "",
                width=max(1, border_width)
            )

            # If decoration == "accent" (Lower Third / News Accent): draw vertical accent stripe on left edge
            if decoration == "accent":
                canvas.create_rectangle(card_x0, card_y0 + 2, card_x0 + 4, card_y1 - 2, fill=accent_color, outline="")

        # Draw slot badge if enabled
        if show_label:
            badge_y = card_y0 + 8
            cur_name = self.name_var.get().strip()
            label_text = cur_name.upper() if cur_name else f"OUTPUT {self.slot}"
            canvas.create_text(
                label_x, badge_y,
                text=label_text,
                font=("Segoe UI", 8, "bold"),
                fill=accent_color if decoration in ("accent", "badge") else "#38BDF8",
                anchor=label_anchor,
            )

        # Glow / Shadow effect
        if has_shadow:
            glow_rad = 2 if shadow_blur < 12 else 3
            for gx in range(-glow_rad, glow_rad + 1):
                for gy in range(-glow_rad, glow_rad + 1):
                    if gx == 0 and gy == 0:
                        continue
                    canvas.create_text(
                        text_x + gx, text_y + gy + (2 if shadow_blur < 10 else 0),
                        text=sample_text,
                        font=font_spec,
                        fill=shadow_color,
                        anchor=anchor_tk,
                    )

        # Outline stroke effect
        if has_outline:
            out_w = max(1, min(3, int(float(self.outline_width_var.get() or 0))))
            for dx in range(-out_w, out_w + 1):
                for dy in range(-out_w, out_w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    canvas.create_text(
                        text_x + dx, text_y + dy,
                        text=sample_text,
                        font=font_spec,
                        fill=outline_color,
                        anchor=anchor_tk,
                    )

        # Main text
        canvas.create_text(
            text_x, text_y,
            text=sample_text,
            font=font_spec,
            fill=text_color,
            anchor=anchor_tk,
        )

    def _notify_change(self) -> None:
        style = self.to_style()
        self._on_change(self.slot, style)

    def load_from_style(self, s: CaptionStyle) -> None:
        self._current_style = s
        self._updating = True
        try:
            self.name_var.set(s.name)
            self.preset_var.set(s.preset)
            self.font_var.set(s.font_family)
            self.size_var.set(s.font_size)
            self.weight_var.set(s.font_weight)
            self.italic_var.set(s.italic)
            self.letter_spacing_var.set(s.letter_spacing)

            self.text_color_var.set(s.text_color)
            self.outline_color_var.set(s.outline_color)
            self.outline_width_var.set(s.outline_width)
            self.shadow_color_var.set(s.shadow_color)
            self.shadow_blur_var.set(s.shadow_blur)

            self.bg_type_var.set(s.background)
            self.bg_color_var.set(s.background_color)
            self.bg_opacity_var.set(s.background_opacity)
            self.radius_var.set(s.radius)

            self.align_var.set(s.align)
            self.max_width_var.set(s.max_width)
            self.margin_y_var.set(s.margin_y)
            self.anim_var.set(s.animation)
            self.timeout_var.set(s.timeout_seconds)
            self.label_var.set(s.show_label)
        finally:
            self._updating = False
            self.after_idle(self._render_preview)

    def to_style(self) -> CaptionStyle:
        s = getattr(self, "_current_style", None) or self._get_style(self.slot)
        try:
            name = self.name_var.get().strip() or f"Output {self.slot}"
            preset = self.preset_var.get() or "Clean White"
            font_family = self.font_var.get().strip() or "Segoe UI"
            font_size = float(self.size_var.get())
            font_weight = int(self.weight_var.get() or 700)
            italic = bool(self.italic_var.get())
            letter_spacing = float(self.letter_spacing_var.get() or 0)

            text_color = self.text_color_var.get().strip() or "#FFFFFF"
            outline_color = self.outline_color_var.get().strip() or "#121620"
            outline_width = float(self.outline_width_var.get() or 0)
            shadow_color = self.shadow_color_var.get().strip() or "#000000"
            shadow_blur = float(self.shadow_blur_var.get() or 0)

            background = self.bg_type_var.get() or "transparent"
            background_color = self.bg_color_var.get().strip() or "#101625"
            background_opacity = float(self.bg_opacity_var.get() or 0.85)
            radius = float(self.radius_var.get() or 0)

            align = self.align_var.get() or "center"
            max_width = float(self.max_width_var.get() or 96)
            margin_y = float(self.margin_y_var.get() or 16)
            animation = self.anim_var.get() or "fade"
            timeout_seconds = float(self.timeout_var.get() or 8)
            show_label = bool(self.label_var.get())

            new_style = CaptionStyle(
                name=name,
                preset=preset,
                font_family=font_family,
                font_size=font_size,
                font_weight=font_weight,
                italic=italic,
                letter_spacing=letter_spacing,
                line_height=s.line_height,
                text_color=text_color,
                outline_color=outline_color,
                outline_width=outline_width,
                shadow_color=shadow_color,
                shadow_opacity=s.shadow_opacity,
                shadow_blur=shadow_blur,
                shadow_x=s.shadow_x,
                shadow_y=s.shadow_y,
                background=background,
                background_color=background_color,
                background_opacity=background_opacity,
                gradient_color=s.gradient_color,
                gradient_angle=s.gradient_angle,
                border_color=s.border_color,
                border_width=s.border_width,
                radius=radius,
                padding_x=s.padding_x,
                padding_y=s.padding_y,
                margin_x=s.margin_x,
                margin_y=margin_y,
                max_width=max_width,
                align=align,
                anchor=s.anchor,
                show_label=show_label,
                accent_color=s.accent_color,
                decoration=s.decoration,
                animation=animation,
                transition_ms=s.transition_ms,
                timeout_seconds=timeout_seconds,
            )
            new_style.validate()
            self._current_style = new_style
            return new_style
        except Exception:
            return s

    def _on_platform_selected(self, _event=None) -> None:
        p_name = self.platform_var.get()
        if p_name in PLATFORM_LAYOUTS:
            layout = PLATFORM_LAYOUTS[p_name]
            self._updating = True
            try:
                self.max_width_var.set(layout.get("max_width", 96.0))
                self.margin_y_var.set(layout.get("margin_y", 16.0))
                self.align_var.set(layout.get("align", "center"))
                if "font_size" in layout:
                    self.size_var.set(layout["font_size"])
            finally:
                self._updating = False
            self._on_field_changed()

    def _export_css(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title=f"Ekspor Gaya Output {self.slot} ke CSS",
            defaultextension=".css",
            filetypes=[("CSS Files", "*.css"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            style = self.to_style()
            css_content = export_css({self.slot: style})
            with open(path, "w", encoding="utf-8") as f:
                f.write(css_content)
            messagebox.showinfo("Ekspor Berhasil", f"Gaya Output {self.slot} diekspor ke:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Gagal Ekspor", str(exc), parent=self)

    def _import_css(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title=f"Impor Gaya Output {self.slot} dari CSS",
            filetypes=[("CSS Files", "*.css"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read()
            imported = import_css(content)
            # Find style for this slot, or slot 1 as fallback
            style = imported.get(self.slot) or next(iter(imported.values()))
            self.load_from_style(style)
            self._notify_change()
            messagebox.showinfo("Impor Berhasil", f"Gaya Output {self.slot} diperbarui dari CSS.", parent=self)
        except Exception as exc:
            messagebox.showerror("Gagal Impor", str(exc), parent=self)
