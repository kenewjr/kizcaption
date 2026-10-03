"""Caption design tokens. Presets are data, never separate renderers."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import math
import re
from typing import Any

FONTS = ("Segoe UI", "Arial", "Verdana", "Tahoma", "Trebuchet MS", "Georgia", "Consolas")
ANCHORS = ("top-left", "top-center", "top-right", "center-left", "center", "center-right", "bottom-left", "bottom-center", "bottom-right")
NUMBERS = {
    "font_size": (12, 192), "font_weight": (100, 900), "letter_spacing": (-2, 12),
    "line_height": (1, 2.5), "outline_width": (0, 10), "shadow_blur": (0, 30),
    "shadow_x": (-20, 20), "shadow_y": (-20, 20), "shadow_opacity": (0, 1),
    "background_opacity": (0, 1), "gradient_angle": (0, 360), "border_width": (0, 10),
    "radius": (0, 100), "padding_x": (0, 80), "padding_y": (0, 60),
    "margin_x": (0, 120), "margin_y": (0, 360), "max_width": (20, 100),
    "transition_ms": (0, 1000), "timeout_seconds": (0, 60),
}
COLORS = ("text_color", "outline_color", "shadow_color", "background_color", "gradient_color", "border_color", "accent_color")
CHOICES = {"background": ("transparent", "solid", "gradient"), "align": ("left", "center", "right"), "anchor": ANCHORS, "animation": ("none", "fade", "slide", "pop"), "decoration": ("none", "accent", "bubble", "badge")}
BOOLEANS = ("italic", "show_label")

def number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}: angka harus {low}–{high}")

def color(value):
    aliases = {"white": "#FFFFFF", "black": "#000000", "red": "#FF0000", "yellow": "#FFFF00", "cyan": "#00FFFF", "none": "transparent"}
    if not isinstance(value, str):
        raise ValueError("Warna harus teks HEX")
    value = aliases.get(value.lower(), value)
    if value == "transparent" or re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", value):
        return value
    raise ValueError(f"Warna tidak valid: {value!r}; gunakan #RRGGBB")

@dataclass(slots=True)
class CaptionStyle:
    name: str = "Output"
    preset: str = "Clean White"
    font_family: str = "Segoe UI"
    font_size: float = 48
    font_weight: int = 700
    italic: bool = False
    letter_spacing: float = 0
    line_height: float = 1.25
    text_color: str = "#FFFFFF"
    outline_color: str = "#121620"
    outline_width: float = 2
    shadow_color: str = "#000000"
    shadow_opacity: float = 0.65
    shadow_blur: float = 4
    shadow_x: float = 0
    shadow_y: float = 2
    background: str = "transparent"
    background_color: str = "#101625"
    background_opacity: float = 0.85
    gradient_color: str = "#30244C"
    gradient_angle: float = 135
    border_color: str = "#414968"
    border_width: float = 0
    radius: float = 16
    padding_x: float = 16
    padding_y: float = 8
    margin_x: float = 24
    margin_y: float = 16
    max_width: float = 96
    align: str = "center"
    anchor: str = "bottom-center"
    show_label: bool = False
    accent_color: str = "#806CFF"
    decoration: str = "none"
    animation: str = "fade"
    transition_ms: float = 180
    timeout_seconds: float = 8

    def validate(self):
        for name in ("name", "preset"):
            value = getattr(self, name)
            if not isinstance(value, str) or not 1 <= len(value) <= 80 or any(ord(c) < 32 for c in value):
                raise ValueError(f"{name}: teks 1–80 karakter diperlukan")
        if not isinstance(self.font_family, str) or not re.fullmatch(r"[\w .+\-]{1,80}", self.font_family, re.UNICODE):
            raise ValueError("Nama font tidak valid")
        for name, (low, high) in NUMBERS.items():
            number(getattr(self, name), low, high, name)
        for name in COLORS:
            color(getattr(self, name))
        for name, choices in CHOICES.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name}: pilihan tidak valid")
        for name in BOOLEANS:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name}: harus boolean")

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Properti gaya tidak dikenal")
        style = cls(**raw)
        style.validate()
        return style

    def to_dict(self):
        self.validate()
        return asdict(self)

# Composition variations; highly polished modern styling: cute kawaii to cool cyberpunk.
_PRESETS = (
    # --- MINIMAL: Bersih, Tajam, Subtitle Pro ---
    ("Minimal", "Clean White", dict(
        font_family="Segoe UI", font_size=48, font_weight=700, text_color="#FFFFFF",
        outline_color="#000000", outline_width=2.5, shadow_color="#000000", shadow_blur=6,
        shadow_y=2, shadow_opacity=0.85, radius=12
    )),
    ("Minimal", "Studio Subtitle", dict(
        font_family="Arial", font_size=46, font_weight=700, text_color="#FFFFFF",
        outline_color="#000000", outline_width=2.0, shadow_color="#000000", shadow_blur=4,
        shadow_y=2, shadow_opacity=0.90, letter_spacing=0.5
    )),
    ("Minimal", "Mono Console", dict(
        font_family="Consolas", font_weight=700, font_size=44, align="center", text_color="#39FF14",
        background="solid", background_color="#060D17", background_opacity=0.92, border_color="#10B981",
        border_width=1.5, radius=10, padding_x=28, padding_y=12, letter_spacing=1.0, outline_width=0,
        shadow_color="#10B981", shadow_blur=10
    )),
    ("Minimal", "Bold Contrast", dict(
        font_family="Arial", font_weight=900, font_size=52, text_color="#FFFFFF", outline_color="#000000",
        outline_width=4.0, shadow_color="#000000", shadow_blur=0, shadow_x=3, shadow_y=3,
        shadow_opacity=1.0, letter_spacing=0.5
    )),
    ("Minimal", "Soft Shadow", dict(
        font_family="Segoe UI", font_weight=600, font_size=48, text_color="#FFFFFF", outline_width=0,
        shadow_color="#000000", shadow_blur=16, shadow_y=4, shadow_opacity=0.90, letter_spacing=0.5
    )),

    # --- ANIME / VTUBER: Imut, Manis, Kawaii, Pastel ---
    ("Anime / VTuber", "Sakura", dict(
        text_color="#FFF5F8", outline_color="#F43F5E", outline_width=3.5, shadow_color="#FB7185",
        shadow_blur=16, shadow_opacity=0.95, font_weight=800, letter_spacing=0.5
    )),
    ("Anime / VTuber", "Lavender Glow", dict(
        text_color="#FAF5FF", outline_color="#7C3AED", outline_width=3.5, shadow_color="#C084FC",
        shadow_blur=22, shadow_opacity=0.95, font_weight=800, letter_spacing=0.5
    )),
    ("Anime / VTuber", "Candy Pop", dict(
        font_family="Trebuchet MS", text_color="#FEF08A", outline_color="#DB2777", outline_width=4.0,
        shadow_color="#F472B6", shadow_blur=12, shadow_y=3, shadow_opacity=0.95, font_size=50, font_weight=900
    )),
    ("Anime / VTuber", "Manga Stroke", dict(
        text_color="#FFFFFF", outline_color="#111827", outline_width=4.5, italic=True,
        font_weight=900, shadow_color="#1F2937", shadow_blur=0, shadow_x=3, shadow_y=3, shadow_opacity=1.0
    )),
    ("Anime / VTuber", "Pastel Mint", dict(
        text_color="#ECFEFF", outline_color="#0D9488", outline_width=3.5, shadow_color="#2DD4BF",
        shadow_blur=14, shadow_opacity=0.95, font_weight=800, letter_spacing=0.5
    )),

    # --- CARD: Modern Glassmorphism, Rounded Pills & Sleek Containers ---
    ("Card", "Midnight Card", dict(
        background="solid", background_color="#090E1A", background_opacity=0.90, border_color="#334155",
        border_width=1.5, radius=18, padding_x=32, padding_y=14, font_size=46, font_weight=700, outline_width=0,
        text_color="#F8FAFC", shadow_color="#000000", shadow_blur=16, shadow_opacity=0.70
    )),
    ("Card", "Glass Lite", dict(
        background="solid", background_color="#1E293B", background_opacity=0.55, border_color="#94A3B8",
        border_width=1.5, radius=20, padding_x=32, padding_y=14, font_size=46, font_weight=700, outline_width=0,
        text_color="#FFFFFF", shadow_color="#000000", shadow_blur=18, shadow_opacity=0.50
    )),
    ("Card", "Rounded Slate", dict(
        background="solid", background_color="#1E293B", background_opacity=0.88, radius=28,
        padding_x=32, padding_y=14, border_color="#475569", border_width=1.5, outline_width=0,
        font_size=46, font_weight=700, text_color="#F1F5F9", shadow_color="#0F172A", shadow_blur=14
    )),
    ("Card", "Paper Light", dict(
        background="solid", background_color="#F8F6F0", background_opacity=0.96, text_color="#0F172A",
        border_color="#E2E8F0", border_width=1.5, radius=16, padding_x=30, padding_y=14, font_size=46,
        font_weight=700, outline_width=0, shadow_color="#000000", shadow_blur=12, shadow_opacity=0.20
    )),
    ("Card", "Compact Pill", dict(
        background="solid", background_color="#070B14", background_opacity=0.92, radius=100,
        padding_x=32, padding_y=12, font_size=44, font_weight=700, border_color="#38BDF8",
        border_width=1.5, outline_width=0, text_color="#F8FAFC", shadow_color="#0284C7",
        shadow_blur=12, shadow_opacity=0.60
    )),

    # --- BROADCAST: Pro Esports, Studio & TV News ---
    ("Broadcast", "Lower Third", dict(
        background="solid", background_color="#060A14", background_opacity=0.94, decoration="accent",
        accent_color="#00D4FF", align="left", anchor="bottom-left", radius=8, border_color="#1E293B",
        border_width=1, show_label=True, padding_x=28, padding_y=14, font_size=46, font_weight=700,
        outline_width=0, text_color="#F8FAFC"
    )),
    ("Broadcast", "News Accent", dict(
        background="gradient", background_color="#070D1E", gradient_color="#0F172A", gradient_angle=135,
        decoration="accent", accent_color="#F59E0B", align="left", radius=8, show_label=True,
        border_color="#3B82F6", border_width=1.5, padding_x=28, padding_y=14, font_size=46,
        font_weight=700, outline_width=0, text_color="#FFFFFF"
    )),
    ("Broadcast", "Sport Strip", dict(
        background="solid", background_color="#0B0F19", background_opacity=0.95, text_color="#CCFF00",
        italic=True, font_weight=900, font_size=46, border_color="#CCFF00", border_width=2,
        radius=8, padding_x=30, padding_y=12, outline_width=0, shadow_color="#CCFF00",
        shadow_blur=10, shadow_opacity=0.50
    )),
    ("Broadcast", "Interview", dict(
        background="solid", background_color="#0D1117", background_opacity=0.90, align="left",
        font_family="Segoe UI", font_size=44, font_weight=600, show_label=True, outline_width=0,
        radius=12, border_color="#334155", border_width=1.5, padding_x=28, padding_y=14,
        text_color="#F8FAFC", accent_color="#38BDF8"
    )),
    ("Broadcast", "Documentary", dict(
        font_family="Georgia", font_weight=600, font_size=46, text_color="#FEF9C3",
        shadow_blur=12, shadow_opacity=0.90, outline_width=1.5, outline_color="#0F172A", letter_spacing=0.5
    )),

    # --- NEON: Keren, Cyberpunk, Electric & Outrun ---
    ("Neon", "Cyber Violet", dict(
        text_color="#FDF4FF", outline_color="#7E22CE", outline_width=2.5, shadow_color="#D946EF",
        shadow_blur=22, shadow_opacity=0.95, font_weight=800
    )),
    ("Neon", "Cyan Edge", dict(
        text_color="#ECFEFF", outline_color="#0891B2", outline_width=2.5, shadow_color="#06B6D4",
        shadow_blur=20, shadow_opacity=0.95, border_color="#22D3EE", border_width=1, radius=8,
        decoration="accent", accent_color="#00F0FF"
    )),
    ("Neon", "Sunset Duo", dict(
        background="gradient", background_color="#E11D48", gradient_color="#EA580C", gradient_angle=135,
        background_opacity=0.92, border_color="#FDE047", border_width=1.5, radius=20, padding_x=32,
        padding_y=14, font_size=46, font_weight=700, outline_width=0, text_color="#FFFFFF",
        shadow_color="#E11D48", shadow_blur=16, shadow_opacity=0.70
    )),
    ("Neon", "Electric Lime", dict(
        font_family="Consolas", font_weight=700, font_size=46, text_color="#F0FDF4", outline_color="#065F46",
        outline_width=2.5, shadow_color="#10B981", shadow_blur=20, shadow_opacity=0.95
    )),
    ("Neon", "Synthwave", dict(
        background="gradient", background_color="#240046", gradient_color="#5A189A", gradient_angle=135,
        background_opacity=0.92, border_color="#F72585", border_width=2, radius=16, padding_x=32,
        padding_y=14, font_size=46, font_weight=800, text_color="#00F0FF", outline_color="#7209B7",
        outline_width=1.5, shadow_color="#F72585", shadow_blur=20, shadow_opacity=0.90
    )),

    # --- CREATIVE: Pop Art, Retro Phosphor & Haute Couture ---
    ("Creative", "Retro Mono", dict(
        font_family="Consolas", font_weight=700, font_size=44, text_color="#FFB000", background="solid",
        background_color="#0A0E17", background_opacity=0.94, border_color="#D97706", border_width=1.5,
        radius=10, outline_width=0, letter_spacing=1.5, padding_x=28, padding_y=12,
        shadow_color="#F59E0B", shadow_blur=14, shadow_opacity=0.85
    )),
    ("Creative", "Comic Bubble", dict(
        background="solid", background_color="#FEF08A", text_color="#0F172A", border_color="#0F172A",
        border_width=3.0, outline_width=0, radius=24, decoration="bubble", font_weight=900,
        font_size=46, padding_x=32, padding_y=14, shadow_color="#0F172A", shadow_blur=0,
        shadow_x=4, shadow_y=4, shadow_opacity=1.0
    )),
    ("Creative", "Gradient Ribbon", dict(
        background="gradient", background_color="#0284C7", gradient_color="#7C3AED", gradient_angle=135,
        background_opacity=0.92, radius=24, padding_x=34, padding_y=14, font_size=46, font_weight=700,
        border_color="#C4B5FD", border_width=1.5, outline_width=0, text_color="#FFFFFF",
        shadow_color="#8B5CF6", shadow_blur=18, shadow_opacity=0.80
    )),
    ("Creative", "Elegant Serif", dict(
        font_family="Georgia", font_weight=600, italic=True, font_size=46, text_color="#FEF3C7",
        letter_spacing=1.0, outline_width=0, shadow_color="#000000", shadow_blur=14, shadow_opacity=0.85
    )),
    ("Creative", "Stream Badge", dict(
        background="solid", background_color="#0F132A", background_opacity=0.92, border_color="#6366F1",
        border_width=1.5, radius=18, padding_x=30, padding_y=12, font_size=44, font_weight=700,
        decoration="badge", show_label=True, accent_color="#818CF8", outline_width=0, text_color="#FFFFFF",
        shadow_color="#6366F1", shadow_blur=14, shadow_opacity=0.60
    )),
)
PRESETS = {name: (group, CaptionStyle(preset=name, **values)) for group, name, values in _PRESETS}

def preset_style(name, profile_name="Output"):
    try:
        return replace(PRESETS[name][1], name=profile_name)
    except KeyError as exc:
        raise ValueError(f"Preset tidak dikenal: {name}") from exc


PLATFORM_LAYOUTS: dict[str, dict[str, Any]] = {
    "Standard 16:9": {
        "max_width": 96.0,
        "margin_y": 16.0,
        "align": "center",
    },
    "YouTube 1080p": {
        "max_width": 88.0,
        "margin_y": 35.0,
        "align": "center",
    },
    "Twitch Stream": {
        "max_width": 82.0,
        "margin_y": 45.0,
        "align": "center",
    },
    "TikTok Live 9:16": {
        "max_width": 72.0,
        "margin_y": 200.0,
        "align": "center",
        "font_size": 44.0,
    },
}

