from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from lumacaption.output.styles import ANCHORS, CaptionStyle, color, number, preset_style

SCHEMA_VERSION = 2


class FutureConfigError(ValueError):
    """Never reset configuration written by a newer app."""


def backup_file(path: Path) -> Path:
    candidate = path.with_name(path.name + ".bak")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{index}.bak")
        index += 1
    with candidate.open("xb") as out, path.open("rb") as source:
        shutil.copyfileobj(source, out)
    return candidate


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


MODEL_SIZES = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3-turbo",
    "distil-large-v3",
    "large-v3",
    "whisper-small-id",
    "whisper-medium-id",
)
NLLB_MODELS = (
    "nllb",
    "nllb-1.3b",
    "mijuanlo/nllb-200-distilled-600M-ct2-int8",
    "mijuanlo/nllb-200-distilled-1.3B-int8-ct2",
)
COMPUTE_TYPES = ("auto", "int8_float16", "int8_float32", "float16", "int8", "float32")
DEVICES = ("auto", "cuda", "cpu")
CAPTION_FONTS = (
    "Segoe UI",
    "Arial",
    "Verdana",
    "Tahoma",
    "Trebuchet MS",
    "Georgia",
    "Consolas",
)
CAPTION_FONT_MIN_SIZE = 16
CAPTION_FONT_MAX_SIZE = 96
CAPTION_THEMES = ("vtuber", "card")
CAPTION_TEXT_COLORS = (
    "#FFFFFF",
    "#FFEE55",
    "#4DD8E7",
    "#FF718D",
    "#58D6A8",
)
CAPTION_OUTLINE_COLORS = (
    "#806CFF",
    "#A855F7",
    "#000000",
    "#4DD8E7",
    "#FF718D",
    "#F0BA66",
    "none",
)


@dataclass(slots=True)
class TargetConfig:
    language: str
    profile: int = 1
    enabled: bool = True


@dataclass(slots=True)
class OverlayConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    font_family: str = "Segoe UI"
    font_size: int = 48
    text_color: str = "#FFFFFF"
    outline_color: str = "#806CFF"
    theme: str = "vtuber"
    profiles: list[CaptionStyle] = field(default_factory=lambda: [CaptionStyle(name=f"Output {i}") for i in range(1, 4)])
    anchor: str = "bottom-center"
    gap: float = 8
    order: list[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass(slots=True)
class AppConfig:
    # Names survive PortAudio index changes; int remains accepted for old configs.
    microphone_device: str | int | None = None
    source_language: str = "id"
    targets: list[TargetConfig] = field(default_factory=lambda: [TargetConfig("English")])
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    whisper_model: str = "small"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    mt_device: str = "cpu"
    mt_compute_type: str = "auto"
    nllb_model: str = "mijuanlo/nllb-200-distilled-600M-ct2-int8"
    whisper_beam_size: int = 3
    whisper_hotwords: str = ""
    vad_threshold: float = 0.5
    min_silence_ms: int = 650
    min_speech_ms: int = 150
    max_utterance_seconds: int = 20
    caption_timeout_seconds: int = 8
    schema_version: int = SCHEMA_VERSION
    audio_channel: str = "mix"
    audio_gain_db: float = 0
    normalize_audio: bool = True
    audio_clarity: bool = True
    cpu_threads: int = 4
    mt_beam_size: int = 1
    resource_preset: str = "Custom"
    vocabulary_packages: list[str] = field(default_factory=lambda: ["names", "brands"])
    regional_assistance: bool = False
    ui_theme: str = "dark"
    slang_normalization: bool = True
    denoise_engine: str = "clarity"
    ui_mode: str = "ez"
    ui_language: str = "id"
    profanity_filter: bool = False

    def __post_init__(self) -> None:
        # Preserve explicit slot IDs; optional output 2 must not renumber output 3.
        for index, target in enumerate(self.targets, 1):
            if isinstance(target, TargetConfig) and target.profile == 0:
                target.profile = index

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        """Load current config and silently migrate pre-overlay-only settings."""
        if not isinstance(raw, dict):
            raise TypeError("Config root must be an object")

        version = raw.get("schema_version", 1)
        if type(version) is not int or version < 1:
            raise ValueError("Invalid config schema version")
        if version > SCHEMA_VERSION:
            raise FutureConfigError("Konfigurasi berasal dari aplikasi lebih baru; file tidak diubah")
        defaults = cls()
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in raw.items() if key in allowed}

        raw_targets = values.get("targets", defaults.targets)
        if not isinstance(raw_targets, list):
            raise TypeError("Targets must be a list")
        targets: list[TargetConfig] = []
        for item in raw_targets:
            if isinstance(item, TargetConfig):
                targets.append(item)
            elif isinstance(item, dict):
                # Legacy obs_source is intentionally ignored.
                targets.append(TargetConfig(str(item.get("language", "")).strip(), item.get("profile", len(targets) + 1)))
            elif isinstance(item, str):
                targets.append(TargetConfig(item.strip()))
            else:
                raise TypeError("Each target must be an object or language name")
        values["targets"] = targets

        raw_overlay = values.get("overlay", asdict(defaults.overlay))
        if isinstance(raw_overlay, OverlayConfig):
            overlay = raw_overlay
        elif isinstance(raw_overlay, dict):
            overlay = OverlayConfig(
                host=str(raw_overlay.get("host", defaults.overlay.host)),
                port=raw_overlay.get("port", defaults.overlay.port),
                font_family=raw_overlay.get("font_family", defaults.overlay.font_family),
                font_size=raw_overlay.get("font_size", defaults.overlay.font_size),
                text_color=str(raw_overlay.get("text_color", defaults.overlay.text_color)),
                outline_color=str(raw_overlay.get("outline_color", defaults.overlay.outline_color)),
                theme=str(raw_overlay.get("theme", defaults.overlay.theme)),
            )
            if "profiles" in raw_overlay:
                if not isinstance(raw_overlay["profiles"], list) or len(raw_overlay["profiles"]) != 3:
                    raise ValueError("Tepat tiga profil caption diperlukan")
                overlay.profiles = [CaptionStyle.from_dict(item) for item in raw_overlay["profiles"]]
            elif version < 2:
                for index in range(3):
                    style = preset_style("Midnight Card" if overlay.theme == "card" else "Lavender Glow", f"Output {index + 1}")
                    style.font_family, style.font_size = overlay.font_family, overlay.font_size
                    style.text_color, style.outline_color = overlay.text_color, overlay.outline_color
                    style.timeout_seconds = raw.get("caption_timeout_seconds", 8)
                    overlay.profiles[index] = style
            overlay.anchor = raw_overlay.get("anchor", "bottom-center")
            overlay.gap = raw_overlay.get("gap", 8)
            overlay.order = raw_overlay.get("order", [1, 2, 3])
        else:
            raise TypeError("Overlay config must be an object")
        values["overlay"] = overlay
        values["schema_version"] = SCHEMA_VERSION
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        from lumacaption.languages import BY_NAME, source_whisper_code

        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise FutureConfigError("Unsupported config schema")
        if not isinstance(self.overlay, OverlayConfig):
            raise ValueError("Overlay config must be an object")
        if not isinstance(self.targets, list) or any(not isinstance(t, TargetConfig) or not isinstance(t.language, str) for t in self.targets):
            raise ValueError("Targets must be a list of languages")
        if not isinstance(self.overlay.profiles, list) or len(self.overlay.profiles) != 3:
            raise ValueError("Tepat tiga profil caption diperlukan")
        for profile in self.overlay.profiles:
            if not isinstance(profile, CaptionStyle):
                raise ValueError("Profil caption tidak valid")
            profile.validate()
        if (self.overlay.anchor not in ANCHORS or not isinstance(self.overlay.order, list)
                or any(type(i) is not int for i in self.overlay.order) or sorted(self.overlay.order) != [1, 2, 3]):
            raise ValueError("Layout gabungan tidak valid")
        number(self.overlay.gap, 0, 80, "Jarak caption")
        ids = [t.profile for t in self.targets]
        if any(type(i) is not int or i not in (1, 2, 3) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("ID profil output harus unik 1–3")
        if self.audio_channel not in ("mix", "left", "right"):
            raise ValueError("Channel harus mix/left/right")
        number(self.audio_gain_db, -24, 18, "Gain mic")
        if type(self.normalize_audio) is not bool or type(self.audio_clarity) is not bool or type(self.regional_assistance) is not bool:
            raise ValueError("Pilihan audio/kamus harus boolean")
        if type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 16:
            raise ValueError("Thread CPU harus 1–16")
        if type(self.mt_beam_size) is not int or self.mt_beam_size not in (1, 3):
            raise ValueError("Beam MT harus 1 atau 3")
        valid_presets = (
            "Ultra Hemat", "Hemat", "Seimbang", "Akurasi", "Ultra Studio", "Custom",
            "ultra_low", "low", "medium", "high", "ultra", "custom", "potato", "extreme", "studio"
        )
        if self.resource_preset not in valid_presets:
            raise ValueError("Preset resource tidak valid")
        if self.ui_theme not in ("dark", "light"):
            raise ValueError("ui_theme harus 'dark' atau 'light'")
        if self.ui_mode not in ("ez", "advanced"):
            raise ValueError("ui_mode harus 'ez' atau 'advanced'")
        if self.ui_language not in ("id", "en"):
            raise ValueError("ui_language harus 'id' atau 'en'")
        if self.denoise_engine not in ("off", "clarity", "dtln", "hybrid"):
            raise ValueError("denoise_engine harus 'off', 'clarity', 'dtln', atau 'hybrid'")
        if type(self.profanity_filter) is not bool:
            raise ValueError("profanity_filter harus boolean")
        if not isinstance(self.vocabulary_packages, list) or any(p not in ("names", "brands", "id", "jw", "su", "betawi", "minang", "learned") for p in self.vocabulary_packages):
            raise ValueError("Paket kamus tidak valid")

        if self.whisper_model not in MODEL_SIZES:
            raise ValueError(f"Unsupported Whisper model: {self.whisper_model}")
        if self.stt_device not in DEVICES or self.mt_device not in DEVICES:
            raise ValueError("Device must be auto, cuda, or cpu")
        if self.stt_compute_type not in COMPUTE_TYPES or self.mt_compute_type not in COMPUTE_TYPES:
            raise ValueError("Compute type must be auto, int8_float16, int8_float32, float16, int8, or float32")
        if type(self.whisper_beam_size) is not int or self.whisper_beam_size not in (1, 3, 5):
            raise ValueError("Whisper beam size must be 1, 3, or 5")
        if not isinstance(self.whisper_hotwords, str) or len(self.whisper_hotwords) > 300:
            raise ValueError("Whisper hints must be text of at most 300 characters")
        if any(ord(char) < 32 for char in self.whisper_hotwords):
            raise ValueError("Whisper hints must be a single line without control characters")
        if not 1 <= len(self.targets) <= 3:
            raise ValueError("Configure one to three target languages")
        languages = [target.language.strip() for target in self.targets]
        if any(not language for language in languages):
            raise ValueError("Target language cannot be blank")
        if len(languages) != len(set(languages)):
            raise ValueError("Target languages must be distinct")
        unknown = [language for language in languages if language not in BY_NAME]
        if unknown:
            raise ValueError(f"Unsupported target language: {unknown[0]}")
        if not isinstance(self.source_language, str):
            raise ValueError("Source language must be text")
        source_whisper_code(self.source_language)
        if self.microphone_device is not None and type(self.microphone_device) not in (str, int):
            raise ValueError("Invalid microphone device")
        if not isinstance(self.nllb_model, str) or not self.nllb_model.strip():
            raise ValueError("Invalid NLLB model path or repository")
        if self.nllb_model not in NLLB_MODELS and not Path(self.nllb_model).exists():
            raise ValueError(f"Unknown NLLB model: {self.nllb_model}")
        if self.overlay.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Overlay host must stay local")
        if type(self.overlay.port) is not int or not 1 <= self.overlay.port <= 65535:
            raise ValueError(f"Invalid overlay port: {self.overlay.port}")
        # Legacy URL overrides remain separate from modern per-slot profiles.
        if self.overlay.font_family not in CAPTION_FONTS:
            raise ValueError(f"Unsupported legacy caption font: {self.overlay.font_family}")
        number(self.overlay.font_size, CAPTION_FONT_MIN_SIZE, CAPTION_FONT_MAX_SIZE, "Legacy font size")
        if self.overlay.theme not in CAPTION_THEMES:
            raise ValueError(f"Unsupported caption theme: {self.overlay.theme}")
        color(self.overlay.text_color)
        color(self.overlay.outline_color)
        number(self.vad_threshold, 0.05, 0.95, "VAD threshold")
        for value, low, high, label in (
            (self.min_silence_ms, 100, 3000, "Silence duration"),
            (self.min_speech_ms, 100, 3000, "Speech duration"),
            (self.max_utterance_seconds, 3, 60, "Maximum utterance"),
            (self.caption_timeout_seconds, 0, 60, "Caption timeout"),
        ):
            number(value, low, high, label)


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> tuple[AppConfig, str | None]:
        if not self.path.exists():
            return AppConfig(), None
        # Read failures must not reset or overwrite user data.
        if self.path.stat().st_size > 1_048_576:
            raise ValueError("Config melebihi 1 MiB; file tidak diubah")
        raw_text = self.path.read_text(encoding="utf-8-sig")
        try:
            raw = json.loads(raw_text)
            config = AppConfig.from_dict(raw)
        except FutureConfigError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            backup = backup_file(self.path)
            config = AppConfig()
            self.save(config)
            return config, f"Corrupt config; backup {backup.name}: {exc}"
        if raw.get("schema_version", 1) < SCHEMA_VERSION:
            backup = backup_file(self.path)
            self.save(config)
            return config, f"Konfigurasi diperbarui; backup {backup.name}"
        return config, None

    def save(self, config: AppConfig) -> None:
        config.validate()
        atomic_json(self.path, asdict(config))
