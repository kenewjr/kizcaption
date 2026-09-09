from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3-turbo")
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


@dataclass(slots=True)
class OverlayConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    font_family: str = "Segoe UI"
    font_size: int = 48
    text_color: str = "#FFFFFF"
    outline_color: str = "#806CFF"
    theme: str = "vtuber"


@dataclass(slots=True)
class AppConfig:
    # Names survive PortAudio index changes; int remains accepted for old configs.
    microphone_device: str | int | None = None
    source_language: str = "id"
    targets: list[TargetConfig] = field(default_factory=lambda: [TargetConfig("English")])
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    whisper_model: str = "small"
    stt_device: str = "auto"
    mt_device: str = "cpu"
    nllb_model: str = "mijuanlo/nllb-200-distilled-600M-ct2-int8"
    whisper_beam_size: int = 3
    whisper_hotwords: str = ""
    vad_threshold: float = 0.5
    min_silence_ms: int = 650
    min_speech_ms: int = 250
    max_utterance_seconds: int = 20
    caption_timeout_seconds: int = 8

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        """Load current config and silently migrate pre-overlay-only settings."""
        if not isinstance(raw, dict):
            raise TypeError("Config root must be an object")

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
                targets.append(TargetConfig(str(item.get("language", "")).strip()))
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
                port=int(raw_overlay.get("port", defaults.overlay.port)),
                font_family=str(raw_overlay.get("font_family", defaults.overlay.font_family)),
                font_size=int(raw_overlay.get("font_size", defaults.overlay.font_size)),
                text_color=str(raw_overlay.get("text_color", defaults.overlay.text_color)),
                outline_color=str(raw_overlay.get("outline_color", defaults.overlay.outline_color)),
                theme=str(raw_overlay.get("theme", defaults.overlay.theme)),
            )
        else:
            raise TypeError("Overlay config must be an object")
        values["overlay"] = overlay

        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        from languages import BY_NAME, source_whisper_code

        if self.whisper_model not in MODEL_SIZES:
            raise ValueError(f"Unsupported Whisper model: {self.whisper_model}")
        if self.stt_device not in DEVICES or self.mt_device not in DEVICES:
            raise ValueError("Device must be auto, cuda, or cpu")
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
        source_whisper_code(self.source_language)
        if self.overlay.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Overlay host must stay local")
        if not 1 <= int(self.overlay.port) <= 65535:
            raise ValueError(f"Invalid overlay port: {self.overlay.port}")
        if self.overlay.font_family not in CAPTION_FONTS:
            raise ValueError(f"Unsupported caption font: {self.overlay.font_family}")
        if not CAPTION_FONT_MIN_SIZE <= int(self.overlay.font_size) <= CAPTION_FONT_MAX_SIZE:
            raise ValueError(
                f"Caption font size must be between {CAPTION_FONT_MIN_SIZE} and "
                f"{CAPTION_FONT_MAX_SIZE}"
            )
        if self.overlay.theme not in CAPTION_THEMES:
            raise ValueError(f"Unsupported caption theme: {self.overlay.theme}")
        if not re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$|^[a-zA-Z]+$", self.overlay.text_color):
            raise ValueError(f"Invalid text color: {self.overlay.text_color}")
        if not re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$|^[a-zA-Z]+$", self.overlay.outline_color):
            raise ValueError(f"Invalid outline color: {self.overlay.outline_color}")
        if not 0.05 <= self.vad_threshold <= 0.95:
            raise ValueError("VAD threshold must be between 0.05 and 0.95")
        if not 100 <= self.min_silence_ms <= 3000:
            raise ValueError("Silence duration must be between 100 and 3000 ms")
        if not 100 <= self.min_speech_ms <= 3000:
            raise ValueError("Speech duration must be between 100 and 3000 ms")
        if not 3 <= self.max_utterance_seconds <= 60:
            raise ValueError("Maximum utterance must be between 3 and 60 seconds")
        if not 0 <= self.caption_timeout_seconds <= 60:
            raise ValueError("Caption timeout must be between 0 and 60 seconds")


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> tuple[AppConfig, str | None]:
        if not self.path.exists():
            return AppConfig(), None
        try:
            with self.path.open("r", encoding="utf-8-sig") as handle:
                return AppConfig.from_dict(json.load(handle)), None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            corrupt_path = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                os.replace(self.path, corrupt_path)
            except OSError:
                pass
            return AppConfig(), f"Corrupt config reset: {exc}"

    def save(self, config: AppConfig) -> None:
        config.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
