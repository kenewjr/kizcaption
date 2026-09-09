from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Language:
    name: str
    whisper: str
    nllb: str


LANGUAGES = (
    Language("Indonesian", "id", "ind_Latn"),
    Language("English", "en", "eng_Latn"),
    Language("Japanese", "ja", "jpn_Jpan"),
    Language("Korean", "ko", "kor_Hang"),
    Language("Chinese (Simplified)", "zh", "zho_Hans"),
    Language("Chinese (Traditional)", "zh", "zho_Hant"),
    Language("Arabic", "ar", "arb_Arab"),
    Language("Bengali", "bn", "ben_Beng"),
    Language("Dutch", "nl", "nld_Latn"),
    Language("French", "fr", "fra_Latn"),
    Language("German", "de", "deu_Latn"),
    Language("Hindi", "hi", "hin_Deva"),
    Language("Italian", "it", "ita_Latn"),
    Language("Javanese", "jw", "jav_Latn"),
    Language("Malay", "ms", "zsm_Latn"),
    Language("Persian", "fa", "pes_Arab"),
    Language("Polish", "pl", "pol_Latn"),
    Language("Portuguese", "pt", "por_Latn"),
    Language("Russian", "ru", "rus_Cyrl"),
    Language("Spanish", "es", "spa_Latn"),
    Language("Sundanese", "su", "sun_Latn"),
    Language("Tagalog", "tl", "tgl_Latn"),
    Language("Tamil", "ta", "tam_Taml"),
    Language("Thai", "th", "tha_Thai"),
    Language("Turkish", "tr", "tur_Latn"),
    Language("Ukrainian", "uk", "ukr_Cyrl"),
    Language("Urdu", "ur", "urd_Arab"),
    Language("Vietnamese", "vi", "vie_Latn"),
)

BY_NAME = {language.name: language for language in LANGUAGES}
BY_WHISPER: dict[str, Language] = {}
for language in LANGUAGES:
    BY_WHISPER.setdefault(language.whisper, language)

SOURCE_CHOICES = ("Auto-detect",) + tuple(language.name for language in LANGUAGES)
TARGET_CHOICES = tuple(language.name for language in LANGUAGES)


def source_whisper_code(choice: str) -> str | None:
    if choice in ("auto", "Auto-detect"):
        return None
    if choice in BY_NAME:
        return BY_NAME[choice].whisper
    if choice in BY_WHISPER:
        return choice
    raise ValueError(f"Unsupported source language: {choice}")


def source_nllb_code(configured: str, detected: str | None) -> str:
    code = detected if configured in ("auto", "Auto-detect") else source_whisper_code(configured)
    if not code or code not in BY_WHISPER:
        raise ValueError(f"NLLB cannot route detected language: {code or 'unknown'}")
    return BY_WHISPER[code].nllb


def target_nllb_code(choice: str) -> str:
    try:
        return BY_NAME[choice].nllb
    except KeyError as exc:
        raise ValueError(f"Unsupported target language: {choice}") from exc


def source_display_name(code: str) -> str:
    if code in ("auto", "Auto-detect"):
        return "Auto-detect"
    return BY_WHISPER.get(code, BY_NAME.get(code, Language(code, code, code))).name
