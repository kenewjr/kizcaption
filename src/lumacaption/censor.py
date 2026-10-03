from __future__ import annotations

import re

# Comprehensive list of offensive, profane, and stream-TOS violating terms (ID & EN)
_PROFANITY_WORDS = (
    # Indonesian
    "anjing", "anjir", "anjay", "babi", "bangsat", "bajingan",
    "kontol", "memek", "pantek", "itil", "jembut", "pepek",
    "ngentot", "entot", "mengentot", "dientot", "asu", "jancok",
    "dancok", "jancuk", "cok", "taek", "tai", "perek", "lonte",
    "peler", "tolol", "goblok", "idiot", "bego", "kampret",
    "brengsek", "jablay", "kimak", "puki", "pukimak", "titit",
    "dianjingin", "keanjingan", "dikontolin",
    # English
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fucks", "fuckin",
    "shit", "shits", "shitty", "bullshit", "bitch", "bitches", "bitching",
    "cunt", "cunts", "asshole", "assholes", "bastard", "bastards",
    "dick", "dicks", "pussy", "pussies", "whore", "whores", "slut", "sluts",
    "motherfucker", "motherfuckers", "cock", "cocks", "nigger", "nigga",
)

_PATTERN = r"\b(?:" + "|".join(re.escape(w) for w in sorted(_PROFANITY_WORDS, key=len, reverse=True)) + r")\b"
_CENSOR_RE = re.compile(_PATTERN, re.IGNORECASE)


def censor_text(text: str, enabled: bool = True) -> str:
    """Mask vulgar and offensive words with '***' to protect stream TOS.

    Leaves benign words intact (e.g. 'class', 'pass', 'kontak', 'pantai').
    """
    if not enabled or not text:
        return text
    return _CENSOR_RE.sub("***", text)
