from __future__ import annotations

import re


# Multi-word phrase rules (processed before single-word rules)
PHRASE_SLANG_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blagi\s+otw\b", re.IGNORECASE), "sedang di jalan"),
    (re.compile(r"\botw\b", re.IGNORECASE), "sedang di jalan"),
    (re.compile(r"\blagi\s+ngapain\b", re.IGNORECASE), "sedang apa"),
    (re.compile(r"\bmantap\s+jiwa\b", re.IGNORECASE), "luar biasa"),
    (re.compile(r"\bkeren\s+abis\b", re.IGNORECASE), "sangat keren"),
    (re.compile(r"\bbikin\s+pusing\s+aja\b", re.IGNORECASE), "sangat membingungkan"),
    (re.compile(r"\bbikin\s+pusing\b", re.IGNORECASE), "membingungkan"),
    (re.compile(r"\bgak\s+kelar-kelar|ga\s+kelar-kelar\b", re.IGNORECASE), "tidak pernah selesai"),
    (re.compile(r"\bkelar-kelar\b", re.IGNORECASE), "pernah selesai"),
    (re.compile(r"\bjangan\s+baper\b", re.IGNORECASE), "jangan tersinggung"),
    (re.compile(r"\bmager\s+banget\s+(?:gue|gw|gua|gwa)\b", re.IGNORECASE), "saya sangat malas"),
    (re.compile(r"\bmager\s+banget\b", re.IGNORECASE), "sangat malas"),
    (re.compile(r"\bkepo\s+banget(?:\s+sih)?\s+(?:lu|lo|elu)\s+jadi\s+orang\b", re.IGNORECASE), "kamu terlalu ingin tahu"),
    (re.compile(r"\bkepo\s+banget(?:\s+sih)?\s+(?:lu|lo|elu)\b", re.IGNORECASE), "kamu terlalu ingin tahu"),
    (re.compile(r"\bkepo\s+banget\b", re.IGNORECASE), "terlalu ingin tahu"),
    (re.compile(r"\bkocak\s+banget\b", re.IGNORECASE), "sangat lucu"),
    (re.compile(r"\bbikin\s+ngakak\b", re.IGNORECASE), "membuat tertawa"),
    (re.compile(r"\bgaskeun\s+(?:bro|cuy|bray|kawan)\b", re.IGNORECASE), "ayo kawan,"),
    (re.compile(r"\bgaskeun\b", re.IGNORECASE), "ayo mulai"),
    (re.compile(r"\brertain\s+musuhnya|ratain\s+musuhnya\b", re.IGNORECASE), "kalahkan musuhnya"),
    (re.compile(r"\brertain|ratain\b", re.IGNORECASE), "kalahkan"),
    (re.compile(r"\bhoki\s+parah\b", re.IGNORECASE), "sangat beruntung"),
    (re.compile(r"\bgabut\s+parah\b", re.IGNORECASE), "sangat bosan"),
    (re.compile(r"\bsantuy\s+aja|woles\s+aja\b", re.IGNORECASE), "santai saja"),
    (re.compile(r"\bmabar\s+game\s+bareng\b", re.IGNORECASE), "bermain game bersama"),
    (re.compile(r"\bmabar\s+bareng\b", re.IGNORECASE), "bermain bersama"),
    (re.compile(r"\bmabar\b", re.IGNORECASE), "bermain bersama"),
    (re.compile(r"\bmakasih\s+banyak\b", re.IGNORECASE), "terima kasih banyak"),
    (re.compile(r"\bterima\s+kasih\s+banyak\b", re.IGNORECASE), "terima kasih banyak"),
    (re.compile(r"\s+sumpah\b", re.IGNORECASE), ", sumpah"),
]

# Single-word slang rules
WORD_SLANG_RULES: list[tuple[re.Pattern, str]] = [
    # Pronouns & addresses
    (re.compile(r"\b(?:gue|gw|gua|gwa)\b", re.IGNORECASE), "saya"),
    (re.compile(r"\b(?:lu|lo|elu|loe)\b", re.IGNORECASE), "kamu"),
    (re.compile(r"\b(?:doi|doy)\b", re.IGNORECASE), "dia"),
    (re.compile(r"\b(?:cuy|bro|bray|gan|sob)\b", re.IGNORECASE), "kawan"),
    (re.compile(r"\b(?:guys|gaes)\b", re.IGNORECASE), "teman-teman"),
    (re.compile(r"\b(?:temen-temen)\b", re.IGNORECASE), "teman-teman"),
    (re.compile(r"\b(?:temen)\b", re.IGNORECASE), "teman"),

    # Common slang verbs & actions
    (re.compile(r"\b(?:mager)\b", re.IGNORECASE), "malas bergerak"),
    (re.compile(r"\b(?:baper)\b", re.IGNORECASE), "tersinggung"),
    (re.compile(r"\b(?:kepo)\b", re.IGNORECASE), "penasaran"),
    (re.compile(r"\b(?:santuy|woles)\b", re.IGNORECASE), "santai"),
    (re.compile(r"\b(?:gaskeun|gaspol|skuy|kuy)\b", re.IGNORECASE), "ayo mulai"),
    (re.compile(r"\b(?:mabar)\b", re.IGNORECASE), "main bersama"),
    (re.compile(r"\b(?:gabut)\b", re.IGNORECASE), "bosan"),
    (re.compile(r"\b(?:kocak)\b", re.IGNORECASE), "lucu"),
    (re.compile(r"\b(?:ngakak)\b", re.IGNORECASE), "tertawa"),
    (re.compile(r"\b(?:hoki)\b", re.IGNORECASE), "beruntung"),
    (re.compile(r"\b(?:curhat)\b", re.IGNORECASE), "bercerita"),
    (re.compile(r"\b(?:nemenin|nemennin)\b", re.IGNORECASE), "menemani"),
    (re.compile(r"\b(?:ngoding)\b", re.IGNORECASE), "menulis kode"),
    (re.compile(r"\b(?:omongin)\b", re.IGNORECASE), "bicarakan"),
    (re.compile(r"\b(?:ngomong)\b", re.IGNORECASE), "berbicara"),
    (re.compile(r"\b(?:ngerti)\b", re.IGNORECASE), "mengerti"),
    (re.compile(r"\b(?:nyoba)\b", re.IGNORECASE), "mencoba"),
    (re.compile(r"\b(?:nyari)\b", re.IGNORECASE), "mencari"),
    (re.compile(r"\b(?:liat|ngeliat)\b", re.IGNORECASE), "melihat"),
    (re.compile(r"\b(?:dengerin|denger)\b", re.IGNORECASE), "mendengarkan"),
    (re.compile(r"\b(?:pake)\b", re.IGNORECASE), "memakai"),
    (re.compile(r"\b(?:bikin)\b", re.IGNORECASE), "membuat"),
    (re.compile(r"\b(?:dapet)\b", re.IGNORECASE), "dapat"),
    (re.compile(r"\b(?:sampe)\b", re.IGNORECASE), "sampai"),
    (re.compile(r"\b(?:kelar)\b", re.IGNORECASE), "selesai"),
    (re.compile(r"\b(?:ngelag|lag)\b", re.IGNORECASE), "tersendat"),

    # Adverbs, degree & intensifiers
    (re.compile(r"\b(?:bgt|buanget)\b", re.IGNORECASE), "sangat"),
    (re.compile(r"\b(?:banget)\b", re.IGNORECASE), "sekali"),
    (re.compile(r"\b(?:parah)\b", re.IGNORECASE), "sekali"),
    (re.compile(r"\b(?:beneran)\b", re.IGNORECASE), "sungguh"),
    (re.compile(r"\b(?:bener)\b", re.IGNORECASE), "benar"),
    (re.compile(r"\b(?:emang|emg)\b", re.IGNORECASE), "memang"),
    (re.compile(r"\b(?:cuman|cm|cma)\b", re.IGNORECASE), "hanya"),

    # Negation, time & interrogatives
    (re.compile(r"\b(?:gak|ga|gk|nggak|ngga|kagak|ndak)\b", re.IGNORECASE), "tidak"),
    (re.compile(r"\b(?:udah|uda|udh)\b", re.IGNORECASE), "sudah"),
    (re.compile(r"\b(?:belom|blm|blom)\b", re.IGNORECASE), "belum"),
    (re.compile(r"\b(?:lagi|lg)\b", re.IGNORECASE), "sedang"),
    (re.compile(r"\b(?:kalo|kl)\b", re.IGNORECASE), "kalau"),
    (re.compile(r"\b(?:abis|abs)\b", re.IGNORECASE), "habis"),
    (re.compile(r"\b(?:terus|trs)\b", re.IGNORECASE), "lalu"),
    (re.compile(r"\b(?:gimana|gmn)\b", re.IGNORECASE), "bagaimana"),
    (re.compile(r"\b(?:kenape|knp)\b", re.IGNORECASE), "kenapa"),
    (re.compile(r"\b(?:ngapain|ngapa)\b", re.IGNORECASE), "sedang apa"),
    (re.compile(r"\b(?:ntar)\b", re.IGNORECASE), "nanti"),
    (re.compile(r"\b(?:malem)\b", re.IGNORECASE), "malam"),
    (re.compile(r"\b(?:makasih|makasi|trims|tq|thx)\b", re.IGNORECASE), "terima kasih"),
    (re.compile(r"\b(?:gapapa|gpp)\b", re.IGNORECASE), "tidak apa-apa"),
    (re.compile(r"\b(?:yodah|yaudah)\b", re.IGNORECASE), "baiklah"),
    (re.compile(r"\b(?:anjir|anjay)\b", re.IGNORECASE), "wah"),
    (re.compile(r"\b(?:mantap|mantep)\b", re.IGNORECASE), "bagus sekali"),
    (re.compile(r"\b(?:aja|aj)\b", re.IGNORECASE), "saja"),
    (re.compile(r"\b(?:yuk|yuks)\b", re.IGNORECASE), "ayo"),
    (re.compile(r"\b(?:capek|cape)\b", re.IGNORECASE), "lelah"),
    (re.compile(r"\b(?:laper)\b", re.IGNORECASE), "lapar"),
    (re.compile(r"\b(?:ngantuk)\b", re.IGNORECASE), "mengantuk"),
    (re.compile(r"\b(?:kesel)\b", re.IGNORECASE), "kesal"),
]

# Combined list maintaining priority
ALL_SLANG_RULES = PHRASE_SLANG_RULES + WORD_SLANG_RULES


def normalize_indonesian_slang(text: str) -> str:
    """Transform Indonesian colloquial & streamer slang to standard Indonesian before MT."""
    if not text:
        return ""
    normalized = text
    for pattern, replacement in ALL_SLANG_RULES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def split_into_sentences(text: str) -> list[str]:
    """Split compound utterance into individual sentences by punctuation boundaries."""
    clean = text.strip()
    if not clean:
        return []
    # Match sentences ending with ?, !, or . followed by space or end of string
    raw_parts = re.split(r"(?<=[?!.])\s+", clean)
    sentences = [p.strip() for p in raw_parts if p.strip()]
    return sentences if sentences else [clean]
