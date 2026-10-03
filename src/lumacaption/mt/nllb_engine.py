from __future__ import annotations

from collections.abc import Callable, Sequence
import gc
from pathlib import Path
import re

from lumacaption.gpu_runtime import configure_cuda_runtime
from lumacaption.languages import target_nllb_code


CONVERSATIONAL_MAP: dict[str, dict[str, str]] = {
    "sama-sama": {
        "English": "You're welcome.",
        "Japanese": "どういたしまして。",
        "Korean": "천만에요.",
        "Chinese (Simplified)": "不客气。",
        "Chinese (Traditional)": "不客氣。",
        "Spanish": "De nada.",
        "French": "De rien.",
        "German": "Gern geschehen.",
        "Arabic": "عفواً.",
        "Russian": "Пожалуйста.",
        "Javanese": "Sami-sami.",
        "Sundanese": "Sami-sami.",
        "Italian": "Prego.",
        "Portuguese": "De nada.",
        "Dutch": "Graag gedaan.",
        "Tagalog": "Walang anuman.",
        "Vietnamese": "Không có chi.",
        "Thai": "ไม่เป็นไรครับ/ค่ะ",
        "Malay": "Sama-sama.",
        "Hindi": "आपका स्वागत है।",
        "Turkish": "Rica ederim.",
        "Ukrainian": "Будь ласка.",
    },
    "semangat": {
        "English": "Keep it up!",
        "Japanese": "がんばって！",
        "Korean": "화이팅!",
        "Chinese (Simplified)": "加油！",
        "Chinese (Traditional)": "加油！",
        "Spanish": "¡Ánimo!",
        "French": "Bon courage !",
        "German": "Viel Erfolg!",
        "Arabic": "بالتوفيق!",
        "Russian": "Удачи!",
        "Javanese": "Semangat!",
        "Sundanese": "Semangat!",
        "Italian": "Forza!",
        "Portuguese": "Força!",
        "Dutch": "Succes!",
        "Tagalog": "Laban lang!",
        "Vietnamese": "Cố lên!",
        "Thai": "สู้ๆ นะครับ/ค่ะ",
        "Malay": "Semangat!",
        "Hindi": "लगे रहो!",
        "Turkish": "Kolay gelsin!",
        "Ukrainian": "Тримайся!",
    },
    "mantap": {
        "English": "Awesome!",
        "Japanese": "素晴らしい！",
        "Korean": "대박!",
        "Chinese (Simplified)": "太棒了！",
        "Chinese (Traditional)": "太棒了！",
        "Spanish": "¡Genial!",
        "French": "Génial !",
        "German": "Großartig!",
        "Arabic": "رائع!",
        "Russian": "Отлично!",
        "Javanese": "Mantep!",
        "Sundanese": "Mantap!",
        "Italian": "Fantastico!",
        "Portuguese": "Demais!",
        "Dutch": "Geweldig!",
        "Tagalog": "Ayos!",
        "Vietnamese": "Tuyệt vời!",
        "Thai": "สุดยอดเลย",
        "Malay": "Mantap!",
        "Hindi": "बहुत बढ़िया!",
        "Turkish": "Harika!",
        "Ukrainian": "Чудово!",
    },
    "selamat datang": {
        "English": "Welcome everyone!",
        "Japanese": "皆さん、ようこそ！",
        "Korean": "여러분, 환영합니다!",
        "Chinese (Simplified)": "欢迎大家！",
        "Chinese (Traditional)": "歡迎大家！",
        "Spanish": "¡Bienvenidos a todos!",
        "French": "Bienvenue à tous !",
        "German": "Willkommen zusammen!",
        "Arabic": "أهلاً وسهلاً بالجميع!",
        "Russian": "Добро пожаловать всем!",
        "Javanese": "Sugeng rawuh sedaya!",
        "Sundanese": "Wilujeng sumping sadayana!",
        "Italian": "Benvenuti a tutti!",
        "Portuguese": "Bem-vindos a todos!",
        "Dutch": "Welkom allemaal!",
        "Tagalog": "Maligayang pagdating sa lahat!",
        "Vietnamese": "Chào mừng mọi người!",
        "Thai": "ยินดีต้อนรับทุกคนครับ/ค่ะ",
        "Malay": "Selamat datang semua!",
        "Hindi": "सभी का स्वागत है!",
        "Turkish": "Herkese hoş geldiniz!",
        "Ukrainian": "Ласкаво просимо всіх!",
    },
    "sampai jumpa": {
        "English": "See you later!",
        "Japanese": "またね！",
        "Korean": "또 만나요!",
        "Chinese (Simplified)": "再见！",
        "Chinese (Traditional)": "再見！",
        "Spanish": "¡Hasta luego!",
        "French": "À bientôt !",
        "German": "Bis bald!",
        "Arabic": "إلى اللقاء!",
        "Russian": "До скорой встречи!",
        "Javanese": "Ngantos kepanggih malih!",
        "Sundanese": "Dugi ka tepang deui!",
        "Italian": "A presto!",
        "Portuguese": "Até logo!",
        "Dutch": "Tot ziens!",
        "Tagalog": "Hanggang sa muli!",
        "Vietnamese": "Hẹn gặp lại!",
        "Thai": "แล้วเจอกันใหม่ครับ/ค่ะ",
        "Malay": "Jumpa lagi!",
        "Hindi": "फिर मिलेंगे!",
        "Turkish": "Görüşmek üzere!",
        "Ukrainian": "До зустрічі!",
    },
    "tidak apa-apa": {
        "English": "No problem.",
        "Japanese": "大丈夫ですよ。",
        "Korean": "괜찮아요.",
        "Chinese (Simplified)": "没关系。",
        "Chinese (Traditional)": "沒關係。",
        "Spanish": "No pasa nada.",
        "French": "Ce n'est rien.",
        "German": "Kein Problem.",
        "Arabic": "لا مشكلة.",
        "Russian": "Ничего страшного.",
        "Javanese": "Mboten napa-napa.",
        "Sundanese": "Teu sawios-wios.",
        "Italian": "Nessun problema.",
        "Portuguese": "Sem problemas.",
        "Dutch": "Geen probleem.",
        "Tagalog": "Walang anuman.",
        "Vietnamese": "Không sao đâu.",
        "Thai": "ไม่เป็นไรครับ/ค่ะ",
        "Malay": "Tak apa-apa.",
        "Hindi": "कोई बात नहीं।",
        "Turkish": "Sorun değil.",
        "Ukrainian": "Нічого страшного.",
    },
    "tunggu sebentar": {
        "English": "Just a moment.",
        "Japanese": "少々お待ちください。",
        "Korean": "잠시만 기다려주세요.",
        "Chinese (Simplified)": "请稍等一下。",
        "Chinese (Traditional)": "請稍等一下。",
        "Spanish": "Un momento, por favor.",
        "French": "Un instant s'il vous plaît.",
        "German": "Einen Moment bitte.",
        "Arabic": "لحظة من فضلك.",
        "Russian": "Подождите минутку.",
        "Javanese": "Sekedhap nggih.",
        "Sundanese": "Sakedap nya.",
        "Italian": "Un momento per favore.",
        "Portuguese": "Um momento, por favor.",
        "Dutch": "Een momentje alsjeblieft.",
        "Tagalog": "Sandali lang po.",
        "Vietnamese": "Xin đợi một chút.",
        "Thai": "รอสักครู่นะครับ/ค่ะ",
        "Malay": "Tunggu sekejap.",
        "Hindi": "एक पल रुकिए।",
        "Turkish": "Bir dakika lütfen.",
        "Ukrainian": "Зачекайте хвилинку.",
    },
    "hati-hati": {
        "English": "Take care!",
        "Japanese": "気をつけてね！",
        "Korean": "조심하세요!",
        "Chinese (Simplified)": "小心点！",
        "Chinese (Traditional)": "小心點！",
        "Spanish": "¡Cuídate!",
        "French": "Prenez soin de vous !",
        "German": "Pass auf dich auf!",
        "Arabic": "انتبه على نفسك!",
        "Russian": "Береги себя!",
        "Javanese": "Ngati-ati ya!",
        "Sundanese": "Kade nya!",
        "Italian": "Fai attenzione!",
        "Portuguese": "Cuidado!",
        "Dutch": "Pas goed op jezelf!",
        "Tagalog": "Ingat ka!",
        "Vietnamese": "Bảo trọng nhé!",
        "Thai": "ดูแลตัวเองด้วยนะครับ/ค่ะ",
        "Malay": "Hati-hati ya!",
        "Hindi": "अपना ध्यान रखना!",
        "Turkish": "Kendine dikkat et!",
        "Ukrainian": "Бережи себе!",
    },
}

CONVERSATIONAL_ALIASES: dict[str, str] = {
    "sama sama": "sama-sama",
    "sama sama ya": "sama-sama",
    "sama-sama ya": "sama-sama",
    "sama2": "sama-sama",
    "terima kasih kembali": "sama-sama",
    "makasih kembali": "sama-sama",
    "makasih ya": "sama-sama",
    "semangat ya": "semangat",
    "semangat yaa": "semangat",
    "semangat semuanya": "semangat",
    "ayo semangat": "semangat",
    "tetap semangat": "semangat",
    "mantap sekali": "mantap",
    "mantap banget": "mantap",
    "mantap jiwa": "mantap",
    "keren": "mantap",
    "keren banget": "mantap",
    "keren sekali": "mantap",
    "selamat datang semuanya": "selamat datang",
    "selamat datang kawan-kawan": "selamat datang",
    "selamat datang kawan": "selamat datang",
    "selamat datang teman-teman": "selamat datang",
    "sampai jumpa lagi": "sampai jumpa",
    "sampai ketemu lagi": "sampai jumpa",
    "sampai nanti": "sampai jumpa",
    "dadah": "sampai jumpa",
    "dadah semuanya": "sampai jumpa",
    "dah semuanya": "sampai jumpa",
    "bye bye": "sampai jumpa",
    "gak apa-apa": "tidak apa-apa",
    "nggak apa-apa": "tidak apa-apa",
    "gapapa": "tidak apa-apa",
    "gak papa": "tidak apa-apa",
    "sebentar ya": "tunggu sebentar",
    "sebentar": "tunggu sebentar",
    "tunggu ya": "tunggu sebentar",
    "tunggu dulu": "tunggu sebentar",
    "tunggu sebentar ya": "tunggu sebentar",
    "hati-hati ya": "hati-hati",
    "hati-hati di jalan": "hati-hati",
}

INDONESIAN_SLANG_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:gue|gw|gua|gwa)\b", re.IGNORECASE), "saya"),
    (re.compile(r"\b(?:lu|lo|elu|loe)\b", re.IGNORECASE), "kamu"),
    (re.compile(r"\b(?:doi|doy)\b", re.IGNORECASE), "dia"),
    (re.compile(r"\b(?:gak|ga|gk|nggak|ngga|kagak|ndak)\b", re.IGNORECASE), "tidak"),
    (re.compile(r"\b(?:udah|uda|udh)\b", re.IGNORECASE), "sudah"),
    (re.compile(r"\b(?:lagi|lg)\b", re.IGNORECASE), "sedang"),
    (re.compile(r"\b(?:kalo|kl)\b", re.IGNORECASE), "kalau"),
    (re.compile(r"\b(?:abis|abs)\b", re.IGNORECASE), "habis"),
    (re.compile(r"\b(?:terus|trs)\b", re.IGNORECASE), "lalu"),
    (re.compile(r"\b(?:bgt|buanget)\b", re.IGNORECASE), "sangat"),
    (re.compile(r"\b(?:emang|emg)\b", re.IGNORECASE), "memang"),
    (re.compile(r"\b(?:cuman|cm|cma)\b", re.IGNORECASE), "hanya"),
    (re.compile(r"\b(?:beneran|bener)\b", re.IGNORECASE), "sungguh"),
    (re.compile(r"\b(?:banget)\b", re.IGNORECASE), "sangat"),
    (re.compile(r"\b(?:gimana|gmn)\b", re.IGNORECASE), "bagaimana"),
    (re.compile(r"\b(?:kenape|knp)\b", re.IGNORECASE), "kenapa"),
    (re.compile(r"\b(?:ngapain|ngapa)\b", re.IGNORECASE), "sedang apa"),
    (re.compile(r"\b(?:makasih|makasi|trims|tq|thx)\b", re.IGNORECASE), "terima kasih"),
    (re.compile(r"\b(?:mager)\b", re.IGNORECASE), "malas bergerak"),
    (re.compile(r"\b(?:baper)\b", re.IGNORECASE), "bawa perasaan"),
    (re.compile(r"\b(?:kepo)\b", re.IGNORECASE), "penasaran"),
    (re.compile(r"\b(?:woles|santuy)\b", re.IGNORECASE), "santai"),
    (re.compile(r"\b(?:gaskeun)\b", re.IGNORECASE), "ayo mulai"),
    (re.compile(r"\b(?:anjir|anjay)\b", re.IGNORECASE), "wah"),
    (re.compile(r"\b(?:mantap|mantep)\b", re.IGNORECASE), "bagus sekali"),
    (re.compile(r"\b(?:cuy|bro|bray)\b", re.IGNORECASE), "kawan"),
    (re.compile(r"\b(?:gapapa|gpp)\b", re.IGNORECASE), "tidak apa-apa"),
    (re.compile(r"\b(?:yodah|yaudah)\b", re.IGNORECASE), "baiklah"),
    (re.compile(r"\b(?:bikin)\b", re.IGNORECASE), "membuat"),
    (re.compile(r"\b(?:dapet)\b", re.IGNORECASE), "dapat"),
    (re.compile(r"\b(?:ngomong)\b", re.IGNORECASE), "berbicara"),
    (re.compile(r"\b(?:liat|ngeliat)\b", re.IGNORECASE), "melihat"),
    (re.compile(r"\b(?:sampe)\b", re.IGNORECASE), "sampai"),
    (re.compile(r"\b(?:pake)\b", re.IGNORECASE), "memakai"),
    (re.compile(r"\b(?:mabar)\b", re.IGNORECASE), "main bersama"),
    (re.compile(r"\b(?:gabut)\b", re.IGNORECASE), "bosan"),
    (re.compile(r"\b(?:kocak)\b", re.IGNORECASE), "lucu"),
    (re.compile(r"\b(?:hoki)\b", re.IGNORECASE), "beruntung"),
]


def normalize_indonesian_slang(text: str) -> str:
    """Transform Indonesian colloquial & streamer slang to standard Indonesian before MT."""
    if not text:
        return ""
    normalized = text
    for pattern, replacement in INDONESIAN_SLANG_MAP:
        normalized = pattern.sub(replacement, normalized)
    return normalized


class NllbEngine:
    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        cache_dir: str | Path,
        on_warning: Callable[[str], None] | None = None,
        *,
        compute_type: str = "auto",
        beam_size: int = 2,
        cpu_threads: int = 4,
        normalize_slang: bool = True,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.normalize_slang = normalize_slang
        self.cache_dir = Path(cache_dir)
        self.on_warning = on_warning or (lambda _message: None)
        self._translator = None
        self._sentencepiece = None
        self._model_path: Path | None = None
        self.active_device = ""
        self.active_compute_type = ""
        self._cache: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}
        self._item_cache: dict[tuple[str, str, str], str] = {}

    def _resolve_model(self) -> Path:
        path = Path(self.model_id_or_path).expanduser()
        if path.exists():
            return path.resolve()
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        required = ("model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model")
        
        # Resolve revision per repo catalog
        revision = "16bc5ff0482f9f1c0d35bdef950721ce58640789"
        if "1.3b" in self.model_id_or_path.lower() or "1.3B" in self.model_id_or_path:
            revision = "6eee5eda03ff1441d2a6117d34a02e44504ce321"

        kwargs = dict(
            repo_id=self.model_id_or_path,
            revision=revision,
            cache_dir=self.cache_dir,
            allow_patterns=required,
        )
        try:
            cached = Path(snapshot_download(**kwargs, local_files_only=True))
            if all((cached / name).is_file() for name in required):
                return cached
        except LocalEntryNotFoundError:
            pass
        self.on_warning("Mengunduh NLLB; progres ada di terminal")
        cached = Path(snapshot_download(**kwargs))
        if not all((cached / name).is_file() for name in required):
            raise RuntimeError("Cache NLLB belum lengkap")
        return cached

    def _resolve_compute_type(self, device: str) -> str:
        if device == "cpu":
            return "int8"
        if self.requested_compute_type != "auto":
            return self.requested_compute_type
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda", 0)
            if "int8_float16" in supported:
                return "int8_float16"
            if "float16" in supported:
                return "float16"
            if "int8_float32" in supported:
                return "int8_float32"
            return "float32"
        except Exception:
            return "int8_float16"

    def _load(self, device: str) -> None:
        configure_cuda_runtime()
        import ctranslate2
        import sentencepiece as spm

        if self._model_path is None:
            self._model_path = self._resolve_model()
        compute_type = self._resolve_compute_type(device)
        self._translator = ctranslate2.Translator(
            str(self._model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=max(1, min(self.cpu_threads, 16)) if device == "cpu" else 1,
        )
        self._sentencepiece = spm.SentencePieceProcessor(
            model_file=str(self._model_path / "sentencepiece.bpe.model")
        )
        self.active_device = device
        self.active_compute_type = compute_type
        self.on_warning(f"NLLB active: {device} ({compute_type})")

    def _preferred_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() else "cpu"

    def _release(self) -> None:
        if self._translator is not None:
            try:
                self._translator.unload_model()
            except RuntimeError:
                pass
        self._translator = None
        self._sentencepiece = None
        self._cache.clear()
        self._item_cache.clear()
        gc.collect()

    def _lookup_cache(self, text: str, source_nllb: str, targets: Sequence[str]) -> dict[str, str] | None:
        clean = text.strip()
        if len(clean) > 160:
            return None
        t_targets = tuple(targets)
        # 1. Exact match
        k_exact = (clean, source_nllb, t_targets)
        if k_exact in self._cache:
            return dict(self._cache[k_exact])
        # 2. Punctuation-stripped match
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_stripped = (stripped, source_nllb, t_targets)
        if k_stripped in self._cache:
            return dict(self._cache[k_stripped])
        # 3. Case-insensitive fast keys
        k_lower = (stripped.lower(), source_nllb, t_targets)
        if k_lower in self._cache:
            return dict(self._cache[k_lower])
        k_title = (stripped.capitalize(), source_nllb, t_targets)
        if k_title in self._cache:
            return dict(self._cache[k_title])

        # 4. Fallback scan over bounded LRU cache (<= 128 entries)
        low = stripped.lower()
        for (ck, csrc, ctgts), val in self._cache.items():
            if csrc == source_nllb and ctgts == t_targets:
                if re.sub(r"[.!?…,:;\"'“”]+$", "", ck).strip().lower() == low:
                    return dict(val)
        return None

    def _put_cache(self, text: str, source_nllb: str, targets: Sequence[str], res: dict[str, str]) -> None:
        clean = text.strip()
        if len(clean) > 160:
            return
        t_targets = tuple(targets)
        k_exact = (clean, source_nllb, t_targets)
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_lower = (stripped.lower(), source_nllb, t_targets)

        for k in (k_exact, k_lower):
            if len(self._cache) >= 128 and k not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[k] = dict(res)

    def _lookup_target_cache(self, text: str, source_nllb: str, target_name: str) -> str | None:
        clean = text.strip()
        if len(clean) > 160:
            return None
        k_exact = (clean, source_nllb, target_name)
        if k_exact in self._item_cache:
            return self._item_cache[k_exact]
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_stripped = (stripped, source_nllb, target_name)
        if k_stripped in self._item_cache:
            return self._item_cache[k_stripped]
        k_lower = (stripped.lower(), source_nllb, target_name)
        if k_lower in self._item_cache:
            return self._item_cache[k_lower]
        return None

    def _put_target_cache(self, text: str, source_nllb: str, target_name: str, translation: str) -> None:
        clean = text.strip()
        if len(clean) > 160 or not translation:
            return
        stripped = re.sub(r"[.!?…,:;\"'“”]+$", "", clean).strip()
        k_exact = (clean, source_nllb, target_name)
        k_lower = (stripped.lower(), source_nllb, target_name)
        for k in (k_exact, k_lower):
            if len(self._item_cache) >= 512 and k not in self._item_cache:
                oldest_key = next(iter(self._item_cache))
                del self._item_cache[oldest_key]
            self._item_cache[k] = translation

    def prepare(self, targets: Sequence[str]) -> None:
        """Exercise translation kernels, but never publish the warm-up text."""
        if targets:
            self.translate("Hello.", "eng_Latn", targets[:1])

    def translate(self, text: str, source_nllb: str, targets: Sequence[str]) -> dict[str, str]:
        clean_original = text.strip()
        if not clean_original or not targets:
            return {}

        cached = self._lookup_cache(clean_original, source_nllb, targets)
        if cached is not None:
            return cached

        translations: dict[str, str] = {}
        needed_targets: list[str] = []
        for name in targets:
            try:
                code = target_nllb_code(name)
            except KeyError:
                needed_targets.append(name)
                continue
            if code == source_nllb:
                translations[name] = clean_original
            else:
                cached_item = self._lookup_target_cache(clean_original, source_nllb, name)
                if cached_item is not None:
                    translations[name] = cached_item
                else:
                    needed_targets.append(name)

        if not needed_targets:
            res = {name: translations[name] for name in targets if name in translations}
            self._put_cache(clean_original, source_nllb, targets, res)
            return res

        # Conversational stream formula fast-path for Indonesian
        if source_nllb == "ind_Latn":
            norm = re.sub(r"[^\w\s-]", "", clean_original.lower())
            norm = re.sub(r"\s+", " ", norm).strip()
            canonical = CONVERSATIONAL_ALIASES.get(norm, norm)
            if canonical in CONVERSATIONAL_MAP:
                formula = CONVERSATIONAL_MAP[canonical]
                if all(t in formula for t in needed_targets):
                    for t in needed_targets:
                        translations[t] = formula[t]
                    res = {name: translations[name] for name in targets if name in translations}
                    self._put_cache(clean_original, source_nllb, targets, res)
                    for t_name, t_val in res.items():
                        self._put_target_cache(clean_original, source_nllb, t_name, t_val)
                    return res

        if self._translator is None:
            preferred = self._preferred_device()
            try:
                self._load(preferred)
            except Exception as exc:
                if preferred == "cpu":
                    raise
                self.on_warning(f"NLLB CUDA unavailable: {exc}; using CPU")
                self._release()
                self._load("cpu")
                return self.translate(text, source_nllb, targets)

        assert self._sentencepiece is not None
        # Normalize casing and closure for optimal NLLB sentence parsing
        clean_text = text.strip()
        if source_nllb == "ind_Latn" and getattr(self, "normalize_slang", True):
            clean_text = normalize_indonesian_slang(clean_text)
        if clean_text and clean_text[0].islower():
            clean_text = clean_text[0].upper() + clean_text[1:]
        if clean_text and clean_text[-1] not in ".?!…:;":
            clean_text += "."

        pieces = self._sentencepiece.encode(clean_text, out_type=str)
        # NLLB accepts 512 tokens; reserve source language and EOS tokens.
        if len(pieces) > 510:
            self.on_warning("Ucapan melebihi batas 510 token NLLB; input dipotong")
            pieces = pieces[:510]
        source = [source_nllb, *pieces, "</s>"]
        target_codes = [target_nllb_code(name) for name in needed_targets]

        # Dynamic max tokens prevents slow autoregressive overgeneration on short utterances
        max_tokens = min(256, max(32, int(len(pieces) * 1.8) + 12))
        effective_beam = max(1, self.beam_size)
        try:
            results = self._translator.translate_batch(
                [source.copy() for _ in target_codes],
                target_prefix=[[code] for code in target_codes],
                beam_size=effective_beam,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                disable_unk=True,
                max_decoding_length=max_tokens,
                return_end_token=True,
                batch_type="tokens",
                max_batch_size=1024,
            )
        except Exception as exc:
            if self.active_device != "cuda":
                raise
            self.on_warning(f"NLLB CUDA inference failed: {exc}; retrying on CPU")
            self._release()
            self._load("cpu")
            return self.translate(text, source_nllb, targets)

        for name, code, result in zip(needed_targets, target_codes, results, strict=True):
            tokens = list(result.hypotheses[0])
            hit_limit = False
            if not tokens or tokens[-1] != "</s>":
                hit_limit = True
                self.on_warning("Terjemahan mencapai batas decoder; teks diakhiri elipsis")
            else:
                tokens.pop()
            if tokens and tokens[0] == code:
                tokens.pop(0)
            decoded = self._sentencepiece.decode(tokens).strip()

            # Clean leading dialog turn dashes if original text didn't start with a dash
            if decoded.startswith(("- ", "— ", "– ", "-")) and not text.lstrip().startswith(("-", "—", "–")):
                decoded = decoded.lstrip("-—– ").strip()

            # Clean unknown token artifacts and collapse whitespace
            decoded = decoded.replace("⁇", "").replace("\ufffd", "").strip()
            decoded = re.sub(r"\s+", " ", decoded)

            # CJK / Korean punctuation spacing cleanup
            if code in ("jpn_Jpan", "zho_Hans", "zho_Hant"):
                decoded = re.sub(r"\s*([。、！？，：])\s*", r"\1", decoded)
            elif code == "kor_Hang":
                decoded = re.sub(r"\s*([.!?])", r"\1", decoded)

            if hit_limit and decoded:
                decoded += "…"
            translations[name] = decoded
            self._put_target_cache(clean_original, source_nllb, name, decoded)

        # Cache short phrases (LRU 128 max entries)
        res = {name: translations[name] for name in targets if name in translations}
        self._put_cache(clean_original, source_nllb, targets, res)
        return res

    def close(self) -> None:
        self._release()
