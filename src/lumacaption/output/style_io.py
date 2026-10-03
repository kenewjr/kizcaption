"""Restricted, round-trippable CSS for captions; never execute imported CSS."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import tempfile
from lumacaption.output.styles import CaptionStyle, NUMBERS, COLORS, BOOLEANS, color

MAX_CSS_BYTES = 65536
PX = {'font_size', 'letter_spacing', 'outline_width', 'shadow_blur', 'shadow_x', 'shadow_y', 'border_width', 'radius', 'padding_x', 'padding_y', 'margin_x', 'margin_y'}
UNITS = {**{key: 'px' for key in PX}, 'max_width': '%', 'gradient_angle': 'deg', 'transition_ms': 'ms', 'timeout_seconds': 's'}
HEADER = '/* LumaCaption CSS v1 — caption styles only */'
BLOCK = re.compile(r'#caption-([123])\s*\{((?:"(?:[^"\\]|\\.)*"|[^{}"])*?)\}', re.DOTALL)
DECLARATION = re.compile(r'\s*--lc-([a-z-]+)\s*:\s*("(?:[^"\\]|\\.)*"|[^;{}]+)\s*;', re.DOTALL)


def css_value(key, value):
    if key in ('name', 'preset', 'font_family'):
        return json.dumps(value, ensure_ascii=False)
    if key in BOOLEANS:
        return '1' if value else '0'
    if key in NUMBERS:
        return f'{value:g}{UNITS.get(key, "")}'
    return color(value) if key in COLORS else value


def export_css(profiles: dict[int, CaptionStyle]) -> str:
    if not profiles or any(type(slot) is not int or slot not in (1, 2, 3) for slot in profiles):
        raise ValueError('Pilih profil 1–3')
    lines = [HEADER]
    for slot, style in sorted(profiles.items()):
        style.validate()
        lines.append(f'#caption-{slot} {{')
        lines.extend(f'  --lc-{key.replace("_", "-")}: {css_value(key, value)};' for key, value in asdict(style).items())
        lines.append('}')
    return '\n'.join(lines) + '\n'


def import_css(text: str) -> dict[int, CaptionStyle]:
    if len(text.encode('utf-8')) > MAX_CSS_BYTES:
        raise ValueError('CSS melebihi 64 KiB')
    text = text.lstrip('\ufeff').strip()
    if not text.startswith(HEADER):
        raise ValueError('Header LumaCaption CSS v1 diperlukan; versi lain tidak didukung')
    # Keep comment-looking text inside quoted profile names intact.
    text = re.sub(r'"(?:[^"\\]|\\.)*"|/\*.*?\*/',
                  lambda m: m[0] if m[0].startswith('"') else '', text, flags=re.DOTALL)
    profiles, position = {}, 0
    for block in BLOCK.finditer(text):
        if text[position:block.start()].strip():
            raise ValueError('Hanya selector #caption-1, #caption-2, #caption-3 diterima')
        position = block.end()
        slot = int(block[1])
        if slot in profiles:
            raise ValueError(f'Profil {slot} duplikat')
        values, cursor = {}, 0
        body = block[2]
        for item in DECLARATION.finditer(body):
            if body[cursor:item.start()].strip():
                raise ValueError(f'CSS profil {slot}: deklarasi tidak valid')
            cursor = item.end()
            key, value = item[1].replace('-', '_'), item[2].strip()
            if key not in CaptionStyle.__dataclass_fields__ or key in values:
                raise ValueError(f'Properti tidak dikenal/duplikat: {key}')
            if key in NUMBERS:
                unit = UNITS.get(key, '')
                if not re.fullmatch(r'-?(?:\d+(?:\.\d+)?|\.\d+)' + re.escape(unit), value):
                    raise ValueError(f'{key}: angka dengan satuan {unit or "tanpa satuan"} diperlukan')
                value = float(value[:-len(unit)] if unit else value)
            elif key in BOOLEANS:
                if value not in ('0', '1'):
                    raise ValueError(f'{key}: harus 0 atau 1')
                value = value == '1'
            elif key in ('name', 'preset', 'font_family'):
                value = json.loads(value)
            values[key] = value
        if body[cursor:].strip() or not values:
            raise ValueError(f'CSS profil {slot} kosong/tidak valid')
        profiles[slot] = CaptionStyle.from_dict(values)
    if text[position:].strip() or not profiles:
        raise ValueError('CSS caption tidak valid; gunakan format ekspor LumaCaption')
    return profiles


def read_css(path: Path):
    with path.open('rb') as handle:
        data = handle.read(MAX_CSS_BYTES + 1)
    if len(data) > MAX_CSS_BYTES:
        raise ValueError('CSS melebihi 64 KiB')
    return import_css(data.decode('utf-8-sig'))


def write_css(path: Path, profiles):
    text = export_css(profiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
