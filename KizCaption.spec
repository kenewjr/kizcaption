# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

raw_datas = [
    ('src/lumacaption/output/overlay.html', 'output'),
    ('src/lumacaption/output/overlay.html', 'lumacaption/output'),
    ('src/lumacaption/assets/silero_vad.onnx', 'models'),
    ('src/lumacaption/assets/silero_vad.onnx', 'lumacaption/assets'),
    ('src/lumacaption/assets/silero_vad.onnx', 'assets'),
    ('src/lumacaption/assets/vocabulary.json', '.'),
    ('src/lumacaption/assets/vocabulary.json', 'assets'),
    ('src/lumacaption/assets/vocabulary.json', 'lumacaption/assets'),
    ('config.example.json', '.'),
    ('src/lumacaption/assets/kzp_logo.png', 'lumacaption/assets'),
    ('src/lumacaption/assets/kzp_icon.png', 'lumacaption/assets'),
    ('src/lumacaption/assets/kzp_icon.ico', 'lumacaption/assets'),
    ('src/lumacaption/assets/resources.json', 'lumacaption/assets'),
]

for src, dst in [
    ('output/overlay.html', 'output'),
    ('models/silero_vad.onnx', 'models'),
    ('vocabulary.json', '.'),
    ('config.json', '.'),
]:
    if Path(src).is_file():
        raw_datas.append((src, dst))

datas = [(s, d) for s, d in raw_datas if Path(s).is_file()]
binaries = []
hiddenimports = [
    'websockets.asyncio.server',
    'websockets.asyncio.client',
    'sounddevice',
    'ctranslate2',
    'faster_whisper',
    'onnxruntime',
    'lumacaption.vocabulary',
    'lumacaption.model_manager',
    'lumacaption.output.style_io',
    'lumacaption.audio.denoiser',
    'lumacaption.censor',
    'lumacaption.updater',
]

for package_name in (
    'ctranslate2',
    'faster_whisper',
    'onnxruntime',
    'sounddevice',
    'websockets',
    'tokenizers',
    'huggingface_hub',
):
    try:
        p_datas, p_binaries, p_hidden = collect_all(package_name)
        datas += p_datas
        binaries += p_binaries
        hiddenimports += p_hidden
    except Exception:
        pass

for nvidia_pkg in ('nvidia.cublas', 'nvidia.cudnn', 'nvidia.cuda_nvrtc'):
    try:
        import importlib
        importlib.import_module(nvidia_pkg)
        p_datas, p_binaries, p_hidden = collect_all(nvidia_pkg)
        datas += p_datas
        binaries += p_binaries
        hiddenimports += p_hidden
    except Exception:
        pass

a = Analysis(
    ['src/lumacaption/main.py'],
    pathex=['.', 'src', 'src/lumacaption'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KizCaption',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/lumacaption/assets/kzp_icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='KizCaption',
)
