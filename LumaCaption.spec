# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = [
    ('output/overlay.html', 'output'),
    ('models/silero_vad.onnx', 'models'),
    ('config.example.json', '.'),
    ('config.json', '.'),
]
binaries = []
hiddenimports = [
    'websockets.asyncio.server',
    'websockets.asyncio.client',
    'sounddevice',
    'ctranslate2',
    'faster_whisper',
    'onnxruntime',
]

for package_name in (
    'ctranslate2',
    'faster_whisper',
    'onnxruntime',
    'sounddevice',
    'websockets',
    'tokenizers',
    'huggingface_hub',
    'nvidia.cublas',
    'nvidia.cudnn',
    'nvidia.cuda_nvrtc',
):
    try:
        p_datas, p_binaries, p_hidden = collect_all(package_name)
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
    name='LumaCaption',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LumaCaption',
)
