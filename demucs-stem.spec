# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for demucs-stem binary.

Bundles htdemucs_ft model weights (~320MB) inside the binary
so it works fully offline without downloading anything.

Audio I/O uses soundfile (libsndfile) instead of torchaudio/torchcodec
to avoid FFmpeg dynamic library hell.
"""

from pathlib import Path

block_cipher = None

import demucs
demucs_pkg = Path(demucs.__file__).parent

# Find _soundfile_data (contains bundled libsndfile)
import _soundfile_data
sndfile_data = Path(_soundfile_data.__file__).parent

a = Analysis(
    ['separate.py'],
    pathex=[],
    binaries=[
        # lameenc is a C extension (.so) needed by demucs.audio for MP3
        (str(next(Path(demucs_pkg).parent.glob('lameenc*.so'))), '.'),
    ],
    datas=[
        # Embed model weights + YAML config
        ('models', 'models'),
        # Embed demucs remote configs (needed for model loading)
        (str(demucs_pkg / 'remote'), 'demucs/remote'),
        # Embed libsndfile shared library
        (str(sndfile_data), '_soundfile_data'),
    ],
    hiddenimports=[
        'demucs',
        'demucs.apply',
        'demucs.audio',
        'demucs.hdemucs',
        'demucs.htdemucs',
        'demucs.pretrained',
        'demucs.states',
        'demucs.spec',
        'demucs.transformer',
        'demucs.demucs',
        'demucs.utils',
        'demucs.repo',
        'torch',
        'soundfile',
        '_soundfile_data',
        'yaml',
        'tqdm',
        'dora',
        'diffq',
        'julius',
        'openunmix',
        'numpy',
        'numpy.core',
        'numpy.core.multiarray',
        'torchaudio',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rthook_torch.py'],
    excludes=[
        # Exclude torchcodec (needs FFmpeg .dylib) — we use soundfile instead
        'torchcodec',
        # Exclude heavy unused modules to reduce binary size
        'matplotlib',
        'PIL',
        'tkinter',
        'notebook',
        'jupyter',
        'pytest',
        'setuptools',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode={
        'demucs': 'py+pyz',
        'torch': 'py+pyz',
    },
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='demucs-stem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can corrupt torch binaries
    console=True,
    icon=None,
)
