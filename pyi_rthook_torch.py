"""PyInstaller runtime hook: patches applied before any application imports.

1. inspect.getsource — torch calls this at import time to parse @compile_ignored
   comments. In PyInstaller bundles, .py source files don't exist → OSError.

2. lameenc stub — demucs.audio imports lameenc (LGPL) at module level for MP3
   encoding. We only output WAV, so we inject a no-op stub to avoid bundling it.
"""
import inspect
import sys
import types

# --- Patch inspect.getsource for PyTorch compatibility ---
_orig_getsource = inspect.getsource


def _getsource_safe(obj, **kwargs):
    try:
        return _orig_getsource(obj, **kwargs)
    except OSError:
        return ""


inspect.getsource = _getsource_safe

# --- Stub out lameenc (LGPL) — we never encode MP3 ---
_lameenc = types.ModuleType("lameenc")
_lameenc.__doc__ = "Stub module: lameenc excluded from binary (LGPL, unused)"
sys.modules["lameenc"] = _lameenc
