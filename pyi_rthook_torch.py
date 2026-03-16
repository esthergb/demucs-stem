"""PyInstaller runtime hook: patch inspect.getsource before torch imports.

torch.utils._config_module.get_assignments_with_compile_ignored_comments()
calls inspect.getsource() at torch import time to parse @compile_ignored
comments. In PyInstaller bundles, .py source files may not exist → OSError.

We only patch getsource (not findsource/getsourcelines) because torch.jit
uses those separately with its own error handling for @overload functions.
"""
import inspect

_orig_getsource = inspect.getsource


def _getsource_safe(obj, **kwargs):
    try:
        return _orig_getsource(obj, **kwargs)
    except OSError:
        return ""


inspect.getsource = _getsource_safe
