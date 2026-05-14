"""Hard-fail import if mlx-audio is not pinned to 0.4.2.

mlx-audio 0.4.3 ships a Fish Speech regression that produces degenerative
token loops. 0.4.2 is the known-good version.
"""
import mlx_audio

REQUIRED_VERSION = "0.4.2"
_actual = getattr(mlx_audio, "__version__", None)
if _actual is None:
    try:
        from importlib.metadata import version as _v
        _actual = _v("mlx-audio")
    except Exception:
        _actual = "unknown"

if _actual != REQUIRED_VERSION:
    raise ImportError(
        f"vo/ requires mlx-audio=={REQUIRED_VERSION} (found {_actual}). "
        f"Run: .venv-mlx/bin/pip install 'mlx-audio=={REQUIRED_VERSION}'"
    )
