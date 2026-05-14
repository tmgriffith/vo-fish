"""Test the import-time version pin."""
import importlib
import sys

import pytest


def test_version_check_passes_on_correct_version():
    """Sanity: the version pin doesn't fire under the current pinned env."""
    # If we got here at all (vo is already imported), the pin passed.
    import vo._version_check  # noqa: F401


def test_version_check_raises_on_wrong_version(monkeypatch):
    """Force a wrong version and confirm ImportError fires on reload."""
    import mlx_audio
    monkeypatch.setattr(mlx_audio, "__version__", "0.4.3", raising=False)

    # Also mock importlib.metadata.version since _version_check falls back to it
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.4.3")

    # Reload the module fresh
    sys.modules.pop("vo._version_check", None)
    with pytest.raises(ImportError, match="requires mlx-audio==0.4.2"):
        import vo._version_check  # noqa: F401
