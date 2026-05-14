"""vo - Fish Audio S2 Pro voiceover renderer for Content Machine."""
from vo import _version_check  # noqa: F401  (import-time version gate)

__all__ = ["render", "RenderResult"]


def __getattr__(name: str):
    # Lazy import so just importing the package doesn't load the model.
    if name in ("render", "RenderResult"):
        from vo.render import render, RenderResult
        return {"render": render, "RenderResult": RenderResult}[name]
    raise AttributeError(name)
