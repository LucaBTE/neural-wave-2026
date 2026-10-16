"""Sartiq reference-similarity scorer (v2).

Public API (``api.py``)::

    from sartiq_score import score_pair, Scorer
    report = score_pair("ref.jpg", "sub.png")

``types``, ``io``, ``segment`` and ``align`` are the preprocessing side;
``metrics``, ``aggregate`` and ``api`` the scoring side; ``cli``, ``batch``,
``sanity`` and ``debug_sheet`` the presentation/event side.

``api`` is imported lazily (PEP 562) so the preprocessing modules can be used
without pulling in torch.
"""

from __future__ import annotations

import os

from .types import (
    BG_RGB,
    CANON_FILL,
    CANON_H,
    CANON_MARGIN,
    CANON_SHAPE,
    CANON_W,
    MAX_ASPECT,
    MAX_IMAGE_PIXELS,
    MAX_SIDE,
    MIN_SIDE,
    Canon,
    InvalidImage,
    InvalidReason,
    Loaded,
    ScoreReport,
    flags_from_canon,
)

__version__ = "2.2.0"

#: Pinned model weight cache (SPEC: U2NET_HOME / TORCH_HOME / HF_HOME here).
MODEL_DIR = os.environ.get(
    "SCORER_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".models"),
)

__all__ = [
    "BG_RGB",
    "CANON_FILL",
    "CANON_H",
    "CANON_MARGIN",
    "CANON_SHAPE",
    "CANON_W",
    "MAX_ASPECT",
    "MAX_IMAGE_PIXELS",
    "MAX_SIDE",
    "MIN_SIDE",
    "Canon",
    "InvalidImage",
    "InvalidReason",
    "Loaded",
    "MODEL_DIR",
    "ScoreReport",
    "Scalar",
    "__version__",
    "canonicalize",
    "flags_from_canon",
    "load_image",
    "score_pair",
]

api = None  # placeholder replaced by the real submodule on first access


def _get_api():
    """Import (and cache) :mod:`sartiq_score.api`."""
    global api
    if api is None or not hasattr(api, "score_pair"):
        import importlib  # noqa: PLC0415

        api = importlib.import_module(".api", __name__)
    return api


def __getattr__(name: str):  # PEP 562
    if name in {"score_pair", "Scorer", "Scalar"}:
        return getattr(_get_api(), name)
    if name == "load_image":
        from .io import load_image  # noqa: PLC0415

        return load_image
    if name == "canonicalize":
        from .align import canonicalize  # noqa: PLC0415

        return canonicalize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
