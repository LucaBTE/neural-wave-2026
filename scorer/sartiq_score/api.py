"""Public API: :func:`score_pair` and the model-caching :class:`Scorer`.

This is the only module that touches the filesystem and the
segmentation/alignment modules; everything downstream
(:mod:`~sartiq_score.metrics`, :mod:`~sartiq_score.aggregate`) is pure.

``Scorer`` loads LPIPS + OpenCLIP once, pins torch's thread count and
determinism, and records package/model identities (including SHA-256 of the
weight files) into every :class:`~sartiq_score.types.ScoreReport`.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time
from typing import Any

import numpy as np

from . import aggregate
from .metrics import MetricContext
from .types import Canon, InvalidImage, ScoreReport

#: Public scalar type alias (re-exported by ``sartiq_score.__init__``).
Scalar = float

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.environ.get("SCORER_MODEL_DIR", os.path.join(_HERE, ".models"))

#: Fixed thread count used for every scoring run (SPEC: torch.set_num_threads(4)).
DEFAULT_THREADS = 4


# ---------------------------------------------------------------------------
# Weight-file discovery / hashing
# ---------------------------------------------------------------------------


def _sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        hits = [p for p in hits if os.path.isfile(p)]
        if hits:
            # prefer the largest (the real checkpoint, not a small config)
            return max(hits, key=os.path.getsize)
    return None


def _configure_model_env(model_dir: str) -> None:
    """Point every cache at the pinned ``.models`` bundle (SPEC)."""
    torch_home = os.path.join(model_dir, "torch")
    hf_home = os.path.join(model_dir, "hf")
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("U2NET_HOME", os.path.join(model_dir, "u2net"))
    os.environ.setdefault("XDG_CACHE_HOME", model_dir)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    """Loads LPIPS + OpenCLIP once and scores reference/submission pairs.

    Parameters
    ----------
    device:
        Torch device string; the challenge runs on CPU.
    threads:
        ``torch.set_num_threads`` value (SPEC default 4).
    model_dir:
        Pinned weight cache (defaults to ``SCORER_MODEL_DIR`` or ``.models``).
    load_models:
        Set False for unit tests that only exercise the preprocessing glue.
    """

    def __init__(
        self,
        device: str = "cpu",
        threads: int = DEFAULT_THREADS,
        model_dir: str | None = None,
        load_models: bool = True,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.threads = int(threads)
        torch.set_num_threads(self.threads)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass

        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        _configure_model_env(self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        self.ctx = MetricContext(device=self.device, clip_size=224)
        self._versions: dict[str, Any] | None = None
        self._lpips_module = None
        self._clip_hub_model_id = "ViT-B-32/openai"

        if load_models:
            self._load_models()

    # -- model loading -----------------------------------------------------

    def _load_models(self) -> None:
        import lpips  # noqa: PLC0415
        import open_clip  # noqa: PLC0415

        # LPIPS alex v0.1 (weights bundled with the pip package).
        self._lpips_module = lpips
        lpips_fn = lpips.LPIPS(net="alex")
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        lpips_fn.to(self.device)
        self.ctx.lpips_fn = lpips_fn

        # OpenCLIP ViT-B-32 / openai (pinned cache dir).
        clip_cache = os.path.join(self.model_dir, "open_clip")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", cache_dir=clip_cache
        )
        model.eval()
        model.to(self.device)
        self.ctx.clip_model = model
        self.ctx.clip_preprocess = preprocess
        self.ctx.clip_size = self._resolve_clip_size(model)

    @staticmethod
    def _resolve_clip_size(model: Any) -> int:
        try:
            size = model.visual.image_size
        except Exception:  # pragma: no cover
            return 224
        if isinstance(size, (tuple, list)):
            return int(max(size))
        return int(size)

    # -- versions ----------------------------------------------------------

    @property
    def versions(self) -> dict[str, Any]:
        if self._versions is None:
            self._versions = self._build_versions()
        return self._versions

    def _build_versions(self) -> dict[str, Any]:
        import numpy  # noqa: PLC0415
        import PIL  # noqa: PLC0415
        import skimage  # noqa: PLC0415
        import torch  # noqa: PLC0415

        versions: dict[str, Any] = {
            "package": "sartiq-score 2.2.0",
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "scikit-image": skimage.__version__,
            "threads": self.threads,
        }
        try:
            import lpips as _lp  # noqa: PLC0415

            versions["lpips"] = getattr(_lp, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass
        try:
            import open_clip as _oc  # noqa: PLC0415

            versions["open-clip"] = getattr(_oc, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass
        try:
            import rembg as _rb  # noqa: PLC0415

            versions["rembg"] = getattr(_rb, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass

        # --- LPIPS alex weights + backbone sha256 ---
        versions["lpips_model"] = "alex@v0.1"
        try:
            import lpips as _lp  # noqa: PLC0415

            alex = os.path.join(os.path.dirname(_lp.__file__), "weights", "v0.1", "alex.pth")
            if os.path.exists(alex):
                versions["lpips_alex_sha256"] = _sha256_path(alex)
        except Exception:  # pragma: no cover
            pass
        backbone = _first_existing(
            [os.path.join(self.model_dir, "torch", "hub", "checkpoints", "alexnet-*.pth")]
        )
        if backbone:
            versions["lpips_backbone_sha256"] = _sha256_path(backbone)

        # --- CLIP weights sha256 ---
        versions["clip_model"] = self._clip_hub_model_id
        clip_path = _first_existing(
            [
                os.path.join(self.model_dir, "open_clip", "**", "*.safetensors"),
                os.path.join(self.model_dir, "open_clip", "**", "*.bin"),
                os.path.join(self.model_dir, "open_clip", "**", "*.pt"),
            ]
        )
        if clip_path:
            versions["clip_sha256"] = _sha256_path(clip_path)

        # --- segmenter identity ---
        try:
            from . import segment as _seg  # noqa: PLC0415

            versions["segmenter"] = _seg.model_id()
        except Exception:  # pragma: no cover
            pass
        return versions

    # -- scoring -----------------------------------------------------------

    def score_canons(
        self,
        ref: Canon,
        sub: Canon,
        *,
        view: str | None = None,
        timing_s: float = 0.0,
    ) -> ScoreReport:
        """Score a pre-canonicalised pair (used by tests and by ``score_pair``)."""
        return aggregate.build_report(
            ref, sub, self.ctx, versions=self.versions, timing_s=timing_s, view=view
        )

    def score_against_canon(
        self,
        ref_canon: Any,
        sub_path: str | os.PathLike,
        view: str | None = None,
        t0: float | None = None,
    ) -> ScoreReport:
        """Score a submission file against an already-canonicalised reference.

        v2.1: the acceptance harness scores dozens of variants of the *same*
        reference, so re-decoding and re-segmenting the reference for every
        variant was pure waste (~40 % of the per-pair cost).  This is the same
        code path ``score_pair`` uses, minus the reference work.

        Unreadable or out-of-limit submissions yield ``status="invalid"``,
        score 0; a flat-field submission yields ``blank_or_uniform``.
        """
        from . import align, io, segment  # noqa: PLC0415

        t0 = time.perf_counter() if t0 is None else t0
        try:
            sub_loaded = io.load_image(sub_path)
        except InvalidImage as exc:
            return aggregate.invalid_report(
                exc.reason, versions=self.versions, view=view, detail=exc.detail
            )
        except Exception as exc:  # never crash on a bad file
            return aggregate.invalid_report(
                "processing_error", versions=self.versions, view=view,
                detail=f"sub:{type(exc).__name__}",
            )

        # --- v2.1 reference-free flat-field gate (blank / solid blob) --------
        if aggregate.is_uniform_submission(sub_loaded.rgb):
            return aggregate.invalid_report(
                aggregate.BLANK_REASON,
                versions=self.versions,
                view=view,
                detail=f"luma_std={aggregate.luma_std(sub_loaded.rgb):.3f}",
            )

        try:
            sub_mask = segment.segment(sub_loaded.rgb)
            sub_canon = align.canonicalize(sub_loaded, sub_mask)
        except Exception as exc:
            return aggregate.invalid_report(
                "processing_error", versions=self.versions, view=view,
                detail=f"{type(exc).__name__}:{exc}",
            )
        return self.score_canons(
            ref_canon, sub_canon, view=view, timing_s=round(time.perf_counter() - t0, 3)
        )

    def score_pair(
        self,
        ref_path: str | os.PathLike,
        sub_path: str | os.PathLike,
        view: str | None = None,
        ref_mask_path: str | os.PathLike | None = None,
    ) -> ScoreReport:
        """Score a reference file against a submission file.

        Unreadable or out-of-limit inputs yield ``status="invalid", score=0``;
        no other failure mode raises out of this method.
        """
        from . import align, io, segment  # noqa: PLC0415

        t0 = time.perf_counter()
        # --- decode (limit violations -> invalid) ---
        try:
            ref_loaded = io.load_image(ref_path)
        except InvalidImage as exc:
            return aggregate.invalid_report(
                exc.reason, versions=self.versions, view=view, detail=exc.detail
            )
        except Exception as exc:  # never crash on a bad file
            return aggregate.invalid_report(
                "processing_error", versions=self.versions, view=view,
                detail=f"ref:{type(exc).__name__}",
            )

        # --- segment + canonicalise the reference ---------------------------
        try:
            ref_mask = segment.segment(ref_loaded.rgb)
            if ref_mask_path is not None:
                override = align.load_reference_mask(
                    os.path.dirname(str(ref_mask_path)),
                    os.path.splitext(os.path.basename(str(ref_mask_path)))[0],
                    (ref_loaded.width, ref_loaded.height),
                )
                if override is not None:
                    ref_mask = override
            ref_canon = align.canonicalize(ref_loaded, ref_mask)
        except Exception as exc:
            return aggregate.invalid_report(
                "processing_error", versions=self.versions, view=view,
                detail=f"ref:{type(exc).__name__}:{exc}",
            )

        return self.score_against_canon(ref_canon, sub_path, view=view, t0=t0)


# ---------------------------------------------------------------------------
# Module-level convenience API (lazy default Scorer)
# ---------------------------------------------------------------------------

_DEFAULT_SCORER: Scorer | None = None


def _default_scorer() -> Scorer:
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = Scorer()
    return _DEFAULT_SCORER


def score_pair(
    ref_path: str | os.PathLike,
    sub_path: str | os.PathLike,
    view: str | None = None,
    ref_mask_path: str | os.PathLike | None = None,
) -> ScoreReport:
    """Score *sub_path* against *ref_path* using a cached default :class:`Scorer`."""
    return _default_scorer().score_pair(ref_path, sub_path, view=view, ref_mask_path=ref_mask_path)


def reset_default_scorer() -> None:
    """Drop the cached default scorer (tests / thread changes)."""
    global _DEFAULT_SCORER
    _DEFAULT_SCORER = None


__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_THREADS",
    "Scalar",
    "Scorer",
    "reset_default_scorer",
    "score_pair",
]
