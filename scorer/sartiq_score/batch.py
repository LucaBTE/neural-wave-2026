#!/usr/bin/env python3
"""Leaderboard batch scoring — SPEC "Event rules".

Layouts::

    <refs>/<garment_id>/<view>.<png|jpg|jpeg|webp>
    <subs>/<team>/<garment_id>/<view>.<png|jpg|jpeg|webp>

Rules implemented here:

* required views per garment = the views present in ``<refs>``;
* garment score = **mean over required views**, a missing submission view scores 0;
* team score = **mean over all reference garments**, a missing garment scores 0;
* scores rounded to 2 decimals; ties broken by the aggregate colour sub-score,
  then by the aggregate lpips sub-score;
* outputs ``leaderboard.csv``, ``scores.jsonl``, ``review.csv`` and
  ``debug/<team>/<garment>_<view>.jpg`` contact sheets.

``<refs>/garments.json`` may declare ``{"<garment_id>": {"target_style": "flat_lay"|"ghost"}}``
(also accepted nested under ``"garments"``); it is carried into the outputs.
"""

from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .cli import build_scorer, jsonable, report_to_dict, score_with

IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".heic")
HIDDEN = re.compile(r"^\..*")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _view_files(directory: Path) -> dict[str, Path]:
    """Map ``view -> path`` for one garment/reference directory."""
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for entry in sorted(directory.iterdir()):
        if HIDDEN.match(entry.name):
            continue
        if not (entry.is_symlink() or entry.is_file()):
            continue
        ext = entry.suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        view = entry.stem
        prior = found.get(view)
        if prior is None or IMAGE_EXTS.index(ext) < IMAGE_EXTS.index(prior.suffix.lower()):
            found[view] = entry
    return found


def load_garment_meta(refs_dir: Path | None) -> dict[str, dict]:
    if refs_dir is None:
        return {}
    path = Path(refs_dir) / "garments.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if isinstance(data, dict) and isinstance(data.get("garments"), dict):
        data = data["garments"]
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for gid, meta in data.items():
        out[str(gid)] = meta if isinstance(meta, dict) else {}
    return out


#: v2.2: every target is flat-lay; ``garments.json`` is
#: optional and only ever overrides this default.
DEFAULT_TARGET_STYLE = "flat_lay"

_GID_KEYS = ("garment_id", "garment", "id", "gid", "name")
_VIEW_KEYS = ("view", "view_name", "angle")
_REF_KEYS = ("ref", "ref_path", "reference", "reference_path", "path", "target", "target_path")
_SUB_KEYS = ("sub", "submission", "submission_path", "generated", "input", "input_path")


def _first_key(row: dict, keys) -> str | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return None


def load_manifest(manifest) -> tuple[dict[str, dict[str, Path]], dict[tuple[str, str], Path]]:
    """Parse a JSONL validation manifest into a reference map.

    One JSON object per line, minimally ``{"garment_id": ..., "view": ...,
    "ref": <path>}``.  Field aliases are accepted (``garment``/``id``,
    ``reference``/``ref_path``, ...).  An optional ``sub``/``submission`` field
    is used as a fallback submission for that slot when the submission tree has
    no file for it.

    Returns ``(refs, explicit_subs)`` with ``refs`` in the same shape as
    :func:`discover_refs`.
    """
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    refs: dict[str, dict[str, Path]] = {}
    explicit: dict[tuple[str, str], Path] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                continue
            gid = _first_key(row, _GID_KEYS)
            ref = _first_key(row, _REF_KEYS)
            if not gid or not ref:
                continue
            view = _first_key(row, _VIEW_KEYS) or Path(ref).stem or "front"
            refs.setdefault(gid, {})[view] = Path(ref)
            sub = _first_key(row, _SUB_KEYS)
            if sub:
                explicit[(gid, view)] = Path(sub)
    if not refs:
        raise ValueError(f"manifest {path} declared no (garment_id, view, ref) rows")
    return refs, explicit



def discover_refs(refs_dir, limit: int | None = None) -> dict[str, dict[str, Path]]:
    refs_root = Path(refs_dir)
    if not refs_root.is_dir():
        raise FileNotFoundError(f"reference directory not found: {refs_root}")
    refs: dict[str, dict[str, Path]] = {}
    for entry in sorted(refs_root.iterdir()):
        if not entry.is_dir() or HIDDEN.match(entry.name):
            continue
        views = _view_files(entry)
        if views:
            refs[entry.name] = views
    # also accept '<garment>__<view>.ext' flat files
    flat: dict[str, dict[str, Path]] = {}
    for entry in sorted(refs_root.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTS:
            continue
        if "__" not in entry.stem:
            continue
        gid, _, view = entry.stem.partition("__")
        flat.setdefault(gid, {})[view] = entry
    for gid, views in flat.items():
        refs.setdefault(gid, views)
    if limit is not None:
        refs = {gid: refs[gid] for gid in sorted(refs)[: int(limit)]}
    return refs


def discover_subs(subs_dir) -> dict[str, dict[str, dict[str, Path]]]:
    subs_root = Path(subs_dir)
    if not subs_root.is_dir():
        raise FileNotFoundError(f"submission directory not found: {subs_root}")
    teams: dict[str, dict[str, dict[str, Path]]] = {}
    for entry in sorted(subs_root.iterdir()):
        # v2.2: symlinked team / garment folders are never followed (they could point at refs)
        if entry.is_symlink() or not entry.is_dir() or HIDDEN.match(entry.name):
            continue
        garments: dict[str, dict[str, Path]] = {}
        for gdir in sorted(entry.iterdir()):
            if gdir.is_symlink() or not gdir.is_dir() or HIDDEN.match(gdir.name):
                continue
            views = _view_files(gdir)
            if views:
                garments[gdir.name] = views
        teams[entry.name] = garments
    return teams


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

_SCORER = None


def _init_worker() -> None:  # pragma: no cover - runs in child processes
    global _SCORER
    _SCORER = build_scorer()


def _score_task(task: dict) -> dict:
    global _SCORER
    if _SCORER is None:
        _SCORER = build_scorer()
    record = dict(task)
    record["missing"] = not task.get("sub")
    if record["missing"]:
        record.update(score=0.0, status="missing", subscores={}, raw={}, flags=[], timing_s=0.0)
        return record
    if task.get("from_team_tree") and Path(task["sub"]).is_symlink():
        record.update(score=0.0, status="invalid", subscores={}, raw={},
                      flags=["invalid:symlink_not_allowed"], timing_s=0.0)
        return record
    try:
        report = score_with(_SCORER, task["ref"], task["sub"], task.get("view"), task.get("ref_mask"))
    except Exception as exc:  # never abort a whole event on one bad file
        record.update(
            score=0.0,
            status="invalid",
            subscores={},
            raw={},
            flags=[f"scorer_error:{type(exc).__name__}"],
            timing_s=0.0,
            error=str(exc),
        )
        return record
    d = report_to_dict(report)
    record.update(
        score=float(d.get("score") or 0.0),
        status=d.get("status", "ok"),
        subscores=d.get("subscores") or {},
        raw=d.get("raw") or {},
        flags=d.get("flags") or [],
        timing_s=d.get("timing_s", 0.0),
        versions=d.get("versions") or {},
    )
    return record


def build_tasks(refs: dict, subs: dict, teams: list[str], explicit: dict | None = None,
                ref_mask_dir=None) -> list[dict]:
    from .cli import resolve_ref_mask  # noqa: PLC0415

    explicit = explicit or {}
    ref_masks: dict[tuple[str, str], str | None] = {}
    if ref_mask_dir:
        for gid, views in refs.items():
            for view, path in views.items():
                ref_masks[(gid, view)] = resolve_ref_mask(path, ref_mask_dir, garment=gid, view=view)
    tasks: list[dict] = []
    for team in teams:
        team_subs = subs.get(team, {})
        for gid in sorted(refs):
            views = refs[gid]
            for view in sorted(views):
                team_file = team_subs.get(gid, {}).get(view)
                # v2.2: a manifest `sub` is used ONLY in manifest-only mode (no team tree);
                # with --subs a missing team file must score 0, never inherit another image
                sub_path = team_file if subs else explicit.get((gid, view))
                tasks.append(
                    {
                        "team": team,
                        "garment": gid,
                        "view": view,
                        "ref": str(views[view]),
                        "sub": str(sub_path) if sub_path else None,
                        "ref_mask": ref_masks.get((gid, view)),
                        "from_team_tree": bool(subs),
                    }
                )
    return tasks


def write_garments_csv(refs: dict, meta: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["garment", "view", "reference", "target_style"])
        for gid in sorted(refs):
            style = (meta.get(gid) or {}).get("target_style") or DEFAULT_TARGET_STYLE
            for view in sorted(refs[gid]):
                writer.writerow([gid, view, str(refs[gid][view]), style])


def _aggregate(tasks: list[dict], records: list[dict], refs: dict, meta: dict) -> list[dict]:
    by_team: dict[str, dict] = {}
    for rec in records:
        team = rec["team"]
        entry = by_team.setdefault(
            team,
            {
                "team": team,
                "garment_views": {},
                "garment_any": {},
                "colour": 0.0,
                "lpips": 0.0,
                "clip": 0.0,
                "geometry": 0.0,
                "ssim": 0.0,
                "slots": 0,
                "pairs_scored": 0,
                "pairs_missing": 0,
                "n_review": 0,
                "n_invalid": 0,
            },
        )
        entry["slots"] += 1
        missing = bool(rec.get("missing"))
        entry["pairs_missing" if missing else "pairs_scored"] += 1
        if rec.get("status") == "review":
            entry["n_review"] += 1
        if rec.get("status") == "invalid":
            entry["n_invalid"] += 1
        subs = rec.get("subscores") or {}
        for key in ("colour", "lpips", "clip", "geometry", "ssim"):
            entry[key] += float(subs.get(key) or 0.0)
            entry.setdefault("garment_sub", {}).setdefault(rec["garment"], {}).setdefault(key, []).append(
                float(subs.get(key) or 0.0))
        entry["garment_views"].setdefault(rec["garment"], []).append(float(rec.get("score") or 0.0))
        entry["garment_any"][rec["garment"]] = entry["garment_any"].get(rec["garment"], False) or (
            not missing
        )

    n_garments = max(1, len(refs))
    rows: list[dict] = []
    for team, entry in by_team.items():
        garment_scores = entry.pop("garment_views")
        any_scored = entry.pop("garment_any")
        scores = [sum(garment_scores.get(gid, [0.0])) / len(refs[gid]) for gid in refs]
        gsub = entry.pop("garment_sub", {})

        def _gmean(key: str) -> float:
            # v2.2: sub-score means use the SAME garment weighting as the main score
            vals = [sum(gsub.get(gid, {}).get(key, [0.0])) / len(refs[gid]) for gid in refs]
            return sum(vals) / n_garments
        rows.append(
            {
                "team": team,
                "score": round(sum(scores) / n_garments, 2),
                "colour": round(_gmean("colour"), 6),
                "lpips": round(_gmean("lpips"), 6),
                "clip": round(_gmean("clip"), 6),
                "geometry": round(_gmean("geometry"), 6),
                "ssim": round(_gmean("ssim"), 6),
                "pairs_scored": entry["pairs_scored"],
                "pairs_missing": entry["pairs_missing"],
                "garments_scored": sum(1 for gid in refs if any_scored.get(gid)),
                "garments_missing": sum(1 for gid in refs if not any_scored.get(gid)),
                "n_review": entry["n_review"],
                "n_invalid": entry["n_invalid"],
            }
        )
    rows.sort(key=lambda r: (-r["score"], -r["colour"], -r["lpips"], r["team"]))
    # v2.2: teams tied on every published criterion share a rank (alphabetical order is display only)
    prev = None
    for i, row in enumerate(rows, start=1):
        key = (row["score"], row["colour"], row["lpips"])
        row["rank"] = rows[i - 2]["rank"] if key == prev else i
        prev = key
    return rows


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def write_leaderboard(rows: list[dict], path: Path) -> None:
    cols = [
        "rank", "team", "score", "colour", "lpips", "clip", "geometry", "ssim",
        "pairs_scored", "pairs_missing", "garments_scored", "garments_missing",
        "n_review", "n_invalid",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_scores_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(jsonable(rec), sort_keys=True) + "\n")


def write_review_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team", "garment", "view", "score", "status", "flags", "submission"])
        for rec in records:
            if rec.get("status") != "review":
                continue
            writer.writerow(
                [
                    rec["team"],
                    rec["garment"],
                    rec["view"],
                    f"{rec.get('score', 0.0):.2f}",
                    rec["status"],
                    "|".join(map(str, rec.get("flags") or [])),
                    rec.get("sub") or "",
                ]
            )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_batch(
    refs_dir=None,
    subs_dir=None,
    out_dir=None,
    manifest=None,
    debug_sheets: bool = False,
    limit: int | None = None,
    jobs: int = 1,
    ref_mask_dir=None,
) -> dict:
    """Score a submission tree against either a reference folder or a manifest.

    ``refs_dir`` uses the ``<refs>/<garment_id>/<view>.<ext>`` layout; the
    alternative ``manifest`` is a JSONL validation manifest whose rows declare
    ``garment_id``, ``view`` and ``ref`` (see :func:`load_manifest`).  Exactly
    one of the two must be given.  ``subs_dir`` is optional: when omitted every
    submission must come from the manifest's own ``sub`` field.
    """
    if manifest:
        refs, explicit = load_manifest(manifest)
        if limit is not None:
            refs = {gid: refs[gid] for gid in sorted(refs)[: int(limit)]}
        meta = load_garment_meta(refs_dir)
    elif refs_dir:
        refs = discover_refs(refs_dir, limit=limit)
        explicit = {}
        meta = load_garment_meta(refs_dir)
    else:
        raise ValueError("run_batch needs either refs_dir or a manifest")

    if subs_dir:
        subs = discover_subs(subs_dir)
        teams = sorted(subs)
        if not teams:
            raise ValueError(f"no team directories found under {subs_dir}")
    else:
        subs, teams = {}, ["manifest"]

    out = Path(out_dir or ".")
    out.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(refs, subs, teams, explicit, ref_mask_dir)
    if jobs and jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp  # noqa: PLC0415

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=ctx, initializer=_init_worker
        ) as pool:
            futures = [pool.submit(_score_task, t) for t in tasks]
            records = []
            for t, fut in zip(tasks, futures):
                try:
                    records.append(fut.result())
                except Exception as exc:  # v2.2: a dead worker never aborts the event
                    rec = dict(t, missing=False, score=0.0, status="invalid", subscores={}, raw={},
                               flags=[f"scorer_error:{type(exc).__name__}"], timing_s=0.0, error=str(exc))
                    records.append(rec)
    else:
        _init_worker()
        records = [_score_task(task) for task in tasks]

    for rec in records:
        style = (meta.get(rec["garment"]) or {}).get("target_style") or DEFAULT_TARGET_STYLE
        rec["target_style"] = style

    rows = _aggregate(tasks, records, refs, meta)

    write_leaderboard(rows, out / "leaderboard.csv")
    write_scores_jsonl(records, out / "scores.jsonl")
    write_review_csv(records, out / "review.csv")
    write_garments_csv(refs, meta, out / "garments.csv")

    if debug_sheets:
        from .debug_sheet import render_debug_sheet  # noqa: PLC0415

        scorer = build_scorer()
        for rec in records:
            if rec.get("missing") or rec.get("status") == "invalid":
                continue
            target = out / "debug" / rec["team"] / f"{rec['garment']}_{rec['view']}.jpg"
            try:
                render_debug_sheet(rec["ref"], rec["sub"], target, rec["view"])
            except Exception as exc:  # pragma: no cover - sheet is best effort
                print(f"  ! debug sheet failed for {rec['garment']}/{rec['view']}: {exc}")

    return {
        "leaderboard": str(out / "leaderboard.csv"),
        "scores_jsonl": str(out / "scores.jsonl"),
        "review_csv": str(out / "review.csv"),
        "garments_csv": str(out / "garments.csv"),
        "rows": rows,
        "records": records,
        "n_teams": len(teams),
        "n_garments": len(refs),
        "n_pairs_scored": sum(1 for r in records if not r.get("missing")),
        "n_pairs_missing": sum(1 for r in records if r.get("missing")),
        "n_review": sum(1 for r in records if r.get("status") == "review"),
        "n_invalid": sum(1 for r in records if r.get("status") == "invalid"),
    }
