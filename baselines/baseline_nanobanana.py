#!/usr/bin/env python3
"""Hosted image-model baseline: "just prompt a big model at it".

Reads the first input of every pair, asks a hosted image-edit model to turn it into
a clean flat-lay packshot, and writes the results into the official submission
layout:

    <out>/<team>/<id>/<view>.png

It reads either dataset manifest, exactly like ``baseline_identity.py``:
``train/pairs.jsonl`` or ``dev|test/inputs.jsonl``; paths are relative to the
dataset root, resolved the same way (``--dataset`` overrides).

Provider-agnostic. The API key is read from the environment and is never stored,
logged or written into the repo:

    fal     ->  FAL_KEY          (default provider; fal queue REST API)
    gemini  ->  GEMINI_API_KEY or GOOGLE_API_KEY
    openai  ->  OPENAI_API_KEY   (base URL from OPENAI_BASE_URL, default https://api.openai.com/v1)
    any     ->  IMAGE_API_KEY overrides the provider key

The fal provider is the default and needs no ``--model``: it targets the fal queue
endpoint ``google/nano-banana-2.1/edit``. It submits the input as a JPEG data URI,
polls the queue until the edit completes, then downloads the result. Only the
standard library is used for the HTTP calls.

Provider bytes are decoded and re-saved as a genuine PNG when Pillow is installed
(the repo's venv), so a provider that returns JPEG/WebP still lands as `front.png`.
Only the standard library is required otherwise.

    python baselines/baseline_nanobanana.py --manifest ../sartiq_neural_wave_2026/dev/inputs.jsonl --out predictions --team hosted-edit --dry-run
    python baselines/baseline_nanobanana.py --manifest ../sartiq_neural_wave_2026/train/pairs.jsonl --out predictions --team hosted-edit --limit 4

Every real call appends one line to ``<out>/_manifest.jsonl`` (model, prompt,
latency, error), so a run is reproducible afterwards. ``--dry-run`` writes a
*preview* of each request payload to ``<out>/_requests/`` -- the inline image data
is truncated so the files stay small -- and makes no network call, so it needs no
key. Only a transient failure (HTTP 429/5xx, a timeout, a connection error) is
retried; a deterministic provider error (e.g. 401) is reported once.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from baseline_identity import DEFAULT_VIEW, load_rows, resolve_dataset_root  # same folder

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULT_PROMPT = (
    "Turn this photograph of a single garment into a clean e-commerce flat-lay packshot. "
    "Lay the garment perfectly flat and centred on a plain, uniform light-grey background "
    "(RGB 245,245,245), seen straight from above with no perspective distortion. Use soft, "
    "even studio lighting with no visible shadow on the background. Keep the exact same "
    "garment, colour, material, pattern and proportions as the original. Remove the person, "
    "mannequin, hanger, floor, furniture and background; do not add props, text, logos, "
    "watermarks or reflections. Output only the final packshot image."
)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: fal queue REST API. The model id is appended to the base for the submit call.
FAL_QUEUE_BASE = "https://queue.fal.run"
DEFAULT_FAL_MODEL = "google/nano-banana-2.1/edit"
FAL_POLL_SECONDS = 2.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def api_key(provider: str) -> str:
    names = ["IMAGE_API_KEY"]
    if provider == "fal":
        names.append("FAL_KEY")
        hint = "FAL_KEY"
    elif provider == "gemini":
        names += ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        hint = "GEMINI_API_KEY or GOOGLE_API_KEY"
    else:
        names.append("OPENAI_API_KEY")
        hint = "OPENAI_API_KEY"
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit(
        f"error: no API key in the environment. Set {hint} "
        f"(or IMAGE_API_KEY). Never put a key in the repo."
    )


# --------------------------------------------------------------------------- #
# credential-safe error reporting
# --------------------------------------------------------------------------- #

#: Environment variables a provider credential may live in.
_SECRET_ENV_VARS = ("IMAGE_API_KEY", "FAL_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")

#: Wrapper/key shapes that carry a credential even when the exact value is unknown.
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\b(?:key|bearer)\s+[A-Za-z0-9._\-]{8,}"),          # "Key xxx", "Bearer xxx"
    re.compile(r"(?i)authorization\s*[:=]\s*\S+"),
    re.compile(r"(?i)x-goog-api-key\s*[:=]\s*\S+"),
    re.compile(r'(?i)"(?:api[_-]?key|key|token|access[_-]?token|authorization|secret|password)"'
               r'\s*:\s*"[^"]*"'),
    re.compile(r"(?i)[?&](?:key|api[_-]?key|access[_-]?token|token)=[^&\s\"']+"),
)

#: Error fields safe to show. Everything else in a provider body is dropped, so a body that
#: echoes the request (and therefore the credential) can never leak through.
_ALLOWED_ERROR_KEYS = {
    "code", "status", "message", "msg", "detail", "details", "error", "errors",
    "reason", "type", "title", "name", "description",
}


def _current_secrets(key: str | None = None) -> list[str]:
    values = [os.environ[name] for name in _SECRET_ENV_VARS if os.environ.get(name)]
    if key and key not in values:
        values.append(key)
    return values


def redact(text: object, secrets: list[str] | None = None) -> str:
    """Replace credentials (known values and key-like shapes) with ``[redacted]``."""
    out = str(text)
    for value in secrets or _current_secrets():
        if value:
            out = out.replace(value, "[redacted]")
    for pattern in _CREDENTIAL_PATTERNS:
        out = pattern.sub("[redacted]", out)
    return out


def _allowlist(value: object, depth: int = 0) -> object:
    if depth > 3:
        return "..."
    if isinstance(value, dict):
        return {str(k): _allowlist(v, depth + 1) for k, v in value.items()
                if str(k).lower() in _ALLOWED_ERROR_KEYS}
    if isinstance(value, list):
        return [_allowlist(v, depth + 1) for v in value[:10]]
    if isinstance(value, str):
        return value[:400]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:400]


def summarize_provider_error(payload: object, secrets: list[str] | None = None) -> str:
    """A short, credential-free, allowlisted description of a provider error body."""
    if isinstance(payload, (bytes, bytearray)):
        text = payload.decode("utf-8", "replace")
    elif isinstance(payload, str):
        text = payload
    else:
        return redact(json.dumps(_allowlist(payload), ensure_ascii=False), secrets)[:500]
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return redact(text, secrets)[:500]
    return redact(json.dumps(_allowlist(obj), ensure_ascii=False), secrets)[:500]


def mime_of(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/png"


def data_uri(path: Path) -> str:
    """The input image as a base64 data URI, for the fal ``image_urls`` field."""
    blob = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_of(path)};base64,{blob}"


class TransientError(RuntimeError):
    """A failure worth retrying: HTTP 429/5xx, a timeout, or a connection error.

    Anything else (4xx auth/validation, a malformed provider response, a missing
    result image) is deterministic -- retrying it just burns quota and time.
    """


_TRANSIENT_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})


def _http_error(code: int, detail: str) -> RuntimeError:
    cls = TransientError if (code in _TRANSIENT_HTTP or 500 <= code < 600) else RuntimeError
    return cls(f"HTTP {code}: {detail}")


def _post(url: str, body: bytes, headers: dict[str, str], timeout: float,
          secrets: list[str] | None = None) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Read the body but never echo it raw: providers can reflect the request (and its
        # credential) back in an error. Only allowlisted fields survive, then redacted.
        detail = summarize_provider_error(exc.read(), secrets)
        raise _http_error(exc.code, detail) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransientError(f"network error: {redact(reason, secrets)}") from exc


def _get(url: str, headers: dict[str, str], timeout: float,
         secrets: list[str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = summarize_provider_error(exc.read(), secrets)
        raise _http_error(exc.code, detail) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransientError(f"network error: {redact(reason, secrets)}") from exc


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, (filename, data, ctype) in files.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def build_request(provider: str, model: str, prompt: str, image: Path,
                  base_url: str) -> tuple[str, dict[str, str], bytes]:
    """Return ``(url, headers, body)`` for one edit call. No key is included here."""
    if provider == "fal":
        payload = {"prompt": prompt, "image_urls": [data_uri(image)]}
        return (f"{FAL_QUEUE_BASE}/{model}",
                {"Content-Type": "application/json"},
                json.dumps(payload).encode("utf-8"))

    data = image.read_bytes()
    if provider == "gemini":
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_of(image),
                                     "data": base64.b64encode(data).decode("ascii")}},
                ]
            }]
        }
        url = GEMINI_URL.format(model=model)
        headers = {"Content-Type": "application/json"}
        return url, headers, json.dumps(payload).encode("utf-8")

    # openai-compatible images/edits
    body, ctype = _multipart(
        {"model": model, "prompt": prompt, "n": "1"},
        {"image": (image.name, data, mime_of(image))},
    )
    return f"{base_url.rstrip('/')}/images/edits", {"Content-Type": ctype}, body


def dry_payload(provider: str, body: bytes) -> bytes:
    """Readable payload for ``--dry-run``: truncate an inline image data URI.

    Providers other than fal keep the exact request bytes. For fal the payload is
    re-serialised with the ``image_urls`` data URI shortened, so the file stays small;
    the prompt and the rest of the request are unchanged.
    """
    if provider != "fal":
        return body
    try:
        obj = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    urls = obj.get("image_urls")
    if isinstance(urls, list):
        obj["image_urls"] = [
            f"{u[:48]}...<truncated, {len(u)} chars>"
            if isinstance(u, str) and u.startswith("data:") else u
            for u in urls
        ]
    return json.dumps(obj, indent=2).encode("utf-8")


def fal_generate(url: str, body: bytes, key: str, timeout: float) -> bytes:
    """Submit to the fal queue, poll until COMPLETED, return the first image bytes.

    ``timeout`` bounds the wait for the queued request to finish. Each HTTP request
    gets its own smaller timeout.
    """
    secrets = [key]
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    per_request = max(10.0, min(timeout, 60.0))

    submission = _post(url, body, headers, per_request, secrets)
    request_id = submission.get("request_id")
    if not request_id:
        raise RuntimeError(f"fal submit returned no request_id: "
                           f"{redact(json.dumps(submission), secrets)[:300]}")
    status_url = submission.get("status_url") or f"{url}/requests/{request_id}/status"
    response_url = submission.get("response_url") or f"{url}/requests/{request_id}"
    print(f"  fal request {request_id}", file=sys.stderr)

    deadline = time.time() + timeout
    while True:
        state = _get(status_url, headers, per_request, secrets)
        status = state.get("status")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "ERROR", "CANCELLED"):
            raise RuntimeError(f"fal request {request_id} ended with status {status}: "
                               f"{redact(json.dumps(state), secrets)[:300]}")
        if time.time() > deadline:
            raise TransientError(f"fal request {request_id} still {status or 'unknown'} "
                                 f"after {timeout:.0f}s")
        time.sleep(FAL_POLL_SECONDS)

    result = _get(response_url, headers, per_request, secrets)
    images = result.get("images") or []
    if not images or not images[0].get("url"):
        raise RuntimeError(f"fal result has no image url: "
                           f"{redact(json.dumps(result), secrets)[:300]}")
    try:
        with urllib.request.urlopen(images[0]["url"], timeout=per_request) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TransientError(f"failed to download the result image: "
                             f"{redact(getattr(exc, 'reason', exc), secrets)}") from exc


def extract_image(provider: str, response: dict, timeout: float,
                  secrets: list[str] | None = None) -> bytes:
    if provider == "gemini":
        for cand in response.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inline_data") or part.get("inlineData")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        raise RuntimeError(f"no image in response: "
                           f"{redact(json.dumps(response), secrets)[:300]}")

    for item in response.get("data", []):
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            try:
                with urllib.request.urlopen(item["url"], timeout=timeout) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                raise TransientError(f"failed to download the result image: "
                                     f"{redact(getattr(exc, 'reason', exc), secrets)}") from exc
    raise RuntimeError(f"no image in response: {redact(json.dumps(response), secrets)[:300]}")


def _save_png(data: bytes, dst: Path) -> None:
    """Write provider bytes as a real PNG.

    Decodes with Pillow when it is available (any input format) and re-encodes to PNG;
    without Pillow, only bytes that already are a PNG are accepted, so a JPEG response
    can never be written under a ``.png`` name.
    """
    try:
        from PIL import Image
    except Exception:
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("provider bytes are not a PNG and Pillow is not available")
        dst.write_bytes(data)
        return
    import io as _io

    with Image.open(_io.BytesIO(data)) as im:
        im.convert("RGB").save(dst, "PNG")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="baseline_nanobanana.py",
        description="Hosted image-edit baseline: capture -> flat-lay packshot.",
    )
    ap.add_argument("--manifest", required=True,
                    help="a dataset manifest: train/pairs.jsonl or dev|test/inputs.jsonl")
    ap.add_argument("--out", required=True, help="prediction root (the submission layout is written here)")
    ap.add_argument("--team", required=True, help="team folder name, e.g. hosted-edit")
    ap.add_argument("--dataset", default=None,
                    help="dataset root (folder holding train/, dev/, test/); default: the manifest's root")
    ap.add_argument("--provider", choices=("fal", "gemini", "openai"), default="fal",
                    help="hosted provider (default: fal)")
    ap.add_argument("--model", default=None,
                    help=f"provider model id for image editing; required for gemini and openai, "
                         f"defaults to {DEFAULT_FAL_MODEL} for fal (model names change, so nothing "
                         f"is hardcoded for the other providers)")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--view", default=DEFAULT_VIEW, help=f"view name for rows without a view (default: {DEFAULT_VIEW})")
    ap.add_argument("--limit", type=int, default=None,
                    help="generate at most N images; each row is one image and a retried row is "
                         "still one image, but every retry is a separate request, so total calls "
                         "can exceed N")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout, and for fal the budget for waiting on a queued "
                         "request to finish (default: 300s)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="write the request payloads, make no network call, need no key")
    args = ap.parse_args(argv)

    if args.model is not None and not args.model.strip():
        ap.error("--model must not be empty or whitespace "
                 "(an unset shell/env var expands to an empty string)")
    if not args.model:
        if args.provider == "fal":
            args.model = DEFAULT_FAL_MODEL
        else:
            ap.error("--model is required for --provider gemini/openai "
                     "(model names change, so nothing is hardcoded there)")
    args.model = args.model.strip()

    manifest = Path(args.manifest)
    dataset_root = resolve_dataset_root(manifest, args.dataset)
    try:
        rows = load_rows(manifest, dataset_root)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    out_root = Path(args.out) / args.team
    reqdir = Path(args.out) / "_requests"
    run_manifest = Path(args.out) / "_manifest.jsonl"
    key = "" if args.dry_run else api_key(args.provider)
    secrets = _current_secrets(key)

    ok = failed = 0
    for row in rows:
        src = row["inputs"][0]
        if not src.is_file():
            raise SystemExit(f"error: input image not found: {src}\n"
                             f"hint: pass --dataset <root> if the dataset is not next to the repo")
        view = row["view"] if row["view"] != DEFAULT_VIEW else args.view
        url, headers, body = build_request(args.provider, args.model, args.prompt, src, args.base_url)
        if args.dry_run:
            reqdir.mkdir(parents=True, exist_ok=True)
            target = reqdir / f"{row['id']}_{view}.request"
            if headers.get("Content-Type", "").startswith("application/json"):
                target = target.with_suffix(".json")
            payload = dry_payload(args.provider, body)
            target.write_bytes(payload)
            note = f", image data truncated from {len(body)}" if len(payload) != len(body) else ""
            print(f"[dry-run] {src} -> {url}  "
                  f"(preview payload: {target}, {len(payload)} bytes{note}; nothing sent)")
            continue

        dst = out_root / row["id"] / f"{view}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        error = None
        data = None
        for attempt in range(args.retries + 1):
            error = None  # a successful attempt must clear any earlier failure
            try:
                if args.provider == "fal":
                    data = fal_generate(url, body, key, args.timeout)
                else:
                    hdrs = dict(headers)
                    if args.provider == "gemini":
                        hdrs["x-goog-api-key"] = key
                    else:
                        hdrs["Authorization"] = f"Bearer {key}"
                    resp = _post(url, body, hdrs, args.timeout, secrets)
                    data = extract_image(args.provider, resp, args.timeout, secrets)
                break
            except TransientError as exc:  # 429/5xx, timeout, connection error -> retry
                error = redact(f"{type(exc).__name__}: {exc}", secrets)
                if attempt < args.retries:
                    time.sleep(2.0 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - deterministic failure: do not retry
                error = redact(f"{type(exc).__name__}: {exc}", secrets)
                break
        elapsed = round(time.time() - started, 2)

        entry = {"id": row["id"], "view": view, "input": str(src), "provider": args.provider,
                 "model": args.model, "prompt": args.prompt, "seconds": elapsed, "error": error}
        if data is not None:
            try:
                _save_png(data, dst)
            except Exception as exc:  # noqa: BLE001 - report and continue
                data = None
                error = redact(f"{type(exc).__name__}: {exc}", secrets)
        entry["error"] = error
        if data is None:
            failed += 1
            print(f"  ! {row['id']}/{view}: {error}")
        else:
            ok += 1
            entry["output"] = str(dst)
            print(f"  {row['id']}/{view} -> {dst}  ({elapsed}s)")
        with open(run_manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    if args.dry_run:
        print(f"\n(dry run) {len(rows)} request preview(s) under {reqdir}; "
              f"nothing was sent, no key needed.")
        return 0
    print(f"\n{ok} written, {failed} failed -> {out_root}\nmanifest: {run_manifest}")
    if failed and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
