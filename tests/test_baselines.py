"""End-to-end tests for the two baselines on the contract dataset layout."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request

import _mock
from PIL import Image

sys.path.insert(0, str(_mock.BASELINES))

import baseline_nanobanana  # noqa: E402  (needs baselines/ on the path)


def test_identity_on_train_pairs(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_identity.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "identity",
    ])
    assert proc.returncode == 0, proc.stderr
    assert (out / "identity" / "t0001" / "front.jpg").is_file()
    assert sorted(p.name for p in (out / "identity").iterdir()) == \
        sorted(row["pair_id"] for row in info["pairs"])


def test_identity_on_dev_inputs(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_identity.py",
        "--manifest", info["root"] / "dev" / "inputs.jsonl",
        "--out", out, "--team", "identity",
    ])
    assert proc.returncode == 0, proc.stderr
    assert (out / "identity" / "d001" / "front.jpg").is_file()
    assert sorted(p.name for p in (out / "identity").iterdir()) == sorted(info["dev_ids"])


def test_identity_dry_run_writes_nothing(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_identity.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "identity", "--dry-run",
    ])
    assert proc.returncode == 0, proc.stderr
    assert not (out / "identity").exists()
    assert "dry run" in proc.stdout


def test_nanobanana_dry_run_makes_no_call_and_leaks_no_key(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    secret = "SECRET-do-not-log-1234"
    proc = _mock.run_cmd([
        "baselines/baseline_nanobanana.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "nb",
        "--provider", "gemini", "--model", "test-model", "--dry-run",
    ], env_extra={"GEMINI_API_KEY": secret, "GOOGLE_API_KEY": secret})
    assert proc.returncode == 0, proc.stderr
    requests = list((out / "_requests").glob("*.json"))
    assert len(requests) == len(info["pairs"])
    assert not any((out / "nb").rglob("*.png"))
    assert secret not in proc.stdout and secret not in proc.stderr


def test_nanobanana_requires_key_without_dry_run(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_nanobanana.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "nb",
        "--provider", "gemini", "--model", "test-model",
    ], env_extra={"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "IMAGE_API_KEY": "",
                  "OPENAI_API_KEY": ""})
    assert proc.returncode != 0
    assert "API key" in proc.stderr


# --------------------------------------------------------------------------- #
# fal provider (default): public script, fal queue REST via urllib only
# --------------------------------------------------------------------------- #

FAL_STATUS = "https://queue.fal.run/google/nano-banana-2.1/edit/requests/req-1/status"
FAL_RESULT = "https://queue.fal.run/google/nano-banana-2.1/edit/requests/req-1"
FAL_IMAGE = "https://fal-cdn.example/out-1.jpg"


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, *args):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_nanobanana_fal_submit_poll_result_png(dataset, tmp_path, monkeypatch):
    """A mocked fal queue run: submit -> poll -> result -> download -> real PNG on disk."""
    info = dataset
    out = tmp_path / "preds"

    as_png = io.BytesIO()
    Image.new("RGB", (256, 320), (30, 130, 200)).save(as_png, "PNG")
    png_bytes = as_png.getvalue()

    calls = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        method = req.get_method() if isinstance(req, urllib.request.Request) else "GET"
        body = req.data if isinstance(req, urllib.request.Request) else None
        calls.append({"method": method, "url": url, "body": body})
        if method == "POST" and "google/nano-banana-2.1/edit" in url:
            return _FakeResponse(json.dumps(
                {"request_id": "req-1", "status_url": FAL_STATUS, "response_url": FAL_RESULT}
            ).encode())
        if url == FAL_STATUS:
            seen = sum(1 for c in calls if c["url"] == FAL_STATUS)
            return _FakeResponse(json.dumps(
                {"status": "IN_QUEUE" if seen == 1 else "COMPLETED"}).encode())
        if url == FAL_RESULT:
            return _FakeResponse(json.dumps({"images": [{"url": FAL_IMAGE}]}).encode())
        if url == FAL_IMAGE:
            return _FakeResponse(png_bytes)
        raise AssertionError(f"unexpected request: {method} {url}")

    secret = "SECRET-fal-key-do-not-log"
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setenv("FAL_KEY", secret)
    monkeypatch.delenv("IMAGE_API_KEY", raising=False)

    rc = baseline_nanobanana.main([
        "--manifest", str(info["root"] / "train" / "pairs.jsonl"),
        "--out", str(out), "--team", "nb", "--limit", "1",
    ])
    assert rc == 0

    dst = out / "nb" / "t0001" / "front.png"
    assert dst.is_file()
    assert dst.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "output must be a real PNG"

    urls = [c["url"] for c in calls]
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("google/nano-banana-2.1/edit")
    # submit -> poll -> result -> download, in that order
    assert urls.index(FAL_STATUS) < urls.index(FAL_RESULT) < urls.index(FAL_IMAGE)

    payload = json.loads(calls[0]["body"].decode("utf-8"))
    assert payload["prompt"]
    assert payload["image_urls"][0].startswith("data:image/jpeg;base64,")

    manifest = (out / "_manifest.jsonl").read_text()
    assert secret not in manifest


def test_nanobanana_fal_dry_run_needs_no_key_and_truncates(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_nanobanana.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "nb", "--limit", "1", "--dry-run",
    ], env_extra={"FAL_KEY": "", "IMAGE_API_KEY": ""})
    assert proc.returncode == 0, proc.stderr
    reqs = sorted((out / "_requests").glob("*.json"))
    assert len(reqs) == 1
    payload = json.loads(reqs[0].read_text())
    assert payload["prompt"]
    assert payload["image_urls"][0].startswith("data:image/jpeg;base64,")
    assert "truncated" in payload["image_urls"][0], "the data URI must be truncated on disk"
    assert not any((out / "nb").rglob("*.png"))


def test_nanobanana_fal_missing_key_message(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_nanobanana.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "nb",
    ], env_extra={"FAL_KEY": "", "IMAGE_API_KEY": ""})
    assert proc.returncode != 0
    assert "FAL_KEY" in proc.stderr
    assert "API key" in proc.stderr


def test_nanobanana_gemini_still_needs_model(dataset, tmp_path):
    info = dataset
    out = tmp_path / "preds"
    proc = _mock.run_cmd([
        "baselines/baseline_nanobanana.py",
        "--manifest", info["root"] / "train" / "pairs.jsonl",
        "--out", out, "--team", "nb",
        "--provider", "gemini", "--dry-run",
    ])
    assert proc.returncode != 0
    assert "model is required" in proc.stderr


def test_nanobanana_redacts_provider_error_body(dataset, tmp_path, monkeypatch, capsys):
    """A provider error body that echoes the credential must not leak it anywhere."""
    info = dataset
    out = tmp_path / "preds"
    secret = "SECRET-fal-echo-1234567890"

    def boom(req, timeout=None):
        body = json.dumps({"detail": f"invalid key {secret}", "api_key": secret,
                           "token": secret}).encode()
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {},  # type: ignore[arg-type]
                                     io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setenv("FAL_KEY", secret)
    monkeypatch.delenv("IMAGE_API_KEY", raising=False)

    rc = baseline_nanobanana.main([
        "--manifest", str(info["root"] / "train" / "pairs.jsonl"),
        "--out", str(out), "--team", "nb", "--limit", "1",
    ])
    assert rc == 1  # every request failed
    captured = capsys.readouterr()
    manifest = (out / "_manifest.jsonl").read_text(encoding="utf-8")
    for blob in (captured.out, captured.err, manifest):
        assert secret not in blob, blob
    assert "HTTP 401" in manifest
    assert "detail" in manifest


def test_nanobanana_empty_model_is_rejected(dataset, tmp_path):
    """An empty or whitespace --model (e.g. an unset env var) is rejected for every provider."""
    info = dataset
    out = tmp_path / "preds"
    for provider, model in (("fal", ""), ("fal", "   "), ("gemini", ""), ("openai", "  ")):
        proc = _mock.run_cmd([
            "baselines/baseline_nanobanana.py",
            "--manifest", info["root"] / "train" / "pairs.jsonl",
            "--out", out, "--team", "nb",
            "--provider", provider, "--model", model, "--dry-run",
        ], env_extra={"FAL_KEY": "", "IMAGE_API_KEY": ""})
        assert proc.returncode != 0, (provider, repr(model), proc.stdout, proc.stderr)
        assert "model" in proc.stderr, (provider, repr(model), proc.stderr)
    assert not (out / "_requests").exists()
