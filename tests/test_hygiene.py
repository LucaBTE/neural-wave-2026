"""Guard rails: the participant-facing scripts must not leak machine paths or credentials.

These checks are deliberately generic. The organiser-side scanner owns the
source-specific denylist (internal ids, source names, internal split names); that
list must not ship inside this repo, because the repository is public.
"""

from __future__ import annotations

import re

import _mock

ABSOLUTE_PATHS = [
    (r"/data/", "absolute machine path"),
    (r"/home/", "absolute machine path"),
    (r"/Users/", "absolute machine path"),
    (r"/tmp/", "absolute machine path"),
]

#: Credential-shaped strings. Values are matched generically; no real key ships here.
API_KEYS = [
    r"\bsk-[A-Za-z0-9_-]{16,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bAIza[0-9A-Za-z_-]{35}\b",
    r"\bghp_[A-Za-z0-9]{20,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
]


def test_no_absolute_paths_or_api_keys():
    files = sorted(_mock.BASELINES.glob("*.py")) + sorted(_mock.SCRIPTS.glob("*.py"))
    assert files, "no participant scripts found"
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern, why in ABSOLUTE_PATHS:
            match = re.search(pattern, text)
            assert match is None, f"{path.name}: found {why} {match.group(0)!r}"
        for pattern in API_KEYS:
            match = re.search(pattern, text)
            assert match is None, f"{path.name}: found an API-key-like string {match.group(0)!r}"
