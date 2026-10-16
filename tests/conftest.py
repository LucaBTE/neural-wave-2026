"""Fixtures for the repo-level script tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _mock  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "scorer: needs the scorer model weights (set SCORER_MODEL_DIR; skip with SARTIQ_SKIP_MODEL_TESTS=1)",
    )


@pytest.fixture()
def dataset(tmp_path: Path) -> dict:
    """A throwaway participant dataset with the exact contract layout."""
    return _mock.build_mock_dataset(tmp_path / "sartiq_neural_wave_2026")


@pytest.fixture()
def dataset_shared(tmp_path: Path) -> dict:
    """Same, but two pairs share a byte-identical target (repeated product)."""
    return _mock.build_mock_dataset(tmp_path / "sartiq_neural_wave_2026", shared_targets=1)
