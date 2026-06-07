"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def assets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "osu_mania_renderer_v2" / "assets"
