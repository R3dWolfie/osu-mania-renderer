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


@pytest.fixture
def compose_scene(tmp_path: Path):
    """Draw a SceneState through the canonical element registry."""
    user_skin = tmp_path / "empty-user-skin"
    default_skin = tmp_path / "empty-default-skin"
    user_skin.mkdir()
    default_skin.mkdir()

    def _compose(fr, gl, scene):
        import osu_mania_renderer_v2.render.pipeline  # noqa: F401
        from osu_mania_renderer_v2.render.compositor import (
            compose_frame,
            resolve_elements,
        )
        from osu_mania_renderer_v2.render.frame_context import FrameContext
        from osu_mania_renderer_v2.skin.mania_skin import SkinPair

        skin = SkinPair(user_dir=user_skin, default_dir=default_skin)
        ctx = FrameContext(
            fr=fr,
            skin=skin,
            gl=gl.ctx,
            fbo=gl.fbo,
            width=fr.rc.width,
            height=fr.rc.height,
            key_count=fr.rc.key_count,
            scene=scene,
            t_ms=scene.t_ms,
        )
        compose_frame(ctx, resolve_elements(skin, fr.rc.key_count))
        return ctx

    return _compose
