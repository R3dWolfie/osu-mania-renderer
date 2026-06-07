"""osu! replay → MP4 renderer."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from osu_mania_renderer_v2.errors import (
    BeatmapParseError,
    EncoderError,
    GpuUnavailableError,
    MissingAudioError,
    NotAManiaError,
    RendererError,
    RenderTimeoutError,
    ReplayParseError,
)
from osu_mania_renderer_v2.models import RenderOptions
# GPU renderer imported lazily (its dependency chain pulls in moderngl,
# which may not be available in every Python environment — e.g. CI,
# headless wiki-renderer-only builds).

async def _gpu_render_mania(**kwargs):  # type: ignore
    """Lazy import of the GPU render path."""
    from osu_mania_renderer_v2.render import render_mania as _inner
    return await _inner(**kwargs)

log = logging.getLogger("osu_mania_renderer_v2")

# ----- wiki renderer switch ------------------------------------------------
# Set OSU_USE_WIKI_RENDERER=1 to route renders through the wiki-driven
# pipeline instead of the hand-coded GPU renderer. This is a temporary A/B
# toggle while the wiki pipeline is being built.
USE_WIKI_RENDERER = os.environ.get(
    "OSU_USE_WIKI_RENDERER", "",
).lower() in ("1", "true", "yes")


async def render_mania(
    *,
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    options: RenderOptions,
    progress_callback: Callable[[float], Awaitable[None]] | None = None,
    log_path: Path | None = None,
    skin_dir: Path | None = None,
    allow_converted: bool = False,
    convert_to_keys: int = 4,
) -> None:
    if USE_WIKI_RENDERER:
        import osu_mania_renderer_v2.wiki_elements  # noqa: F401 — populate registries

        from osu_mania_renderer_v2.wiki_renderer import render as _wiki_render

        if skin_dir is None:
            raise RendererError(
                "wiki renderer requires --skin-dir "
                "(set OSU_USE_WIKI_RENDERER=0 to use the GPU pipeline)"
            )

        # Default skin is the variable-tier fallback only; sprite images come
        # from the atlas bundle. The dir may not exist — SkinPair tolerates it
        # and every VariableSpec carries an explicit wiki_default.
        here = Path(__file__).resolve().parent
        default_skin = here / "assets" / "default_skin"

        await _wiki_render(
            osr_path=osr_path,
            beatmap_dir=beatmap_dir,
            output_path=output_path,
            options=options,
            skin_dir=skin_dir,
            default_skin_dir=default_skin,
            progress_callback=progress_callback,
            allow_converted=allow_converted,
            convert_to_keys=convert_to_keys,
        )
        return

    await _gpu_render_mania(
        osr_path=osr_path,
        beatmap_dir=beatmap_dir,
        output_path=output_path,
        options=options,
        progress_callback=progress_callback,
        log_path=log_path,
        skin_dir=skin_dir,
        allow_converted=allow_converted,
        convert_to_keys=convert_to_keys,
    )


__all__ = [
    "RenderOptions",
    "render_mania",
    "RendererError",
    "BeatmapParseError",
    "ReplayParseError",
    "NotAManiaError",
    "MissingAudioError",
    "GpuUnavailableError",
    "EncoderError",
    "RenderTimeoutError",
    "USE_WIKI_RENDERER",
]
