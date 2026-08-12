"""osu! replay → MP4 renderer."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from osu_mania_renderer_v2.beatmap.models import RenderOptions
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

# Compatibility marker retained for callers that imported it during the old
# A/B period. The canonical compositor is now unconditional; the
# OSU_USE_WIKI_RENDERER environment variable is intentionally ignored.
USE_WIKI_RENDERER = True


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
    # Heavy GL/ffmpeg dependencies remain lazy so importing the package's
    # models and errors is still safe in lightweight environments.
    from osu_mania_renderer_v2.render.compositor import render_mania as _render

    await _render(
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
