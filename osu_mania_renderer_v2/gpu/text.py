"""Rasterize HUD/banner strings via PIL → upload to a GL texture.

The text is short and re-rendering per-frame is fine at the bot's scale.
For score/combo (changes often), the caller caches the texture and only
re-rasterizes when the string changes.
"""
from __future__ import annotations

from functools import lru_cache

import moderngl
from PIL import Image, ImageDraw, ImageFont


# Font candidates in priority order. Some Fedora/RHEL containers (incl. our
# Bazzite toolbox) ship Liberation but not DejaVu, so falling back through the
# list lets us land on a real scalable TTF instead of PIL's bitmap default
# (which silently ignores the requested size — caused every text bump to
# render as ~10 pt regardless of what was passed).
_FONT_CANDIDATES: tuple[str, ...] = (
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    "NotoSans-Bold.ttf",
    "Arial Bold.ttf",
)


@lru_cache(maxsize=32)
def _font_bold(size: int) -> ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    # Last resort — PIL's bitmap default; size param is ignored here but the
    # earlier candidates almost always hit on Linux + Bazzite/Fedora hosts.
    return ImageFont.load_default()


# Shared measuring surface for text_to_texture's textbbox() call.
_MEASURE_DRAW = ImageDraw.Draw(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))


def text_to_texture(
    ctx: moderngl.Context,
    text: str,
    size: int = 24,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> tuple[moderngl.Texture, int, int]:
    font = _font_bold(size)
    # Measure. textbbox() never touches pixels, so one shared 4x4 measuring
    # surface replaces the per-call Image.new + Draw construction (this runs
    # once per rasterised string — nearly every frame via the score counter).
    bbox = _MEASURE_DRAW.textbbox((0, 0), text, font=font)
    w = max(8, bbox[2] - bbox[0] + 8)
    h = max(8, bbox[3] - bbox[1] + 8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=color)
    raw = img.tobytes()
    tex = ctx.texture((w, h), 4, raw)
    # No mipmaps: every text draw goes through _draw_external_texture, which
    # samples the single-layer wrap ARRAY on `tex.extra` (plain LINEAR, no
    # mip chain) — the old build_mipmaps() on the 2D texture was never
    # sampled and cost ~0.3 ms per rasterised string (the rolling score
    # counter re-rasterises nearly every frame). Building the wrap array
    # here from the same CPU bytes also saves _draw_external_texture's
    # first-draw tex.read() GPU→CPU roundtrip.
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.extra = ctx.texture_array(size=(w, h, 1), components=4, data=raw)
    return tex, w, h
