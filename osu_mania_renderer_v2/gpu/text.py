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


def text_to_texture(
    ctx: moderngl.Context,
    text: str,
    size: int = 24,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> tuple[moderngl.Texture, int, int]:
    font = _font_bold(size)
    # Measure
    dummy = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w = max(8, bbox[2] - bbox[0] + 8)
    h = max(8, bbox[3] - bbox[1] + 8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=color)
    tex = ctx.texture((w, h), 4, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    return tex, w, h
