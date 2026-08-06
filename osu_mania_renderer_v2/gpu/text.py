"""Rasterize HUD/banner strings via PIL → upload to a GL texture.

The text is short and re-rendering per-frame is fine at the bot's scale.
For score/combo (changes often), the caller caches the texture and only
re-rasterizes when the string changes.
"""
from __future__ import annotations

import os
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


# Bundled OFL Nunito — the lazer-Torus stand-in std/catch/taiko all use, so
# mania's PIL text (labels, key counter, watermark, banner, results labels)
# matches the siblings instead of a machine-dependent system font. Pinned to
# wght 700 (mania's _font_bold was DejaVu-Bold, so stay bold). The DejaVu→Nunito
# cap-fill scale keeps the VISIBLE cap-height unchanged for the existing
# DejaVu-tuned sizes (std/catch: 0.7154 / 0.7025 ≈ 1.0184).
_NUNITO_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "fonts", "Nunito[wght].ttf"))
_NUNITO_WEIGHT = 700
_DEJAVU_NUNITO_CAP_SCALE = 0.7154 / 0.7025


@lru_cache(maxsize=32)
def _font_bold(size: int) -> ImageFont.ImageFont:
    try:
        f = ImageFont.truetype(
            _NUNITO_PATH, max(6, int(round(size * _DEJAVU_NUNITO_CAP_SCALE))))
        try:
            f.set_variation_by_axes([_NUNITO_WEIGHT])
        except Exception:  # noqa: BLE001 — non-variable PIL build → default face
            pass
        return f
    except OSError:
        pass
    # Fallback: system fonts (stripped checkout / no bundled Nunito).
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


def pill_to_texture(
    ctx: moderngl.Context, w: int, h: int,
) -> tuple[moderngl.Texture, int, int]:
    """White capsule (rounded rect, corner radius = h/2) baked at 4×
    supersample for smooth edges. Used by the Argon key counter's
    indicator (lazer draws it as a CircularContainer line; STD/catch use
    their 'pill' sprite — std hud.py:2226-2229). Tint/alpha are applied
    at draw time via _draw_external_texture."""
    w, h = max(2, int(w)), max(2, int(h))
    ss = 4
    mask = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1), radius=(h * ss) // 2, fill=255)
    mask = mask.resize((w, h), Image.LANCZOS)
    white = Image.new("L", (w, h), 255)
    rgba = Image.merge("RGBA", (white, white, white, mask))
    raw = rgba.tobytes()
    tex = ctx.texture((w, h), 4, raw)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.extra = ctx.texture_array(size=(w, h, 1), components=4, data=raw)
    return tex, w, h


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
