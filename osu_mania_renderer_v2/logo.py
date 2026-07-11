"""R3D intro splash (show_logo) for the mania v2 renderer.

Ported from the std renderer (osu_std_renderer/render/effects.py +
render/textures.py::bake_logo_tile) via the catch port
(osu_catch_renderer/effects.py + assets.py) so the R3D 'R' logo splash is
IDENTICAL across modes (the V2 porting-guide coherence rule): same fade
envelope, same timing, same asset (assets/logo.png is byte-identical to the
std/catch copy). gpu/renderer.py::FrameRenderer.draw_logo_splash draws from
these pure functions; only the LOGO section is ported (mania has no
bg-triangles / seizure card / etc.).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

# --- intro logo envelope (identical to std/catch) --------------------------
LOGO_FADE_IN_MS = 300.0
LOGO_FADE_OUT_MS = 500.0       # ends exactly as the first note spawns
LOGO_MIN_WINDOW_MS = 700.0     # not enough intro to read the logo -> skip
LOGO_MAX_ALPHA = 0.92
LOGO_UI_SIZE = 220.0           # tile edge in the 1080-space
LOGO_TILE_RED = (216, 44, 54)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def logo_alpha(t: float, t_start: float, gameplay_in: float) -> float | None:
    """The intro splash alpha at map time t, or None when the logo phase is
    inactive. Window = [t_start, gameplay_in] (render start -> the first
    note's spawn): fade in over LOGO_FADE_IN_MS, hold, fade out over
    LOGO_FADE_OUT_MS ENDING at gameplay_in. Windows too short to read
    (< LOGO_MIN_WINDOW_MS) show nothing."""
    if gameplay_in - t_start < LOGO_MIN_WINDOW_MS:
        return None
    if t < t_start or t >= gameplay_in:
        return None
    a_in = _clamp01((t - t_start) / LOGO_FADE_IN_MS)
    a_out = _clamp01((gameplay_in - t) / LOGO_FADE_OUT_MS)
    a = LOGO_MAX_ALPHA * min(a_in, a_out)
    return a if a > 0.0 else None


def logo_scale(t: float, t_start: float) -> float:
    """Gentle settle: 1.06 -> 1.0 over the first 600 ms (quad-out)."""
    p = _clamp01((t - t_start) / 600.0)
    ease = 1.0 - (1.0 - p) * (1.0 - p)
    return 1.06 - 0.06 * ease


def bake_logo_tile(size: int = 256) -> np.ndarray:
    """RGBA tile for the intro splash. Prefers assets/logo.png (the real R3D
    logo — the SAME file the std/catch splashes load, so the splash is
    identical across modes); procedural fallback (rounded red tile + white R)
    only if the asset is missing."""
    try:
        lp = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        im = Image.open(lp).convert("RGBA").resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8).copy()
    except Exception:
        pass
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)
    drw.rounded_rectangle([0, 0, size - 1, size - 1],
                          radius=int(size * 0.18), fill=LOGO_TILE_RED + (255,))
    try:
        from PIL import ImageFont
        try:
            f = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size * 0.66))
        except Exception:
            f = ImageFont.load_default()
        box = f.getbbox("R")
        rw, rh = box[2] - box[0], box[3] - box[1]
        drw.text(((size - rw) / 2.0 - box[0], (size - rh) / 2.0 - box[1]),
                 "R", font=f, fill=(255, 255, 255, 255))
    except Exception:
        pass
    return np.asarray(img, dtype=np.uint8).copy()


def logo_glow_rgba(size: int = 128) -> np.ndarray:
    """Soft white radial glow (alpha falloff a = (1-d)^3), tinted/additive at
    draw time — the same tile catch bakes as `catch_glow` / std as `glow` for
    the red halo behind the splash."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    d = np.hypot(xx - c, yy - c) / c
    a = np.clip(1.0 - d, 0.0, 1.0) ** 3
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[..., 0] = 255
    img[..., 1] = 255
    img[..., 2] = 255
    img[..., 3] = (255 * a).astype(np.uint8)
    return img
