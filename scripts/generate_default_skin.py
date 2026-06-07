"""Generate the bundled DEFAULT mania skin sprites — a clean-room,
osu!-faithful default (gradient rounded note bars with a bright top edge,
key-style receptors, tileable hold bodies). Replaces the flat placeholder
sprites so "no user skin" renders look like real osu!mania instead of thin
faint bars.

Clean-room: no osu! assets are copied; everything is drawn here. Run after a
fresh clone or when tuning the default look:

    python scripts/generate_default_skin.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEST = Path(__file__).resolve().parent.parent / "osu_mania_renderer_v2" / "assets" / "sprites"
DEST.mkdir(parents=True, exist_ok=True)

# Per-column variant base colours (osu!stable-ish palette): outer lanes
# white/cyan, inner lanes osu! blue, centre lane gold.
VARIANTS = {
    "outer":  (225, 238, 252),
    "inner":  (72, 168, 255),
    "center": (255, 206, 92),
}


def _brighten(c, f):
    return tuple(min(255, int(ci + (255 - ci) * f)) for ci in c)


def _darken(c, f):
    return tuple(max(0, int(ci * (1.0 - f))) for ci in c)


def _rounded_mask(w, h, rad):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=rad, fill=255)
    return m


def _vgrad(w, h, top, bot):
    g = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(g)
    for y in range(h):
        t = y / max(1, h - 1)
        c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c)
    return g


def _note_bar(base, w=160, h=56, rad=14):
    """Rounded gradient bar: lighter top → base bottom, bright top highlight,
    darker outline. Reads as a glossy osu! note."""
    top = _brighten(base, 0.55)
    bot = _darken(base, 0.10)
    body = _vgrad(w, h, top, bot).convert("RGBA")
    mask = _rounded_mask(w, h, rad)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.paste(body, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # Bright top highlight band (top ~28%).
    hl_h = int(h * 0.28)
    hl = Image.new("RGBA", (w, hl_h), (0, 0, 0, 0))
    hg = _vgrad(w, hl_h, _brighten(base, 0.85), top).convert("RGBA")
    hl.paste(hg, (0, 0), _rounded_mask(w, hl_h, rad))
    img.alpha_composite(hl, (0, 2))
    # Outline.
    d.rounded_rectangle((1, 1, w - 2, h - 2), radius=rad,
                        outline=(*_darken(base, 0.45), 255), width=3)
    # Re-apply rounded mask so corners stay clean.
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def _hold_body(base, w=160, h=48):
    """Tileable hold body: vertical gradient column, slightly translucent,
    no rounded ends (the head/tail cap it)."""
    top = _brighten(base, 0.25)
    bot = _darken(base, 0.25)
    body = _vgrad(w, h, top, bot).convert("RGBA")
    a = Image.new("L", (w, h), 220)
    body.putalpha(a)
    d = ImageDraw.Draw(body)
    # Side edges a touch brighter for a tube look.
    edge = (*_brighten(base, 0.6), 230)
    d.line([(2, 0), (2, h)], fill=edge, width=2)
    d.line([(w - 3, 0), (w - 3, h)], fill=edge, width=2)
    return body


def _receptor(base, lit, w=160, h=84, rad=16):
    """Key-style receptor: rounded rect at the bottom of the column.
    off = dark translucent with bright rim; on = lit fill + inner glow."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if lit:
        top = _brighten(base, 0.5)
        bot = base
        body = _vgrad(w, h, top, bot).convert("RGBA")
        mask = _rounded_mask(w, h, rad)
        img.paste(body, (0, 0), mask)
        d.rounded_rectangle((1, 1, w - 2, h - 2), radius=rad,
                            outline=(255, 255, 255, 255), width=4)
    else:
        body = Image.new("RGBA", (w, h), (18, 20, 30, 180))
        mask = _rounded_mask(w, h, rad)
        img.paste(body, (0, 0), mask)
        d.rounded_rectangle((2, 2, w - 3, h - 3), radius=rad,
                            outline=(200, 210, 235, 235), width=4)
        # subtle inner top sheen
        d.rounded_rectangle((10, 8, w - 11, h // 2), radius=rad // 2,
                            outline=(120, 130, 160, 90), width=2)
    return img


def _judgment(name: str, text: str, color) -> None:
    img = Image.new("RGBA", (256, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 4), text, font=font, fill=(*color, 255))
    img.save(DEST / name)


def _argon_sprites() -> None:
    """White Argon shapes, tinted per-column accent at draw time, baked to
    match osu!lazer's Argon mania (notes = accent bar + white chevron + white
    bottom line; column bottom-glow; receptor pill + 3 dots)."""
    W, H = 220, 132          # note body/glyph canvas (aspect ~1.67)
    rad = 10

    # argon_note_body: white rounded bar, faint vertical gradient (lighter
    # top → ~0.78 bottom) so an accent tint reproduces Argon's lighten-top.
    body = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    grad = _vgrad(W, H, (255, 255, 255), (200, 200, 200)).convert("RGBA")
    body.paste(grad, (0, 0), _rounded_mask(W, H, rad))
    body.save(DEST / "argon_note_body.png")

    # argon_note_glyph: white chevron (downward V) upper-centre + a white
    # bright bar along the bottom edge (Argon's hit-line on the note).
    glyph = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glyph)
    cx = W // 2
    cyc = int(H * 0.42)
    arm = int(W * 0.16)
    drop = int(H * 0.16)
    lw = max(6, H // 14)
    gd.line([(cx - arm, cyc - drop), (cx, cyc + drop)], fill=(255, 255, 255, 255), width=lw)
    gd.line([(cx, cyc + drop), (cx + arm, cyc - drop)], fill=(255, 255, 255, 255), width=lw)
    bar_h = max(6, H // 16)
    gd.rounded_rectangle((6, H - bar_h - 3, W - 7, H - 3), radius=bar_h // 2,
                         fill=(255, 255, 255, 255))
    glyph.save(DEST / "argon_note_glyph.png")

    # argon_col_glow: white vertical gradient, transparent top → ~150α bottom.
    gw, gh = 64, 256
    glow = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    gdd = ImageDraw.Draw(glow)
    for y in range(gh):
        a = int(150 * (y / (gh - 1)) ** 1.6)   # bottom-heavy
        gdd.line([(0, y), (gw, y)], fill=(255, 255, 255, a))
    glow.save(DEST / "argon_col_glow.png")

    # argon_key_pill: white rounded-rect outline ("0"/pill icon).
    pw, ph = 132, 76
    pill = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle(
        (8, 8, pw - 9, ph - 9), radius=(ph - 16) // 2,
        outline=(255, 255, 255, 255), width=8)
    pill.save(DEST / "argon_key_pill.png")

    # argon_key_dots: 3 white dots (triangle: two bottom, one top).
    dw, dh = 132, 72
    dots = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    r = 9
    for cx2, cy2 in ((dw // 2, 16), (dw // 2 - 26, dh - 18), (dw // 2 + 26, dh - 18)):
        dd.ellipse((cx2 - r, cy2 - r, cx2 + r, cy2 + r), fill=(255, 255, 255, 255))
    dots.save(DEST / "argon_key_dots.png")


def main() -> None:
    _argon_sprites()
    for variant, base in VARIANTS.items():
        _note_bar(base).save(DEST / f"note_tap_{variant}.png")
        _note_bar(base).save(DEST / f"note_hold_head_{variant}.png")
        _note_bar(base).save(DEST / f"note_hold_tail_{variant}.png")
        _hold_body(base).save(DEST / f"note_hold_body_{variant}.png")
        _receptor(base, lit=False).save(DEST / f"receptor_off_{variant}.png")
        _receptor(base, lit=True).save(DEST / f"receptor_on_{variant}.png")

    # Stage chrome stays minimal/neutral (let the renderer's column dims +
    # the playfield carry the look); 1×1-ish so they don't impose art.
    Image.new("RGBA", (32, 1024), (26, 28, 42, 210)).save(DEST / "stage_left.png")
    Image.new("RGBA", (32, 1024), (26, 28, 42, 210)).save(DEST / "stage_right.png")
    Image.new("RGBA", (128, 32), (180, 220, 255, 90)).save(DEST / "stage_light.png")
    Image.new("RGBA", (256, 128), (255, 255, 255, 180)).save(DEST / "hit_light.png")
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(DEST / "column_bg.png")
    Image.new("RGBA", (16, 16), (120, 120, 140, 0)).save(DEST / "playfield_frame.png")
    Image.new("RGBA", (1024, 1024), (0, 0, 0, 0)).save(DEST / "bg_vignette.png")

    # note_circle fallback (kept for code paths that use it).
    note_img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ImageDraw.Draw(note_img).ellipse((4, 4, 251, 251), fill=(255, 255, 255, 255))
    note_img.save(DEST / "note_circle.png")

    # hit_strip judgment-colour gradient (unchanged from placeholder).
    strip_w, strip_h = 1024, 8
    strip = Image.new("RGBA", (strip_w, strip_h), (0, 0, 0, 0))
    stops = [
        (0.00, (240, 160, 80)), (0.20, (240, 220, 90)), (0.36, (100, 220, 130)),
        (0.45, (80, 150, 240)), (0.50, (150, 215, 255)), (0.55, (80, 150, 240)),
        (0.64, (100, 220, 130)), (0.80, (240, 220, 90)), (1.00, (240, 160, 80)),
    ]
    sd = ImageDraw.Draw(strip)
    for x in range(strip_w):
        t = x / (strip_w - 1)
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                u = (t - t0) / max(1e-9, (t1 - t0))
                c = tuple(int(c0[k] + (c1[k] - c0[k]) * u) for k in range(3))
                sd.line([(x, 0), (x, strip_h)], fill=(*c, 255))
                break
    strip.save(DEST / "hit_strip.png")

    _judgment("judgment_geki.png", "320", (255, 215, 0))
    _judgment("judgment_300.png", "300", (100, 180, 255))
    _judgment("judgment_katu.png", "200", (110, 220, 130))
    _judgment("judgment_100.png", "100", (255, 230, 90))
    _judgment("judgment_50.png", " 50", (180, 180, 180))
    _judgment("judgment_miss.png", "MISS", (240, 80, 80))


if __name__ == "__main__":
    main()
    print(f"Wrote default skin sprites to {DEST}")
