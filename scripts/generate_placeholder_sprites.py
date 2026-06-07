"""Generate functional placeholder sprites so the renderer works before Night05 is staged.

Run once after a fresh clone:  python scripts/generate_placeholder_sprites.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEST = Path(__file__).resolve().parent.parent / "osu_mania_renderer_v2" / "assets" / "sprites"
DEST.mkdir(parents=True, exist_ok=True)


def _rect(name: str, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    img = Image.new("RGBA", size, color)
    img.save(DEST / name)


def _judgment(name: str, text: str, color: tuple[int, int, int]) -> None:
    img = Image.new("RGBA", (256, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 4), text, font=font, fill=(*color, 255))
    img.save(DEST / name)


def main() -> None:
    # Per-column tap notes + receptors: outer (blue), inner (white), center (yellow).
    for variant, color in (
        ("outer",  (180, 220, 255, 255)),
        ("inner",  (235, 235, 235, 255)),
        ("center", (255, 220, 110, 255)),
    ):
        _rect(f"note_tap_{variant}.png",       (128, 32), color)
        _rect(f"note_hold_head_{variant}.png", (128, 32), color)
        body_color = (
            color[0] // 2 + 30, color[1] // 2 + 30, color[2] // 2 + 30, 200,
        )
        _rect(f"note_hold_body_{variant}.png", (128, 32), body_color)
        _rect(f"note_hold_tail_{variant}.png", (128, 32), color)
        _rect(f"receptor_off_{variant}.png",   (128, 64), (60, 60, 80, 255))
        _rect(f"receptor_on_{variant}.png",    (128, 64),
              (color[0], color[1], color[2], 240))

    _rect("stage_left.png",      (32, 1024), (30, 30, 50, 220))
    _rect("stage_right.png",     (32, 1024), (30, 30, 50, 220))
    _rect("stage_light.png",     (128, 32),  (180, 220, 255, 100))
    _rect("hit_light.png",       (256, 128), (255, 255, 255, 180))
    # column_bg is a 1×1 white pixel — tint controls the actual color, so the
    # renderer can use it for both the dim playfield fill and bright border
    # lines without two separate textures.
    _rect("column_bg.png",       (4, 4),     (255, 255, 255, 255))
    _rect("playfield_frame.png", (16, 16),   (120, 120, 140, 220))
    _rect("bg_vignette.png",     (1024, 1024), (0, 0, 0, 0))

    # `note_circle`: anti-aliased filled WHITE circle, 256×256. We render it
    # square so a tint at draw-time controls the actual hue. This is the
    # "osu! web replay viewer" note shape — rounder than Night05's bar notes.
    note_img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    note_draw = ImageDraw.Draw(note_img)
    note_draw.ellipse((4, 4, 251, 251), fill=(255, 255, 255, 255))
    note_img.save(DEST / "note_circle.png")

    # Receptors: hollow circle (off) and filled circle (on / lit). Drawn
    # procedurally so they match the OG web-viewer look, not Night05's
    # bar-style receptors that vanish into the playfield. Both 256×256.
    rec_off = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    rec_off_draw = ImageDraw.Draw(rec_off)
    # Dark transparent fill, thick white border.
    rec_off_draw.ellipse((10, 10, 245, 245),
                         fill=(15, 15, 25, 220),
                         outline=(235, 235, 245, 255), width=6)
    rec_off.save(DEST / "receptor_off_outer.png")
    rec_off.save(DEST / "receptor_off_inner.png")
    rec_off.save(DEST / "receptor_off_center.png")

    rec_on = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    rec_on_draw = ImageDraw.Draw(rec_on)
    # Lit: bright blue fill + matching border.
    rec_on_draw.ellipse((10, 10, 245, 245),
                        fill=(140, 200, 255, 255),
                        outline=(235, 245, 255, 255), width=6)
    rec_on.save(DEST / "receptor_on_outer.png")
    rec_on.save(DEST / "receptor_on_inner.png")
    rec_on.save(DEST / "receptor_on_center.png")

    # `hit_strip`: smooth gradient mapped to osu!mania judgment colours so
    # the bar is also a judgment legend. Centre (0 ms = 320) is light blue,
    # then blue (300) → green (200) → yellow (100) → orange (50) outward.
    strip_w, strip_h = 1024, 8
    strip = Image.new("RGBA", (strip_w, strip_h), (0, 0, 0, 0))
    stops = [
        (0.00, (240, 160,  80)),   # 50 window edge (orange)
        (0.20, (240, 220,  90)),   # 100 (yellow)
        (0.36, (100, 220, 130)),   # 200 (green)
        (0.45, ( 80, 150, 240)),   # 300 (blue)
        (0.50, (150, 215, 255)),   # 320 (light blue / cyan)
        (0.55, ( 80, 150, 240)),
        (0.64, (100, 220, 130)),
        (0.80, (240, 220,  90)),
        (1.00, (240, 160,  80)),
    ]
    for x in range(strip_w):
        t = x / (strip_w - 1)
        # Find bracketing stops and interpolate linearly.
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                u = (t - t0) / max(1e-9, (t1 - t0))
                r = int(c0[0] + (c1[0] - c0[0]) * u)
                g = int(c0[1] + (c1[1] - c0[1]) * u)
                b = int(c0[2] + (c1[2] - c0[2]) * u)
                for y in range(strip_h):
                    strip.putpixel((x, y), (r, g, b, 255))
                break
    strip.save(DEST / "hit_strip.png")

    _judgment("judgment_geki.png", "320",  (255, 215, 0))
    _judgment("judgment_300.png",  "300",  (100, 180, 255))
    _judgment("judgment_katu.png", "200",  (110, 220, 130))
    _judgment("judgment_100.png",  "100",  (255, 230, 90))
    _judgment("judgment_50.png",   " 50",  (180, 180, 180))
    _judgment("judgment_miss.png", "MISS", (240, 80, 80))


if __name__ == "__main__":
    main()
    print(f"Wrote sprites to {DEST}")
