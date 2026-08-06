"""LAZER RESULTS SCREEN for osu!mania — the osu!(lazer) ranking screen,
ported 1:1 from the osu!catch renderer's hud/lazer_results.py (which ports
the osu!std module, itself a port of osu.Game/Screens/Ranking/* — MIT,
ppy/osu, no ppy assets; every texture is procedurally baked; font = the same
bundled Nunito, the lazer-Torus stand-in). Catch's Fruit/Drop/Droplet
judgments and catch-only widgets (bananas, hyperdash) are replaced by
mania's six judgment tiers, exactly as the taiko renderer's port replaced
them with taiko's (taiko hud/lazer_results.py — the per-mode adaptation
model).

WHAT IS PORTED 1:1 (same constants, same geometry, same draw code —
osu_catch_renderer/hud/lazer_results.py:86-224 constants/helpers,
:374-821 procedural bakes, :933-1583 the screen class):
  * the AccuracyCircle — the achieved-accuracy arc with the fixed vertical
    cyan->green gradient (#7CF6FF->#BAFFA9) + sweeping tip; the dim
    background ring; the thin inner GradedCircles rank ring (the
    OsuColour.ForRank bands) with the six RankBadge pills at their
    Interpolation.Lerp positions; the white centre rank letter with the
    rank-coloured glow that punches in at the end of the sweep.
  * the rounded featured panel (560x940 virtual): avatar + name, map title +
    artist, the rolling score (Nunito Light), the played-mods badge row, the
    star-rating pill (ForStarDifficulty colour + procedural star) + diff
    name + "mapped by", the ACCURACY / MAX COMBO / PP row, the judgment
    rows, the "Played on <date>" footer.
  * the two-stage timeline: panel fade (320ms), arc sweep (260ms delay,
    1150ms OutCubic), grade punch (340ms, 1.42->1.0), score roll, the flank
    cards' staggered slide-in; then the featured panel slides LEFT while
    three stats panels unfold from the right (STAGE1_MS=2000, OPEN_MS=900
    OutQuint, STAGGER_MS=160).

MANIA ADAPTATIONS (owner spec, honest + documented — the same deltas the
taiko port made, mania data):
  * rank/ring cutoffs are MANIA's (osu! wiki Gameplay/Grade, the exact
    boundaries render._compute_grade_from_replay uses: S>=95%, A>=90,
    B>=80, C>=70, D below; SS = an all-perfect play). In lazer the
    AccuracyCircle reads the ruleset's cutoffs, so the ring bands + badge
    Lerp positions derive from THESE.
  * judgments are mania's SIX tiers: 320 (geki/rainbow) / 300 / 200 (katu) /
    100 / 50 / MISS, split over two grid rows (320|300|200 then
    100|50|MISS) exactly like std's multi-row judgment grid
    (osu_std_renderer/render/lazer_results.py:1740-1751); coloured with the
    engine's established judgment palette (gpu/renderer.py jcolours /
    _draw_ur_histogram bands) so the whole screen reads as one UI.
  * accuracy PREFERS replay.accuracy (the percentage the rest of the mania
    pipeline shows — parsed with the display geki weight, replay.py) so the
    figure matches the HUD exactly; the grade letter PREFERS replay.grade.
    Both fall back to the mania display formula when the replay carries
    none.
  * score = the plan's standardised (ScoreV3) total — plan.score_final, the
    exact number the gameplay HUD lands on — NOT a re-derivation.
  * stars / pp / max-pp come from the RenderPlan (compute_star_rating /
    compute_pp, --sr/--pp overrides already applied there) — no rosu calls
    in this module's hot path.
  * the player name comes from replay.player_name (the .osr header), NOT
    parsed off the banner string (the old card split the banner on a
    triple-space — fragile; removed with the old card).
  * stage-2 panels: PERFORMANCE (accuracy / combo-vs-map-max / pp bars),
    COMBO (the judgment timeline's combo-over-time area chart with red
    break ticks, falling back to the rosu mania strain curve, then to
    "timeline unavailable"), and JUDGEMENTS (all six tiers as
    share-of-judged bars, with the play's real UR + avg offset in the
    footer — the old card's headline timing stats, kept).
  * the scene under the screen goes to BLACK (the results-fade opacity
    drives a full black wash, matching std/catch/taiko where the gameplay
    has already faded to black by results_start).
  * flank leaderboard cards: the catch draw code (board slide-in timing,
    stage-2 tracking) is ported VERBATIM; the board itself is mania's
    hud/lb_cards.py (build_mania_board — the ported catch/std render-DB +
    osu!-global board), built + baked once up front by render.py /
    compositor.py and handed in via gpu/renderer.py set_results_data.
    board=None (solo render / leaderboard off) → nothing flanks the panel,
    unchanged.
  * GL delivery: mania composites frames on the GPU (no CPU rgb array like
    catch/taiko), so render_overlay() returns a full-frame RGBA layer
    (black wash + screen, straight alpha) plus a pose token;
    gpu/renderer.py uploads it as a texture and draws one fullscreen quad —
    the same OVER composite catch performs CPU-side. The pose-signature
    memo (catch :1307-1337) doubles as the upload-skip token, so the
    settled tail of the outro re-uploads nothing.

Everything is laid out in the same 1080-height virtual space as std/catch
(k = H / 1080). Static elements bake once; the accuracy arc and the rolling
score re-bake only while they change; once the animation settles the full
frame is cached and reused for the hold.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from osu_mania_renderer_v2.gpu.text import _font_bold as _base_font

UH = 1080.0                    # virtual design height (std convention)

# --- timeline (ms from results start) — catch lazer_results.py:89-104 ---------------
FADE_MS = 320.0                # panel opacity ramp
SWEEP_DELAY_MS = 260.0         # accuracy arc sweep start
SWEEP_MS = 1150.0              # AccuracyCircle ACCURACY_TRANSFORM_DURATION (scaled)
BADGE_MS = 340.0               # rank badge (grade letter) punch duration
LB_SLIDE_START_MS = 300.0      # flank-card entrance begins
LB_SLIDE_STAGGER_MS = 80.0     # per-rank stagger (outer cards arrive later)
LB_SLIDE_MS = 460.0            # per-card slide-in duration (OutQuint)
LB_SLIDE_OFFSET = 96.0         # virtual px each card travels inward
# stage 2 (std lazer_results values): the featured panel slides LEFT while
# the stats panels unfold from the right, staggered
STAGE1_MS = 2000.0             # panel settles + holds until here, then opens
OPEN_MS = 900.0                # panel slide + stats unfold (OutQuint)
STAGGER_MS = 160.0             # per-stats-panel unfold stagger
SETTLE_MS = 3800.0             # everything at rest → the frame is cached
                               # (std MIN_TOTAL_MS; last unfold ends ~3340)

# featured panel geometry (virtual px) — catch lazer_results.py:106-111
PANEL_W = 560.0
PANEL_H = 940.0
ACC_DISP = 380.0               # accuracy-circle canvas display size

VIRTUAL_SS_PERCENTAGE = 0.01   # AccuracyCircle: the reserved SS notch

# rank accuracy thresholds — MANIA's cutoffs (osu! wiki Gameplay/Grade; the
# exact boundaries render._compute_grade_from_replay / the live break-overlay
# grade use: purely accuracy-based, no miss requirement). In lazer the
# AccuracyCircle reads the ruleset's cutoffs, so the ring bands + badge
# positions derive from THESE (catch lazer_results.py:113-124 shape).
RANK_THRESHOLDS = [
    (0.00, "D"),
    (0.70, "C"),
    (0.80, "B"),
    (0.90, "A"),
    (0.95, "S"),
    (1.00, "SS"),
]


def _hex(s: str) -> tuple[float, float, float]:
    s = s.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0)


# lazer's OsuColour.ForRank — the EXACT hexes std's port uses
# (osu.Game/Graphics/OsuColour.cs ForRank) — catch lazer_results.py:133-147.
FOR_RANK = {
    "D": _hex("ff5a5a"),
    "C": _hex("ff8e5d"),
    "B": _hex("e3b130"),
    "A": _hex("88da20"),
    "S": _hex("02b5c3"),
    "SS": _hex("de31ae"),
    "F": _hex("ff5a5a"),
}
FOR_RANK["X"] = FOR_RANK["SS"]
FOR_RANK["SSH"] = FOR_RANK["SS"]
FOR_RANK["XH"] = FOR_RANK["SS"]
FOR_RANK["SH"] = FOR_RANK["S"]

GRADE_SPACING_PERCENTAGE = 2.0 / 360.0

# the achieved-accuracy arc's FIXED vertical gradient (NOT rank-coloured)
ARC_GRAD_TOP = _hex("7CF6FF")
ARC_GRAD_BOT = _hex("BAFFA9")

# accuracy-circle geometry as a fraction of the (square) bake canvas S —
# std's values with catch's badge tweak (catch lazer_results.py:156-164:
# pills slightly smaller + further out than std's 0.088/0.050 @ r 0.435 so
# tight top bands keep the A/S/SS badges legible — mania's 90/95/100 top
# bands crowd exactly like catch's did).
ACC_ARC_R = 0.350
ACC_ARC_W = 0.066
ACC_GRAD_R = 0.293
ACC_GRAD_W = 0.020
ACC_BADGE_R = 0.445
ACC_BADGE_W = 0.072
ACC_BADGE_H = 0.046

# results-screen text weights (Nunito variable `wght`, the lazer Torus
# stand-in — same as std/catch/taiko)
RESULTS_TEXT_WEIGHT = 500      # Nunito Medium (body/labels)
RESULTS_SCORE_WEIGHT = 330     # Nunito Light (the big rolling score)

TEXT_SS = 3                    # supersample factors (std's values)
SHAPE_SS = 2

# mod badges (std's results-panel values — catch lazer_results.py:174-189)
MOD_PILL_VH = 34.0
MOD_PILL_TEXT_VPX = 19.0
MOD_PILL_GAP_V = 8.0
MOD_PILL_ALPHA = 235

# osu! mod bitmask → display names (NC eats DT, PF eats SD) — std results.py
# port via catch :181-189. Mania key-mod bits (4K..9K/coop) are intentionally
# absent: lazer's ranking screen shows gameplay mods only.
_MOD_BITS = ((2, "EZ"), (1, "NF"), (256, "HT"), (8, "HD"), (16, "HR"),
             (16384, "PF"), (32, "SD"), (512, "NC"), (64, "DT"),
             (1024, "FL"), (128, "RX"), (8192, "AP"), (4096, "SO"),
             (4, "TD"), (536870912, "V2"))
_MOD_REDUCTION = {"EZ", "NF", "HT", "SO"}
_MOD_AUTOMATION = {"RX", "AP", "V2", "TD"}
MOD_COLOR_REDUCTION = (0.45, 0.78, 0.36)
MOD_COLOR_INCREASE = (0.90, 0.32, 0.42)
MOD_COLOR_AUTOMATION = (0.36, 0.62, 0.92)

# OsuColour.ForStarDifficulty gradient stops (star, hex) — std's port
STAR_SPECTRUM = [
    (0.1, "aaaaaa"), (0.1, "4290fb"), (1.25, "4fc0ff"), (2.0, "4fffd5"),
    (2.5, "7cff4f"), (3.3, "f6f05c"), (4.2, "ff8068"), (4.9, "ff3c71"),
    (5.8, "6563de"), (6.7, "18158e"), (7.7, "000000"), (9.0, "000000"),
]

# mania judgment palette (0..1) — the engine's own results colours
# (gpu/renderer.py jcolours + _draw_ur_histogram's timing bands:
# cyan=320, gold=300, green=200, yellow=100, gray=50, red=miss) so this
# screen reads as the same UI as the HUD it replaces.
MANIA_RESULT_COLORS = {
    "320":  (150 / 255, 215 / 255, 255 / 255),
    "300":  (255 / 255, 230 / 255, 120 / 255),
    "200":  (140 / 255, 220 / 255, 140 / 255),
    "100":  (240 / 255, 220 / 255,  90 / 255),
    "50":   (185 / 255, 190 / 255, 200 / 255),
    "MISS": (240 / 255,  96 / 255,  96 / 255),
}

# the bundled Nunito (copied from the std/catch renderers' assets — SIL OFL)
NUNITO_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "fonts",
    "Nunito[wght].ttf"))


def _nunito_loader(weight: int):
    """A (px)->font loader for the bundled Nunito at a variable `wght`
    (std's lazer-Torus stand-in — catch lazer_results.py:203-217). Falls
    back to the engine's HUD font resolver if the asset is missing so a
    stripped checkout still renders."""
    def _load(px: int):
        try:
            f = ImageFont.truetype(NUNITO_PATH, max(int(px), 6))
        except OSError:
            return _base_font(max(int(px), 6))
        try:
            f.set_variation_by_axes([weight])
        except Exception:  # noqa: BLE001 — non-variable build → default face
            pass
        return f
    return _load


_font_body = _nunito_loader(RESULTS_TEXT_WEIGHT)
_font_score = _nunito_loader(RESULTS_SCORE_WEIGHT)


# --- pure helpers (catch lazer_results.py:224-371, verbatim) -------------------------

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_quint(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 5


def ease_out_cubic(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 3


def mania_grade(acc_frac: float) -> str:
    """osu!mania grade from accuracy [0..1] — the RANK_THRESHOLDS cutoffs
    (the fail-soft fallback when the replay carries no grade; SS handled by
    the caller's all-perfect check, exactly like _compute_grade_from_replay)."""
    acc_frac = _clamp01(acc_frac)
    if acc_frac >= 1.0:
        return "SS"
    grade = "D"
    for lo, g in RANK_THRESHOLDS[:-1]:
        if acc_frac >= lo:
            grade = g
    return grade


def rank_ring_bands() -> list[tuple[float, float, str]]:
    """[(acc_lo, acc_hi, grade)] — lazer's GradedCircles band layout with
    MANIA's cutoffs: the S band stops at 1 − VIRTUAL_SS_PERCENTAGE and a
    distinct SS/X band owns the final slice (the virtual-SS notch)."""
    ss_lo = 1.0 - VIRTUAL_SS_PERCENTAGE
    bands = []
    for i, (lo, g) in enumerate(RANK_THRESHOLDS[:-1]):
        hi = RANK_THRESHOLDS[i + 1][0]
        if g == "S":
            hi = ss_lo
        bands.append((lo, hi, g))
    bands.append((ss_lo, 1.00, "SS"))
    return bands


def rank_badge_positions() -> list[tuple[float, str]]:
    """[(visual_acc, grade)] — lazer AccuracyCircle.cs RankBadge placement
    (each badge at its band's Interpolation.Lerp VISUAL position), computed
    from MANIA's cutoffs (catch lazer_results.py:272-293):

        RankBadge(accuracyD, Lerp(accuracyD, accuracyC, 0.5),  D)
        RankBadge(accuracyC, Lerp(accuracyC, accuracyB, 0.5),  C)
        RankBadge(accuracyB, Lerp(accuracyB, accuracyA, 0.5),  B)
        RankBadge(accuracyA, Lerp(accuracyA, accuracyS, 0.25), A)
        RankBadge(accuracyS, Lerp(accuracyS, accuracyX−VSS, 0.25), S)
        RankBadge(accuracyX, accuracyX, X/SS)
    """
    d, c, b, a, s, x = (t for t, _g in RANK_THRESHOLDS)
    ss_lo = x - VIRTUAL_SS_PERCENTAGE
    return [
        (_lerp(d, c, 0.5), "D"),
        (_lerp(c, b, 0.5), "C"),
        (_lerp(b, a, 0.5), "B"),
        (_lerp(a, s, 0.25), "A"),
        (_lerp(s, ss_lo, 0.25), "S"),
        (x, "SS"),
    ]


def for_star_difficulty(stars: float) -> tuple[float, float, float]:
    """The star-rating pill colour — OsuColour.ForStarDifficulty (std port)."""
    s = max(float(stars), 0.0)
    if s <= STAR_SPECTRUM[0][0]:
        return _hex(STAR_SPECTRUM[0][1])
    for i in range(1, len(STAR_SPECTRUM)):
        s0, h0 = STAR_SPECTRUM[i - 1]
        s1, h1 = STAR_SPECTRUM[i]
        if s <= s1:
            t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
            c0, c1 = _hex(h0), _hex(h1)
            return (_lerp(c0[0], c1[0], t), _lerp(c0[1], c1[1], t),
                    _lerp(c0[2], c1[2], t))
    return _hex(STAR_SPECTRUM[-1][1])


def target_arc_value(acc_frac: float, grade: str) -> float:
    """AccuracyCircle target fill: a true SS closes the ring, everything else
    caps at 1 − VIRTUAL_SS_PERCENTAGE so the SS notch stays open."""
    acc_frac = _clamp01(acc_frac)
    if grade in ("SS", "X", "SSH", "XH"):
        return acc_frac
    return min(1.0 - VIRTUAL_SS_PERCENTAGE, acc_frac)


def acc_to_angle_deg(acc: float) -> float:
    """Accuracy [0,1] → PIL degrees from 3 o'clock clockwise, ring starting
    at 12 o'clock."""
    return 270.0 + _clamp01(acc) * 360.0


def mods_string(mods: int) -> str:
    """Comma-joined mod acronyms of an .osr bitmask; NC/PF absorb DT/SD.
    '' for nomod (std results.py port via catch :327-334)."""
    if mods & 512:
        mods &= ~64
    if mods & 16384:
        mods &= ~32
    return ",".join(name for bit, name in _MOD_BITS if mods & bit)


def mod_pill_color(acr: str) -> tuple[float, float, float]:
    if acr in _MOD_REDUCTION:
        return MOD_COLOR_REDUCTION
    if acr in _MOD_AUTOMATION:
        return MOD_COLOR_AUTOMATION
    return MOD_COLOR_INCREASE


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n - 1] + "…"


def _bake_width(font, text: str, px: int) -> int:
    """The exact pixel width bake_text() produces (std port)."""
    if not text:
        return 1
    try:
        x0, _y0, x1, _y1 = font.getbbox(text)
    except AttributeError:
        w0, _h0 = font.getsize(text)          # type: ignore[attr-defined]
        x0, x1 = 0, w0
    pad = max(int(px) // 12, 2)
    return max(x1 - x0, 1) + 2 * pad


def _ellipsize(font, text: str, max_w: float, px: int) -> str:
    if _bake_width(font, text, px) <= max_w:
        return text
    while len(text) > 1:
        text = text[:-1]
        cand = text.rstrip() + "…"
        if _bake_width(font, cand, px) <= max_w:
            return cand
    return "…"


# --- procedural bakes (PIL) — catch lazer_results.py:374-821, verbatim ---------------

def bake_text(text: str, px: int, color, loader=_font_body,
              ss: int = TEXT_SS) -> Image.Image:
    """One baked text line → PIL RGBA. Supersampled at px*ss then LANCZOS-
    downscaled (std's bake_text). Blank text → 1×1 stub."""
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    px = max(int(px), 6)
    font = loader(px)
    try:
        x0, y0, x1, y1 = font.getbbox(text)
    except AttributeError:
        w0, h0 = font.getsize(text)          # type: ignore[attr-defined]
        x0, y0, x1, y1 = 0, 0, w0, h0
    pad = max(px // 12, 2)
    w = max(x1 - x0, 1) + 2 * pad
    h = max(y1 - y0, 1) + 2 * pad
    rgb = tuple(int(round(c * 255)) for c in color)
    ss = max(int(ss), 1)
    if ss == 1:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font,
                                 fill=(*rgb, 255))
        return img
    fb = loader(px * ss)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(text)
    except AttributeError:
        bw0, bh0 = fb.getsize(text)          # type: ignore[attr-defined]
        bx0, by0, bx1, by1 = 0, 0, bw0, bh0
    padb = pad * ss
    Wb = max(bx1 - bx0, 1) + 2 * padb
    Hb = max(by1 - by0, 1) + 2 * padb
    big = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(big).text((padb - bx0, padb - by0), text, font=fb,
                             fill=(*rgb, 255))
    return big.resize((w, h), Image.LANCZOS)


def fit_text(text: str, px_virtual: float, color, max_w_virtual: float,
             k: float, min_px_virtual: float | None = None,
             loader=_font_body) -> Image.Image:
    """Bake a text row SIZED TO FIT max_w_virtual (std's _fit_text): shrink
    the font toward min_px_virtual until it fits, then ellipsis-truncate."""
    text = text or ""
    if min_px_virtual is None:
        min_px_virtual = max(px_virtual * 0.62, 12.0)
    max_w = max_w_virtual * k
    px = float(px_virtual)
    step = max((px_virtual - min_px_virtual) / 12.0, 1.0)
    chosen = min_px_virtual
    while px >= min_px_virtual:
        fpx = max(int(round(px * k)), 6)
        if _bake_width(loader(fpx), text, fpx) <= max_w:
            chosen = px
            break
        px -= step
    fpx = max(int(round(chosen * k)), 6)
    font = loader(fpx)
    if _bake_width(font, text, fpx) > max_w:
        text = _ellipsize(font, text, max_w, fpx)
    return bake_text(text, fpx, color, loader)


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def _name_hash(name: str) -> int:
    import hashlib
    key = (name or "?").strip().lower().encode("utf-8", "ignore")
    return int(hashlib.md5(key).hexdigest(), 16)


def _avatar_initials(name: str) -> str:
    import re
    name = (name or "").strip()
    if not name:
        return "?"
    tokens = [t for t in re.split(r"[\s_.\-]+", name) if t]
    if len(tokens) >= 2:
        return (tokens[0][0] + tokens[1][0]).upper()
    return tokens[0][0].upper()


def bake_avatar(px: int, name: str, avatar_bytes: bytes | None = None
                ) -> Image.Image:
    """The featured-card CIRCULAR avatar chip (catch lazer_results.py:461-520,
    verbatim): the player's osu! avatar PNG bytes cover-fit + disc-clipped +
    soft ring when available, else the deterministic username-hued disc with
    centred initials. Never raises."""
    px = max(int(px), 8)
    h = _name_hash(name)
    hue = (h % 360) / 360.0
    sat = 0.42 + ((h >> 9) % 18) / 100.0
    if avatar_bytes:
        try:
            from io import BytesIO
            src = Image.open(BytesIO(avatar_bytes)).convert("RGBA")
            sw, sh = src.size
            scale = px / max(min(sw, sh), 1)
            nw = max(int(sw * scale + 0.5), px)
            nh = max(int(sh * scale + 0.5), px)
            src = src.resize((nw, nh), Image.LANCZOS)
            lft, top = (nw - px) // 2, (nh - px) // 2
            img = src.crop((lft, top, lft + px, top + px))
            mask = Image.new("L", (px, px), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, px - 1, px - 1], fill=255)
            img.putalpha(mask)
            d = ImageDraw.Draw(img)
            ring = _hsv(hue, max(sat - 0.16, 0.0), 0.92)
            rc = tuple(int(round(c * 255)) for c in ring)
            lw = max(int(px * 0.045), 2)
            off = lw * 0.5
            d.ellipse([off, off, px - 1 - off, px - 1 - off],
                      outline=(*rc, 150), width=lw)
            return img
        except Exception:  # noqa: BLE001 — corrupt/animated → procedural chip
            pass
    c0 = _hsv(hue, sat, 0.62)
    c1 = _hsv((hue + 0.06) % 1.0, min(sat + 0.08, 1.0), 0.34)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(px):
        f = y / max(px - 1, 1)
        col = tuple(int(round(_lerp(c0[i], c1[i], f) * 255)) for i in range(3))
        d.line([(0, y), (px, y)], fill=(*col, 255))
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, px - 1, px - 1], fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    ring = _hsv(hue, max(sat - 0.16, 0.0), 0.92)
    rc = tuple(int(round(c * 255)) for c in ring)
    lw = max(int(px * 0.045), 2)
    off = lw * 0.5
    d.ellipse([off, off, px - 1 - off, px - 1 - off],
              outline=(*rc, 150), width=lw)
    ini = _avatar_initials(name)
    font = _font_body(int(px * (0.46 if len(ini) >= 2 else 0.56)))
    try:
        x0, y0, x1, y1 = font.getbbox(ini)
    except AttributeError:
        x1, y1 = font.getsize(ini); x0 = y0 = 0     # type: ignore
    d.text(((px - (x1 - x0)) / 2 - x0, (px - (y1 - y0)) / 2 - y0), ini,
           font=font, fill=(255, 255, 255, 240))
    return img


def bake_grade_letter(text: str, px: int, fill, glow, loader=_font_body,
                      ss: int = TEXT_SS) -> Image.Image:
    """The AccuracyCircle centre rank letter (std port via catch :523-564):
    WHITE fill with a soft rank-coloured outer glow (lazer DrawableRank's
    coloured EdgeEffect). Supersampled then LANCZOS-downscaled."""
    px = max(int(px), 8)
    ss = max(int(ss), 1)
    font = loader(px)
    try:
        x0, y0, x1, y1 = font.getbbox(text)
    except AttributeError:
        x1, y1 = font.getsize(text); x0 = y0 = 0     # type: ignore
    tw = max(x1 - x0, 1)
    th = max(y1 - y0, 1)
    pad = max(int(px * 0.42), 8)
    W = tw + 2 * pad
    H = th + 2 * pad
    bpx = px * ss
    fb = loader(bpx)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(text)
    except AttributeError:
        bx1, by1 = fb.getsize(text); bx0 = by0 = 0   # type: ignore
    btw = max(bx1 - bx0, 1)
    bth = max(by1 - by0, 1)
    padb = pad * ss
    Wb = btw + 2 * padb
    Hb = bth + 2 * padb
    ox, oy = padb - bx0, padb - by0
    gc = tuple(int(round(c * 255)) for c in glow)
    glyph = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text((ox, oy), text, font=fb, fill=(*gc, 255))
    tight = glyph.filter(ImageFilter.GaussianBlur(max(bpx * 0.045, 1)))
    wide = glyph.filter(ImageFilter.GaussianBlur(max(bpx * 0.11, 2)))
    out = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    for layer in (wide, wide, tight, tight):          # stack → a bright halo
        out = Image.alpha_composite(out, layer)
    fc = tuple(int(round(c * 255)) for c in fill)
    ImageDraw.Draw(out).text((ox, oy), text, font=fb, fill=(*fc, 255))
    if ss != 1:
        out = out.resize((W, H), Image.LANCZOS)
    return out


def bake_star(px: int, color) -> Image.Image:
    """A filled 5-point star sprite (std port — the font has no ★ glyph)."""
    S = max(int(px), 8)
    ss = 4
    big = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    cx = cy = S * ss / 2.0
    r_out = S * ss * 0.5
    r_in = r_out * 0.42
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        ang = math.radians(-90 + i * 36)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    col = tuple(int(round(c * 255)) for c in color)
    d.polygon(pts, fill=(*col, 255))
    return big.resize((S, S), Image.LANCZOS)


def bake_round_panel(w: int, h: int, radius: int, top, bot, alpha: float,
                     border=None) -> Image.Image:
    """Rounded panel with a vertical top→bot gradient at `alpha` + optional
    border. Returns a PIL RGBA image (catch lb_cards.py:140-164, verbatim —
    inlined here because mania has no lb_cards module)."""
    w = max(int(w), 2)
    h = max(int(h), 2)
    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = grad.load()
    a = int(round(_clamp01(alpha) * 255))
    for y in range(h):
        f = y / max(h - 1, 1)
        r = int(round(_lerp(top[0], bot[0], f) * 255))
        g = int(round(_lerp(top[1], bot[1], f) * 255))
        b = int(round(_lerp(top[2], bot[2], f) * 255))
        for x in range(w):
            px[x, y] = (r, g, b, a)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1],
                                           radius=radius, fill=255)
    grad.putalpha(mask)
    if border is not None:
        br = tuple(int(round(c * 255)) for c in border)
        ImageDraw.Draw(grad).rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=radius, outline=(*br, 160), width=2)
    return grad


def bake_accuracy_base(px: int, ss: int = SHAPE_SS) -> Image.Image:
    """The AccuracyCircle background, the std 1:1 port (osu.Game/Screens/
    Ranking/Expanded/Accuracy/{AccuracyCircle,GradedCircles,RankBadge}.cs via
    catch :586-623): dim gray background ring + the THIN inner GradedCircles
    ForRank ring with the GRADE_SPACING notches + the six RankBadge pills
    OUTSIDE at their Lerp visual positions — with MANIA's cutoffs driving the
    bands/positions. Baked once; the cyan→green achieved arc is a separate
    bake drawn over."""
    S_target = max(int(px), 64)
    ss = max(int(ss), 1)
    S = S_target * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = S / 2.0
    half_gap = GRADE_SPACING_PERCENTAGE / 2.0
    # dim gray "Background circle" (OsuColour.Gray(47), alpha 0.5), full ring
    R = ACC_ARC_R * S
    W = max(int(round(ACC_ARC_W * S)), 2)
    d.arc([cx - R, cy - R, cx + R, cy + R], 0, 360, fill=(47, 47, 47, 128),
          width=W)
    # thin inner GradedCircles rank ring (ForRank bands + boundary notches)
    Rg = ACC_GRAD_R * S
    Wg = max(int(round(ACC_GRAD_W * S)), 2)
    gbox = [cx - Rg, cy - Rg, cx + Rg, cy + Rg]
    for lo, hi, g in rank_ring_bands():
        col = tuple(int(round(c * 255)) for c in FOR_RANK[g])
        a0 = acc_to_angle_deg(lo + half_gap)
        a1 = acc_to_angle_deg(hi - half_gap)
        if a1 > a0:
            d.arc(gbox, a0, a1, fill=(*col, 255), width=Wg)
    # RankBadge pills at their Lerp visual positions, outside the ring
    for vis, g in rank_badge_positions():
        ang = math.radians(acc_to_angle_deg(vis))
        bx = cx + ACC_BADGE_R * S * math.cos(ang)
        by = cy + ACC_BADGE_R * S * math.sin(ang)
        _badge_pill(img, bx, by, S, g)
    if ss != 1:
        img = img.resize((S_target, S_target), Image.LANCZOS)
    return img


def _badge_pill(img, cx, cy, S, grade) -> None:
    """One RankBadge: a rounded pill in the rank's ForRank colour with the
    rank letter + soft drop shadow (std port via catch :626-646)."""
    d = ImageDraw.Draw(img)
    w = ACC_BADGE_W * S
    h = ACC_BADGE_H * S
    col = tuple(int(round(c * 255)) for c in FOR_RANK.get(grade,
                                                          (0.8, 0.8, 0.85)))
    sh = max(h * 0.10, 1.0)
    d.rounded_rectangle([cx - w / 2, cy - h / 2 + sh, cx + w / 2,
                         cy + h / 2 + sh], radius=h / 2, fill=(0, 0, 0, 70))
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        radius=h / 2, fill=(*col, 255))
    label = "SS" if grade in ("SS", "X", "XH") else grade
    font = _font_body(max(int(h * (0.72 if len(label) == 1 else 0.56)), 6))
    try:
        x0, y0, x1, y1 = font.getbbox(label)
    except AttributeError:
        x1, y1 = font.getsize(label); x0 = y0 = 0    # type: ignore
    d.text((cx - (x1 - x0) / 2 - x0, cy - (y1 - y0) / 2 - y0), label,
           font=font, fill=(255, 255, 255, 245))


_ARC_GRAD_CACHE: dict = {}


def _arc_gradient_rgb(S: int):
    """Cached S×S vertical cyan→green gradient (std port)."""
    g = _ARC_GRAD_CACHE.get(S)
    if g is None:
        ys = np.linspace(0.0, 1.0, S).reshape(S, 1)
        top = np.array(ARC_GRAD_TOP)
        bot = np.array(ARC_GRAD_BOT)
        col = top + (bot - top) * ys
        rgb = np.repeat(col[:, None, :], S, axis=1)
        g = np.clip(np.round(rgb * 255), 0, 255).astype("u1")
        _ARC_GRAD_CACHE[S] = g
    return g


def bake_accuracy_arc(px: int, progress_acc: float,
                      ss: int = SHAPE_SS) -> Image.Image:
    """The achieved-accuracy arc (0 → progress_acc) — lazer's "Accuracy
    circle": the FIXED vertical cyan→green gradient with a light tip dash
    (std port via catch :667-697). Re-baked per progress bucket while
    sweeping."""
    S_target = max(int(px), 64)
    ss = max(int(ss), 1)
    S = S_target * ss
    cx = cy = S / 2.0
    R = ACC_ARC_R * S
    W = max(int(round(ACC_ARC_W * S)), 2)
    mask = Image.new("L", (S, S), 0)
    if progress_acc > 0.0005:
        ImageDraw.Draw(mask).arc(
            [cx - R, cy - R, cx + R, cy + R], acc_to_angle_deg(0.0),
            acc_to_angle_deg(progress_acc), fill=255, width=W)
    rgb = _arc_gradient_rgb(S)
    out = np.dstack([rgb, np.asarray(mask, dtype="u1")]).copy()
    img = Image.fromarray(out, "RGBA")
    if progress_acc > 0.0005:
        d = ImageDraw.Draw(img)
        ang = math.radians(acc_to_angle_deg(progress_acc))
        r0 = R - W / 2.0 - 1
        r1 = R + W / 2.0 + 1
        d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)),
                (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang))],
               fill=(255, 255, 255, 235), width=max(int(W * 0.16), 2))
    if ss != 1:
        img = img.resize((S_target, S_target), Image.LANCZOS)
    return img


def bake_star_pill(stars: float | None, k: float) -> Image.Image:
    """The StarRatingDisplay pill (std's _bake_star_pill via catch :700-743):
    rounded ForStarDifficulty-coloured pill + procedural star + the star
    number, flipping to dark text on light backgrounds. Supersampled."""
    txt = f"{stars:.2f}" if stars is not None else "--"
    bg = for_star_difficulty(stars if stars is not None else 0.0)
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg = (0.06, 0.06, 0.09) if lum > 0.6 else (1.0, 1.0, 1.0)
    fpx = max(int(22 * k), 8)
    font = _font_body(fpx)
    try:
        x0, y0, x1, y1 = font.getbbox(txt)
    except AttributeError:
        x1, y1 = font.getsize(txt); x0 = y0 = 0     # type: ignore
    tw, th = x1 - x0, y1 - y0
    star_sz = int(fpx * 0.92)
    padx = int(fpx * 0.5)
    pady = int(fpx * 0.32)
    gap = int(fpx * 0.26)
    H = max(th, star_sz) + 2 * pady
    W = padx + star_sz + gap + tw + padx
    ss = TEXT_SS
    fpxb = fpx * ss
    fb = _font_body(fpxb)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(txt)
    except AttributeError:
        bx1, by1 = fb.getsize(txt); bx0 = by0 = 0   # type: ignore
    twb, thb = bx1 - bx0, by1 - by0
    star_szb = star_sz * ss
    padxb, padyb, gapb = padx * ss, pady * ss, gap * ss
    Hb = max(thb, star_szb) + 2 * padyb
    Wb = padxb + star_szb + gapb + twb + padxb
    img = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    dd = ImageDraw.Draw(img)
    bgc = tuple(int(round(c * 255)) for c in bg)
    dd.rounded_rectangle([0, 0, Wb - 1, Hb - 1], radius=Hb // 2,
                         fill=(*bgc, 255))
    img.alpha_composite(bake_star(star_szb, fg),
                        (padxb, (Hb - star_szb) // 2))
    fgc = tuple(int(round(c * 255)) for c in fg)
    dd.text((padxb + star_szb + gapb - bx0, (Hb - thb) // 2 - by0), txt,
            font=fb, fill=(*fgc, 255))
    return img.resize((W, H), Image.LANCZOS)


def bake_mod_pill(text: str, color, k: float) -> Image.Image:
    """One played-mod badge (std's _bake_mod_pill via catch :746-770):
    rounded category-coloured pill with the acronym in white, fixed height,
    supersampled."""
    H = max(int(round(MOD_PILL_VH * k)), 12)
    fpx = max(int(round(MOD_PILL_TEXT_VPX * k)), 8)
    padx = int(round(fpx * 0.62))
    ss = TEXT_SS
    Hb = H * ss
    fb = _font_body(fpx * ss)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(text)
    except AttributeError:
        bx1, by1 = fb.getsize(text); bx0 = by0 = 0    # type: ignore
    twb, thb = bx1 - bx0, by1 - by0
    padxb = padx * ss
    Wb = twb + 2 * padxb
    W = max(int(round(Wb / ss)), 8)
    img = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    dd = ImageDraw.Draw(img)
    bgc = tuple(int(round(c * 255)) for c in color)
    dd.rounded_rectangle([0, 0, Wb - 1, Hb - 1], radius=Hb // 2,
                         fill=(*bgc, MOD_PILL_ALPHA))
    ty = (Hb - thb) // 2 - by0
    dd.text((padxb - bx0, ty), text, font=fb, fill=(255, 255, 255, 255))
    return img.resize((W, H), Image.LANCZOS)


def _bake_area_chart(values: list, w: int, h: int,
                     tick_fracs: list[float] | None = None) -> Image.Image:
    """A filled area chart of `values` (left→right) for the stage-2 COMBO /
    DIFFICULTY panel (catch :773-821, verbatim): the area under the curve
    wears the results screen's cyan→green accent gradient (the accuracy
    arc's), a brighter line rides the top, a dim baseline sits underneath,
    and optional red ticks mark combo breaks at the given x fractions.
    Supersampled 2× then LANCZOS-downscaled. Returns a PIL RGBA image."""
    w = max(int(w), 32)
    h = max(int(h), 24)
    ss = 2
    W, H = w * ss, h * ss
    vals = [max(float(v), 0.0) for v in values] or [0.0]
    # peak-preserving resample so short combo spikes / strain peaks survive
    cols = min(len(vals), max(w // 3, 60))
    step = len(vals) / cols
    res = [max(vals[int(i * step): max(int((i + 1) * step),
                                       int(i * step) + 1)] or [0.0])
           for i in range(cols)]
    peak = max(res) or 1.0
    inset = max(int(H * 0.04), 2)              # headroom above the peak
    base_y = H - max(int(2 * ss), 2)           # baseline (bottom)
    span_h = base_y - inset
    pts = [(i * (W - 1) / max(cols - 1, 1),
            base_y - span_h * (v / peak)) for i, v in enumerate(res)]
    # gradient-filled area under the curve
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(
        [(0, base_y)] + pts + [(W - 1, base_y)], fill=255)
    ys = np.linspace(0.0, 1.0, H).reshape(H, 1, 1)
    grad = (np.array(ARC_GRAD_TOP) + (np.array(ARC_GRAD_BOT)
                                      - np.array(ARC_GRAD_TOP)) * ys)
    rgb = np.clip(np.round(np.repeat(grad, W, axis=1) * 255), 0,
                  255).astype("u1")
    alpha = (np.asarray(mask, dtype="f4") * (0.62 / 255.0) * 255) \
        .astype("u1")
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    d = ImageDraw.Draw(img)
    # dim baseline + the brighter curve line on top
    d.line([(0, base_y), (W - 1, base_y)], fill=(255, 255, 255, 90),
           width=max(ss, 1))
    d.line(pts, fill=(232, 255, 245, 235), width=2 * ss, joint="curve")
    # red combo-break ticks
    for f in tick_fracs or []:
        x = _clamp01(f) * (W - 1)
        d.line([(x, inset), (x, base_y)], fill=(255, 90, 90, 150),
               width=max(ss, 1))
    return img.resize((w, h), Image.LANCZOS)


# --- data plumbing -------------------------------------------------------------------

@dataclass(frozen=True)
class ManiaResultsData:
    """Everything the ranking screen needs, snapshotted from the RenderPlan
    once per render (results_data_from_plan). Pure data — the screen class
    never touches the plan, the GL context, or the filesystem (except the
    optional avatar PNG read, fail-soft)."""
    replay: Any                          # ReplayInfo (.osr header truth)
    beatmap: Any                         # BeatmapInfo (title/artist/diff/creator)
    score: int                           # standardised total (plan.score_final)
    stars: float                         # 0 hides the pill (plan.stars)
    pp: float                            # plan.player_pp (--pp already applied)
    max_pp: float                        # plan.max_pp; <=0 hides the PP cells
    results_start_ms: int                # plan.results_start_ms (drives age_ms)
    combo_values: tuple[int, ...]        # combo after each judgment, press order
    map_max_combo: int                   # total judgment events (combo ceiling)
    ur: float                            # final unstable rate (10·σ of offsets)
    avg_offset_ms: float                 # final mean signed hit offset
    featured_avatar_png: str | None = None   # options.featured_avatar_png


def results_data_from_plan(plan) -> ManiaResultsData | None:
    """Snapshot a RenderPlan into ManiaResultsData (called once per render by
    render.py / compositor.py, right after set_banner_text). The combo series
    folds plan.judgment_timeline with EXACTLY build_frame_state's combo rule
    (miss → 0, anything else +1 — render.py) so the stage-2 COMBO chart is
    the curve the gameplay HUD showed; UR/avg use render.py's own formula
    over the full offset list. Fail-soft: any problem returns None and the
    renderer stays on the legacy argon card (loudly, at the call site)."""
    try:
        combo_vals: list[int] = []
        c = 0
        for _eff_t, j in plan.judgment_timeline:
            c = 0 if j.judgment == "miss" else c + 1
            combo_vals.append(c)
        offs = [j.hit_offset_ms for j in plan.judgment_events
                if j.hit_offset_ms is not None]
        if offs:
            avg = sum(offs) / len(offs)
            var = sum((x - avg) ** 2 for x in offs) / len(offs)
            ur = 10.0 * (var ** 0.5)
        else:
            avg, ur = 0.0, 0.0
        score = (plan.score_final if plan.score_final is not None
                 else int(getattr(plan.replay, "score", 0) or 0))
        return ManiaResultsData(
            replay=plan.replay,
            beatmap=plan.modded,
            score=int(score),
            stars=float(plan.stars or 0.0),
            pp=float(plan.player_pp or 0.0),
            max_pp=float(plan.max_pp or 0.0),
            results_start_ms=int(plan.results_start_ms),
            combo_values=tuple(combo_vals),
            map_max_combo=len(combo_vals),
            ur=float(ur),
            avg_offset_ms=float(avg),
            featured_avatar_png=getattr(plan.options, "featured_avatar_png",
                                        None),
        )
    except Exception:  # noqa: BLE001 — data plumbing never breaks a render
        return None


def _paste(base: Image.Image, img: Image.Image, cx: float, cy: float,
           alpha: float) -> None:
    """Composite `img` centred at (cx, cy) at `alpha` (catch :918-928)."""
    if alpha <= 0.003 or img is None:
        return
    if alpha < 0.997:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    base.alpha_composite(img, (int(cx - img.width / 2.0),
                               int(cy - img.height / 2.0)))


# --- the screen ----------------------------------------------------------------------

class ManiaLazerResults:
    """Bakes the lazer ranking screen once, animates the two-stage reveal per
    frame, then caches the settled frame (catch lazer_results.py:933-1583,
    CatchLazerResults — adapted from catch's CPU rgb-array compositor to a
    full-frame RGBA overlay layer the GL renderer uploads; see module
    docstring)."""

    def __init__(self, resolution, data: ManiaResultsData, board=None):
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.k = self.H / UH
        self.data = data
        self.board = board                 # lb_cards.BakedBoard | None
        self._settled = None               # cached final RGBA frame
        # pose-signature frame memo (catch :944-953): every animated
        # parameter of the screen is a CLAMPED pure function of (op, age_ms),
        # so once they all saturate — or hold between animation phases — the
        # rendered frame is a pure function of the signature. The GL side
        # also uses the signature as its texture-upload token, so unchanged
        # poses re-upload nothing.
        self._pose_key = None
        self._pose_frame = None
        self._arc_img = None
        self._arc_bucket = -1.0
        self._score_img = None
        self._score_val = -1

        # --- results data (the replay's authoritative mania counts) ---------
        r = data.replay
        geki = int(getattr(r, "count_geki", 0) or 0)   # 320 (rainbow/MAX)
        n300 = int(getattr(r, "count_300", 0) or 0)
        katu = int(getattr(r, "count_katu", 0) or 0)   # 200
        n100 = int(getattr(r, "count_100", 0) or 0)
        n50 = int(getattr(r, "count_50", 0) or 0)
        miss = int(getattr(r, "count_miss", 0) or 0)
        self.counts = (geki, n300, katu, n100, n50, miss)
        total = geki + n300 + katu + n100 + n50 + miss
        # PREFER the accuracy the mania pipeline already carries:
        # replay.accuracy is a PERCENTAGE (0..100) parsed with the display
        # geki weight (replay.py) — the exact figure the HUD/results show —
        # so this matches the rest of the pipeline. Fall back to the same
        # display formula when the replay carries none (taiko port pattern).
        meta_pct = float(getattr(r, "accuracy", 0.0) or 0.0)
        if meta_pct > 0.0:
            self.acc_frac = max(0.0, min(100.0, meta_pct)) / 100.0
        else:
            aw = int(getattr(r, "mania_acc_weight", 305) or 305)
            weighted = (aw * geki + 300 * n300 + 200 * katu + 100 * n100
                        + 50 * n50)
            self.acc_frac = (weighted / (aw * total)) if total else 1.0
        self.acc_pct = 100.0 * self.acc_frac
        # centre grade letter + ring colour: PREFER replay.grade (the
        # pipeline's grade — render._compute_grade_from_replay semantics),
        # else derive from accuracy (taiko port pattern).
        raw_grade = str(getattr(r, "grade", "") or "").upper()
        self.grade = raw_grade if raw_grade in FOR_RANK \
            else mania_grade(self.acc_frac)
        self.grade_letter = {"X": "SS", "XH": "SS", "SSH": "SS",
                             "SH": "S"}.get(self.grade, self.grade)
        self.grade_rgb = FOR_RANK.get(self.grade, (0.8, 0.8, 0.85))
        self.target_arc = target_arc_value(self.acc_frac, self.grade)
        self.max_combo = int(getattr(r, "max_combo", 0) or 0)
        self.mods = int(getattr(r, "mods", 0) or 0)
        self.score_total = int(data.score)
        # stars / pp straight from the plan (overrides already applied there;
        # the plan's 0-gates map to the port's None-gates — 0 hides).
        self.stars = float(data.stars) if data.stars > 0 else None
        self.pp = float(data.pp) if data.max_pp > 0 else None
        self.max_pp = float(data.max_pp) if data.max_pp > 0 else None
        # featured avatar PNG bytes (options.featured_avatar_png — the same
        # service-resolved file the old card used). Fail-soft → procedural
        # username chip (catch's set_featured_avatar_png contract, :886-915).
        self._avatar_bytes = None
        try:
            from pathlib import Path as _P
            if data.featured_avatar_png:
                self._avatar_bytes = _P(data.featured_avatar_png).read_bytes()
        except OSError:
            self._avatar_bytes = None

        # --- stage-2 source data (real mania data; never faked) -------------
        # combo-over-time from the judgment timeline (the running combo the
        # HUD showed) + the play's final UR/avg offset for the JUDGEMENTS
        # footer. Fail-soft: an empty series → the COMBO panel falls back to
        # "timeline unavailable" at bake time.
        self._combo_values = list(data.combo_values)
        self._map_max_combo = int(data.map_max_combo)
        self._ur = float(data.ur)
        self._avg_offset = float(data.avg_offset_ms)
        self._bake_static()

    # -- baking ------------------------------------------------------------------

    def _bake_static(self) -> None:
        """catch lazer_results.py:1014-1111 — mania judgment grids + plan-fed
        stars/pp; everything else verbatim."""
        k = self.k
        r = self.data.replay
        bm = self.data.beatmap
        self.cx_px = self.W / 2.0
        self.panel_cy_v = UH / 2.0
        self.panel_cy_px = self.panel_cy_v * k
        self.panel_top_px = (self.panel_cy_v - PANEL_H / 2.0) * k
        # main panel bg (std's colours/radius/border)
        self.panel_img = bake_round_panel(
            int(PANEL_W * k), int(PANEL_H * k), int(26 * k),
            (0.12, 0.13, 0.17), (0.05, 0.05, 0.07), 0.93,
            border=(0.3, 0.33, 0.4))
        # avatar + header rows (auto-fit, std budgets). Player name = the
        # .osr header (replay.player_name) — NOT the banner-string parse the
        # old card used.
        player_name = getattr(r, "player_name", "") or "Player"
        self.avatar_img = bake_avatar(int(52 * k), player_name,
                                      self._avatar_bytes)
        # header/text row sizes: the STD port's originals (std
        # lazer_results.py:1393,1414-1417 — name 27 / title 30 / artist 21),
        # NOT catch's upsized 30/34/24: std's sizes are the ones authored for
        # a THREE-grid-row panel (std draws grid a+b+c like mania does);
        # catch upsized because its panel only carries two grid rows and the
        # taller text would overflow mania's third row into the date.
        content_w = PANEL_W - 48.0
        self.name_img = fit_text(player_name, 27, (1, 1, 1),
                                 content_w - 64.0, k)
        self.title_img = fit_text(getattr(bm, "title", "") or "", 30,
                                  (0.95, 0.96, 1.0), content_w, k)
        self.artist_img = fit_text(getattr(bm, "artist", "") or "", 21,
                                   (0.72, 0.75, 0.85), content_w, k)
        # accuracy circle base + centre grade letter (white + ForRank glow)
        self.acc_base_img = bake_accuracy_base(int(ACC_DISP * k))
        self.grade_img = bake_grade_letter(self.grade_letter, int(150 * k),
                                           (0.99, 0.99, 1.0), self.grade_rgb)
        # score baked lazily (rolls); seed with 0
        self._score_text(0)
        # star pill + played-mods badges + diff/creator rows
        self.star_pill = bake_star_pill(self.stars, k)
        acr = [a for a in mods_string(self.mods).split(",") if a]
        self.mod_pills = [bake_mod_pill(a, mod_pill_color(a), k) for a in acr]
        # mania's BeatmapInfo names the diff `difficulty` (std/catch: version).
        # Row sizes = std's originals (std :1431-1433, diff 25 / creator 19 —
        # the three-grid-row budget; see the header-size comment above).
        version = getattr(bm, "difficulty", "") or ""
        self.diff_img = bake_text(_clip(version, 28), int(25 * k),
                                  (0.9, 0.92, 1.0))
        creator = getattr(bm, "creator", "") or ""
        self.creator_img = bake_text(
            f"mapped by {_clip(creator, 22)}" if creator else "",
            int(19 * k), (0.65, 0.68, 0.78))
        # The star/diff/creator row is CENTRED in _draw_star_row — pixel
        # budget refit so a long diff + "mapped by …" never runs off the
        # card (catch :1055-1073, verbatim).
        _row_gap = 14.0 * k
        _row_avail = max(content_w * k - self.star_pill.width - 2.0 * _row_gap,
                         1.0)
        if self.diff_img.width + self.creator_img.width > _row_avail:
            _d_budget = min(float(self.diff_img.width),
                            max(_row_avail - self.creator_img.width,
                                _row_avail * 0.5))
            self.diff_img = fit_text(version, 25.0, (0.9, 0.92, 1.0),
                                     _d_budget / k, k)
            _c_budget = max(_row_avail - self.diff_img.width, 1.0)
            self.creator_img = fit_text(
                f"mapped by {creator}" if creator else "", 19.0,
                (0.65, 0.68, 0.78), _c_budget / k, k)
        # stats grids — mania's SIX judgments over TWO rows (std's multi-row
        # grid layout, osu_std_renderer/render/lazer_results.py:1740-1751;
        # catch/taiko fit theirs in one). Colours = the engine's judgment
        # palette so the screen reads as one UI with the HUD.
        geki, n300, katu, n100, n50, miss = self.counts
        pp_txt = (f"{self.pp:.0f}" if self.pp is not None else "--")
        grid_a = [
            ("ACCURACY", f"{self.acc_pct:.2f}%", (0.95, 0.96, 1.0)),
            ("MAX COMBO", f"{self.max_combo}x", (0.95, 0.96, 1.0)),
            ("PP", pp_txt, (0.6, 0.86, 1.0)),
        ]
        grid_b = [
            ("320", str(geki), MANIA_RESULT_COLORS["320"]),
            ("300", str(n300), MANIA_RESULT_COLORS["300"]),
            ("200", str(katu), MANIA_RESULT_COLORS["200"]),
        ]
        grid_c = [
            ("100", str(n100), MANIA_RESULT_COLORS["100"]),
            ("50", str(n50), MANIA_RESULT_COLORS["50"]),
            ("MISS", str(miss), MANIA_RESULT_COLORS["MISS"]),
        ]
        self.grid_a = [self._grid_cell(lbl, val, col) for lbl, val, col in grid_a]
        self.grid_b = [self._grid_cell(lbl, val, col) for lbl, val, col in grid_b]
        self.grid_c = [self._grid_cell(lbl, val, col) for lbl, val, col in grid_c]
        # mania's .osr parse drops the timestamp field (replay.py skips the
        # int64), so this always falls back to now() — the same fallback
        # catch's port uses for a timestamp-less meta (catch :1093-1097).
        ts = getattr(r, "timestamp", None)
        if not isinstance(ts, datetime):
            ts = datetime.now()
        self.date_img = bake_text(f"Played on {ts.strftime('%d %b %Y %H:%M')}",
                                  int(18 * k), (0.6, 0.63, 0.72))
        # stage 2: the featured panel's slide-left target + the stats panels
        # that unfold from the right (std geometry, mania data). Fail-soft:
        # any bake problem disables stage 2 LOUDLY and the screen holds
        # centred exactly as stage 1.
        self._stage2 = False
        try:
            self._bake_stage2()
        except Exception as e:  # noqa: BLE001 — stage 2 never breaks results
            import sys
            import traceback
            print("[mania-renderer] !!! RESULTS STAGE-2 BAKE FAILED — "
                  f"holding the stage-1 screen: {e}", file=sys.stderr)
            traceback.print_exc()
            self._stage2 = False

    def _grid_cell(self, label, value, color):
        # std's STAT_LABEL_VPX=15 / STAT_VALUE_VPX=28 (std :122-123) — the
        # three-grid-row sizes; catch's 16/32 fits only its two rows.
        return (bake_text(label, int(15 * self.k), (0.6, 0.63, 0.73)),
                bake_text(value, int(28 * self.k), color))

    def _score_text(self, value: int) -> None:
        self._score_img = bake_text(f"{value:,}", int(64 * self.k), (1, 1, 1),
                                    loader=_font_score)
        self._score_val = value

    # -- stage-2 bake ------------------------------------------------------------

    def _bake_stage2(self) -> None:
        """Stage-2 geometry + content (catch :1124-1243 — std's exact layout,
        mania's real data).

        std geometry: the panel slides from the screen centre to left_cx =
        PANEL_W/2 + 70; the three stats panels live at STATS_X0 = left_cx +
        PANEL_W/2 + 40, filling the rest of the width minus a 64-virtual-px
        right margin, each (PANEL_H − 2·24)/3 tall with 24 vpx gaps. Content
        is baked ONCE here; only the panel squash + alpha animate per frame."""
        k = self.k
        self.uw = self.W / k                       # virtual width
        self.center_cx_v = self.uw / 2.0
        self.left_cx_v = PANEL_W / 2.0 + 70.0
        self.STATS_X0 = self.left_cx_v + PANEL_W / 2.0 + 40.0
        self.STATS_W = self.uw - self.STATS_X0 - 64.0
        self.STATS_H = (PANEL_H - 2 * 24.0) / 3.0
        if self.STATS_W < 200.0:      # too narrow (portrait render) → stage 1 only
            return
        self.stats_panel_img = bake_round_panel(
            int(self.STATS_W * k), int(self.STATS_H * k), int(20 * k),
            (0.11, 0.12, 0.16), (0.05, 0.05, 0.07), 0.93,
            border=(0.28, 0.31, 0.38))
        title_col = (0.8, 0.83, 0.92)
        label_col = (0.85, 0.88, 0.95)
        foot_col = (0.62, 0.66, 0.76)
        geki, n300, katu, n100, n50, miss = self.counts
        judged_total = geki + n300 + katu + n100 + n50 + miss

        # --- panel 1: PERFORMANCE (acc / combo / pp bars, all real) ---------
        self.perf_title = bake_text("PERFORMANCE", int(22 * k), title_col)
        self._perf_rows = []           # (label_img, pct_img, pct, colour)

        def _perf_row(label, pct, colour, fmt=".0f"):
            self._perf_rows.append((
                bake_text(label, int(20 * k), label_col),
                bake_text(f"{pct * 100:{fmt}}%", int(20 * k), (1, 1, 1)),
                _clamp01(pct), colour))

        # accuracy keeps 2 decimals — .0f would round 99.55% up to "100%"
        # right next to the main panel's exact figure
        _perf_row("Accuracy", self.acc_frac, MANIA_RESULT_COLORS["320"],
                  fmt=".2f")
        # mania combo ceiling = every judgment event in the timeline (the
        # engine's own combo model: each judged head/tail increments, miss
        # resets — render.py build_frame_state), so the bar compares the
        # play's peak against the chart's full-combo value.
        if self._map_max_combo > 0:
            _perf_row("Combo", self.max_combo / self._map_max_combo,
                      MANIA_RESULT_COLORS["300"])
        if self.pp is not None and self.max_pp:
            _perf_row("PP", self.pp / self.max_pp, (0.6, 0.86, 1.0))
        if self.pp is not None and self.max_pp is not None:
            foot = (f"Achieved {self.pp:.0f}pp  /  "
                    f"Maximum {self.max_pp:.0f}pp")
        else:
            foot = "performance unavailable (no rosu)"
        self.perf_foot = bake_text(foot, int(18 * k), foot_col)

        # --- panel 2: COMBO progression (the judgment timeline's series) ----
        pad = 26.0 * k
        th = self.perf_title.height
        chart_w = int(self.STATS_W * k - 2 * pad)
        chart_h = int(self.STATS_H * k - (22 * k + th) - 54 * k)
        vals = self._combo_values
        self.combo_chart = None
        if len(vals) >= 8:
            self.combo_title = bake_text("COMBO", int(22 * k), title_col)
            # combo breaks: a reset to 0 after a positive run (a mania MISS
            # is the only thing that zeroes combo, so every 0 here is a real
            # break)
            breaks = [i for i in range(1, len(vals))
                      if vals[i] == 0 and vals[i - 1] > 0]
            tick_fr = [i / max(len(vals) - 1, 1) for i in breaks]
            self.combo_chart = _bake_area_chart(vals, chart_w, chart_h,
                                                tick_fracs=tick_fr)
            self.combo_foot = bake_text(
                f"peak {self.max_combo}x   ·   {len(breaks)} combo "
                f"break{'' if len(breaks) == 1 else 's'}",
                int(18 * k), foot_col)
        else:                           # honest: nothing graphable
            self.combo_title = bake_text("COMBO", int(22 * k), title_col)
            self.combo_foot = bake_text("timeline unavailable",
                                        int(18 * k), foot_col)

        # --- panel 3: JUDGEMENTS (all six tiers as a share of the judged
        # notes — the same bar widget as catch's CATCHES / taiko's
        # JUDGEMENTS panel, mania data) + the play's real timing stats in
        # the footer (the old card's headline UR/avg readout, kept). --------
        self.judge_title = bake_text("JUDGEMENTS", int(22 * k), title_col)
        self._judge_rows = []          # (label_img, count_img, frac, colour)
        for label, cnt, ckey in (
                ("320", geki, "320"),
                ("300", n300, "300"),
                ("200", katu, "200"),
                ("100", n100, "100"),
                ("50", n50, "50"),
                ("Miss", miss, "MISS")):
            frac = _clamp01(cnt / judged_total) if judged_total > 0 else \
                (1.0 if cnt > 0 else 0.0)
            self._judge_rows.append((
                bake_text(label, int(20 * k), label_col),
                bake_text(f"{cnt}/{judged_total}" if judged_total > 0
                          else str(cnt), int(20 * k), (1, 1, 1)),
                frac, MANIA_RESULT_COLORS[ckey]))
        if self._ur > 0.0 or self._avg_offset != 0.0:
            foot = (f"UR {self._ur:.1f}   ·   "
                    f"avg {self._avg_offset:+.1f} ms")
        else:
            foot = "no timing data"
        self.judge_foot = bake_text(foot, int(18 * k), foot_col)
        self._stage2 = True

    # -- per-frame draw ----------------------------------------------------------

    def render_overlay(self, opacity: float, age_ms: float | None):
        """Compose the results screen as a FULL-FRAME RGBA layer (straight
        alpha): the black results wash at `opacity` with the screen content
        over it — exactly the OVER composite catch's render_frame performs
        onto the CPU rgb array (catch :1247-1305), split out so mania's GL
        renderer can upload it and blend it over the frame in one quad.
        Returns (HxWx4 uint8 array, pose_token); an unchanged pose_token
        means the previous upload is still valid."""
        op = _clamp01(opacity)
        if age_ms is None:                 # back-compat: no timeline → settled
            age_ms = SETTLE_MS if op >= 0.999 else op * FADE_MS
        settled = op >= 0.999 and age_ms >= SETTLE_MS
        if settled and self._settled is not None:
            return self._settled, "settled"
        sig = self._pose_signature(op, age_ms)
        if sig is not None and sig == self._pose_key \
                and self._pose_frame is not None:
            return self._pose_frame, sig

        # BLACK wash: std's scene has faded to black by results_start and the
        # screen dims what's left — the results sit on clean black. As a
        # transparent-backed layer the wash is (0,0,0, op·255); GL blending
        # it over the frame == catch's alpha_composite of the same wash.
        base = Image.new("RGBA", (self.W, self.H), (0, 0, 0, int(op * 255)))

        fade = _clamp01(age_ms / FADE_MS) * op
        if fade > 0.003:
            # stage-2 open: the featured panel slides from centre to left
            # while the stats panels unfold from the right (std timeline)
            stage2 = getattr(self, "_stage2", False)
            open_p = ease_out_quint((age_ms - STAGE1_MS) / OPEN_MS) \
                if (stage2 and age_ms > STAGE1_MS) else 0.0
            panel_cx = _lerp(self.center_cx_v, self.left_cx_v, open_p) \
                * self.k if stage2 else self.cx_px
            # stage-1 flanks: the ranked cards unfurl outward (std timing),
            # then fade out with the panel slide as stage 2 opens
            stage1_a = fade * (1.0 - _clamp01((age_ms - STAGE1_MS)
                                              / (OPEN_MS * 0.6))) \
                if stage2 else fade
            if self.board is not None and stage1_a > 0.003:
                self._draw_board(base, age_ms, stage1_a,
                                 dx=panel_cx - self.cx_px)
            if stage2 and age_ms > STAGE1_MS:
                self._draw_stats_panels(base, age_ms, op)
            self._draw_panel(base, age_ms, fade, panel_cx)

        out = np.asarray(base)             # HxWx4 uint8, straight alpha
        if settled:
            self._settled = out
        if sig is not None:
            self._pose_key, self._pose_frame = sig, out
            return out, sig
        return out, ("nosig", float(op), float(age_ms))

    def _pose_signature(self, op, age_ms):
        """Tuple of EVERY animated parameter render_overlay consumes for
        (op, age_ms) — all of them clamped easings, so the tuple saturates
        when the animation does (catch :1307-1337, verbatim). None disables
        the memo (fail-safe)."""
        try:
            op = _clamp01(op)
            fade = _clamp01(age_ms / FADE_MS) * op
            stage2 = getattr(self, "_stage2", False)
            open_p = ease_out_quint((age_ms - STAGE1_MS) / OPEN_MS) \
                if (stage2 and age_ms > STAGE1_MS) else 0.0
            stage1_a = fade * (1.0 - _clamp01((age_ms - STAGE1_MS)
                                              / (OPEN_MS * 0.6))) \
                if stage2 else fade
            lb = None
            if self.board is not None and fade > 0.003 and stage1_a > 0.003:
                lb = tuple(self._lb_slide(c.stagger, age_ms)
                           for c in (getattr(self.board, "cards", []) or []))
            # featured panel internals (arc sweep / grade punch / score roll)
            sweep = ease_out_cubic((age_ms - SWEEP_DELAY_MS) / SWEEP_MS) \
                if age_ms > SWEEP_DELAY_MS else 0.0
            arc_bucket = round(self.target_arc * sweep, 3)
            score_val = int(round(self.score_total * sweep))
            badge_start = SWEEP_DELAY_MS + SWEEP_MS - 130.0
            grade_p = ease_out_cubic((age_ms - badge_start) / BADGE_MS) \
                if age_ms >= badge_start else None
            unfold = tuple(self._panel_unfold(age_ms, i) for i in range(3)) \
                if (stage2 and age_ms > STAGE1_MS) else None
            return (op, fade, open_p, stage1_a, lb, arc_bucket, score_val,
                    grade_p, unfold)
        except Exception:  # noqa: BLE001 — memo is best-effort, never breaks
            return None

    def _blit(self, base, img, cx, top_y, a) -> float:
        """Paste centred at cx with the image TOP at top_y; return height."""
        _paste(base, img, cx, top_y + img.height / 2.0, a)
        return float(img.height)

    def _draw_panel(self, base, age_ms, a, cx=None) -> None:
        """catch :1344-1384, verbatim except the judgment grid: mania draws
        THREE grid rows (ACC/COMBO/PP, 320/300/200, 100/50/MISS) with std's
        multi-row gaps (std lazer_results.py:1742-1747)."""
        k = self.k
        if cx is None:
            cx = self.cx_px
        _paste(base, self.panel_img, cx, self.panel_cy_px, a)
        top = (self.panel_cy_v - PANEL_H / 2.0 + 40.0) * k
        y = top
        # header: avatar + name (centred group)
        av = float(self.avatar_img.width)
        nw = float(self.name_img.width)
        group_w = av + 12 * k + nw
        gx = cx - group_w / 2.0
        _paste(base, self.avatar_img, gx + av / 2.0, y + av / 2.0, a)
        _paste(base, self.name_img, gx + av + 12 * k + nw / 2.0,
               y + av / 2.0, a)
        y += av + 16 * k
        y += self._blit(base, self.title_img, cx, y, a) + 4 * k
        y += self._blit(base, self.artist_img, cx, y, a) + 20 * k
        # accuracy circle: base ring + sweeping arc + punching grade letter
        acc_d = ACC_DISP * k
        circ_cy = y + acc_d / 2.0
        _paste(base, self.acc_base_img, cx, circ_cy, a)
        self._draw_acc_arc(base, age_ms, cx, circ_cy, a)
        self._draw_grade(base, age_ms, cx, circ_cy, a)
        y += acc_d + 6 * k
        # score (rolls with the sweep)
        self._roll_score(age_ms)
        y += self._blit(base, self._score_img, cx, y, a) + 12 * k
        # played-mod badge row (nomod → no row, layout unchanged)
        mod_h = self._draw_mod_row(base, cx, y, a)
        if mod_h > 0.0:
            y += mod_h + 14 * k
        # with the mod row present the gaps below tighten so the row + date
        # fit the fixed-height panel (std's exact precedent for its extra
        # spinner grid row, std lazer_results.py:1738-1741: 16/14/16 →
        # 12/10/10); a nomod panel keeps the base spacing.
        has_mods = mod_h > 0.0
        g_star, g_a, g_b, g_c = (16, 12, 10, 10) if has_mods \
            else (22, 16, 14, 16)
        # star / diff / creator centred row
        y += self._draw_star_row(base, cx, y, a) + g_star * k
        # stats grids (mania: ACC/COMBO/PP then 320/300/200 then 100/50/MISS)
        y += self._draw_grid_row(base, self.grid_a, cx, y, a,
                                 PANEL_W * 0.86) + g_a * k
        y += self._draw_grid_row(base, self.grid_b, cx, y, a,
                                 PANEL_W * 0.86) + g_b * k
        y += self._draw_grid_row(base, self.grid_c, cx, y, a,
                                 PANEL_W * 0.86) + g_c * k
        # date row FLOWS after the last grid row (std :1748-1751 — catch
        # anchors it to the panel bottom, which only its two-row grid allows;
        # a third row would collide with an anchored date)
        self._blit(base, self.date_img, cx, y, a)

    def _draw_acc_arc(self, base, age_ms, cx, cy, a) -> None:
        sweep = ease_out_cubic((age_ms - SWEEP_DELAY_MS) / SWEEP_MS) \
            if age_ms > SWEEP_DELAY_MS else 0.0
        prog = self.target_arc * sweep
        bucket = round(prog, 3)
        if bucket != self._arc_bucket or self._arc_img is None:
            self._arc_img = bake_accuracy_arc(int(ACC_DISP * self.k), prog)
            self._arc_bucket = bucket
        _paste(base, self._arc_img, cx, cy, a)

    def _draw_grade(self, base, age_ms, cx, cy, a) -> None:
        badge_start = SWEEP_DELAY_MS + SWEEP_MS - 130.0
        if age_ms < badge_start:
            return
        p = ease_out_cubic((age_ms - badge_start) / BADGE_MS)
        scale = _lerp(1.42, 1.0, p)
        ga = a * _clamp01(p * 1.4)
        img = self.grade_img
        if abs(scale - 1.0) > 0.005:
            img = img.resize((max(int(img.width * scale), 1),
                              max(int(img.height * scale), 1)), Image.LANCZOS)
        _paste(base, img, cx, cy, ga)

    def _roll_score(self, age_ms) -> None:
        sweep = ease_out_cubic((age_ms - SWEEP_DELAY_MS) / SWEEP_MS) \
            if age_ms > SWEEP_DELAY_MS else 0.0
        val = int(round(self.score_total * sweep))
        if val != self._score_val:
            self._score_text(val)

    def _draw_mod_row(self, base, cx, top_y, a) -> float:
        if not self.mod_pills:
            return 0.0
        gap = MOD_PILL_GAP_V * self.k
        total = sum(p.width for p in self.mod_pills) \
            + gap * (len(self.mod_pills) - 1)
        h = max(p.height for p in self.mod_pills)
        x = cx - total / 2.0
        for p in self.mod_pills:
            _paste(base, p, x + p.width / 2.0, top_y + h / 2.0, a)
            x += p.width + gap
        return float(h)

    def _draw_star_row(self, base, cx, top_y, a) -> float:
        gap = 14 * self.k
        rows = [self.star_pill, self.diff_img, self.creator_img]
        total = sum(r.width for r in rows) + gap * (len(rows) - 1)
        h = max(r.height for r in rows)
        x = cx - total / 2.0
        for r in rows:
            _paste(base, r, x + r.width / 2.0, top_y + h / 2.0, a)
            x += r.width + gap
        return float(h)

    def _draw_grid_row(self, base, cells, cx, top_y, a, span_virtual) -> float:
        k = self.k
        span = span_virtual * k
        n = len(cells)
        col_w = span / n
        x0 = cx - span / 2.0
        row_h = 0.0
        for i, (label, value) in enumerate(cells):
            ccx = x0 + col_w * (i + 0.5)
            _paste(base, label, ccx, top_y + label.height / 2.0, a)
            _paste(base, value, ccx,
                   top_y + label.height + 6 * k + value.height / 2.0, a)
            row_h = max(row_h, label.height + 6 * k + value.height)
        return row_h

    # -- flank leaderboard (catch :1455-1480 verbatim; board = lb_cards) -----------

    def _lb_slide(self, i: int, age_ms: float) -> tuple[float, float]:
        """std's _lb_slide: card i eases in over LB_SLIDE_MS (OutQuint) from
        LB_SLIDE_OFFSET virtual px outward, starting at LB_SLIDE_START_MS +
        i*LB_SLIDE_STAGGER_MS. Returns (outward_offset_virtual, alpha)."""
        t = age_ms - LB_SLIDE_START_MS - i * LB_SLIDE_STAGGER_MS
        p = ease_out_quint(t / LB_SLIDE_MS)
        return (1.0 - p) * LB_SLIDE_OFFSET, p

    def _draw_board(self, base, age_ms, a, dx: float = 0.0) -> None:
        """`dx` (px) shifts the whole board with the featured panel during
        the stage-2 slide — std positions its flanks off the moving panel
        edges, so the cards + banner track the panel while fading out."""
        board = self.board
        for c in getattr(board, "cards", []) or []:
            off, ca = self._lb_slide(c.stagger, age_ms)
            if ca <= 0.003:
                continue
            x = c.cx + c.out_dir * off * self.k + dx
            _paste(base, c.img, x, c.cy, a * ca)
        banner = getattr(board, "banner", None)
        if banner is not None:
            bimg, _bx, _by = banner
            _paste(base, bimg, self.cx_px + dx,
                   self.panel_top_px - bimg.height / 2.0 - 6 * self.k, a)

    # -- stats panels (stage 2) ----------------------------------------------------

    def _panel_unfold(self, age_ms: float, idx: int) -> float:
        """std's _panel_unfold: panel `idx` starts at STAGE1_MS + 120 +
        idx·STAGGER_MS and eases open over OPEN_MS (OutQuint)."""
        start = STAGE1_MS + 120.0 + idx * STAGGER_MS
        return ease_out_quint((age_ms - start) / OPEN_MS) \
            if age_ms > start else 0.0

    def _draw_stats_panels(self, base, age_ms, op) -> None:
        """The three stats panels unfolding from the right, staggered (catch
        :1491-1515, verbatim): each panel bg is width-squashed by its unfold
        progress, anchored at STATS_X0 (growing rightward), the content
        fading in at full-width position once the panel is ~55% open."""
        k = self.k
        top0 = self.panel_cy_v - PANEL_H / 2.0
        drawers = (self._draw_perf, self._draw_combo, self._draw_judgements)
        for idx, drawer in enumerate(drawers):
            s = self._panel_unfold(age_ms, idx)
            if s <= 0.003:
                continue
            pcy = (top0 + self.STATS_H * (idx + 0.5) + idx * 24.0) * k
            left = self.STATS_X0 * k
            pw, ph = self.stats_panel_img.size
            drawn_w = max(int(pw * s), 1)
            img = self.stats_panel_img if drawn_w >= pw else \
                self.stats_panel_img.resize((drawn_w, ph), Image.BILINEAR)
            _paste(base, img, left + drawn_w / 2.0, pcy,
                   min(s * 1.3, 1.0) * op)
            content_a = _clamp01((s - 0.55) / 0.45) * op
            if content_a > 0.01:
                full_cx = (self.STATS_X0 + self.STATS_W / 2.0) * k
                drawer(base, full_cx,
                       (top0 + idx * (self.STATS_H + 24.0)) * k, content_a)

    def _draw_bar_rows(self, base, d, cx, top_y, a, title_img, rows,
                       foot_img, bar_x_off: float, right_reserve: float
                       ) -> None:
        """Shared std perf-panel layout (catch :1517-1548, verbatim): title
        top-left, label + track + fill + right-aligned value per row, footer
        centred at the bottom. `rows` = (label_img, value_img, frac, colour)."""
        k = self.k
        pad = 26 * k
        left = cx - self.STATS_W * k / 2.0 + pad
        right = cx + self.STATS_W * k / 2.0 - pad
        _paste(base, title_img, left + title_img.width / 2.0,
               top_y + 22 * k + title_img.height / 2.0, a)
        th = title_img.height
        y = top_y + 22 * k + th + 20 * k
        bar_x = left + bar_x_off * k
        bar_w = right - bar_x - right_reserve * k
        bar_h = 16 * k
        row_gap = (self.STATS_H * k - 22 * k - th - 20 * k - 40 * k) / \
            max(len(rows), 1)
        for i, (label, val, frac, col) in enumerate(rows):
            ry = y + i * row_gap
            cyy = ry + bar_h / 2.0
            _paste(base, label, left + label.width / 2.0, cyy, a)
            rc = tuple(int(round(c * 255)) for c in col)
            d.rectangle([bar_x, ry, bar_x + bar_w, ry + bar_h],
                        fill=(51, 56, 71, int(0.7 * a * 255)))
            fw = max(bar_w * _clamp01(frac), 2.0)
            d.rectangle([bar_x, ry, bar_x + fw, ry + bar_h],
                        fill=(*rc, int(0.95 * a * 255)))
            _paste(base, val, right - val.width / 2.0, cyy, a)
        _paste(base, foot_img, cx,
               top_y + self.STATS_H * k - 20 * k - foot_img.height / 2.0, a)

    def _draw_perf(self, base, cx, top_y, a) -> None:
        """PERFORMANCE: accuracy / combo-vs-map-max / achieved-vs-SS-pp bars
        (std's perf-bar layout, mania's real numbers)."""
        d = ImageDraw.Draw(base, "RGBA")
        self._draw_bar_rows(base, d, cx, top_y, a, self.perf_title,
                            self._perf_rows, self.perf_foot,
                            bar_x_off=130.0, right_reserve=60.0)

    def _draw_combo(self, base, cx, top_y, a) -> None:
        """COMBO progression (the judgment timeline's series) — the pre-baked
        area chart in std's timing-panel slot (title top-left, chart body,
        stat footer)."""
        k = self.k
        pad = 26 * k
        left = cx - self.STATS_W * k / 2.0 + pad
        _paste(base, self.combo_title, left + self.combo_title.width / 2.0,
               top_y + 22 * k + self.combo_title.height / 2.0, a)
        th = self.combo_title.height
        chart = self.combo_chart
        if chart is not None:
            _paste(base, chart, cx,
                   top_y + 22 * k + th + 16 * k + chart.height / 2.0, a)
            foot_cy = top_y + 22 * k + th + 16 * k + chart.height + 22 * k
        else:
            foot_cy = top_y + self.STATS_H * k / 2.0
        _paste(base, self.combo_foot, cx, foot_cy, a)

    def _draw_judgements(self, base, cx, top_y, a) -> None:
        """JUDGEMENTS: all six mania tiers as share-of-judged bars in the
        engine's judgment palette + the UR/avg-offset footer."""
        d = ImageDraw.Draw(base, "RGBA")
        self._draw_bar_rows(base, d, cx, top_y, a, self.judge_title,
                            self._judge_rows, self.judge_foot,
                            bar_x_off=130.0, right_reserve=110.0)
