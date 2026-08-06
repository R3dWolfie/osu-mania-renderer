"""HUD elements — lazer-faithful (per r3drenderer_lazer_fidelity).

Health = the legacy scorebar (scorebar-bg/colour/marker, top-left) when the
skin ships it, else the Argon procedural health capsule. Score/accuracy use
the skin score font. R3D web-viewer chrome (title, side UR bars, vertical HP)
is stripped to no-ops; the legacy render_mania keeps the R3D look as fallback.
"""
from __future__ import annotations

from osu_mania_renderer_v2.argon import is_argon_default
from osu_mania_renderer_v2.gpu.text import pill_to_texture
from osu_mania_renderer_v2.render.element_common import (
    is_keycount_acronym,
    mod_fill_colour,
)

# Argon counter geometry. The argon-counter glyphs are 240px square boxes with
# the digit content inset ~31px per side; ARGON_OVERLAP pulls the boxes together
# so the visible digits sit tight/condensed like lazer's counter. The HUD is
# authored in lazer's 1080p design space (absolute px), so everything scales by
# height/1080 → pixel-exact at 1080p output.
ARGON_OVERLAP = 60          # native px of box-to-box overlap (out of 240);
                            # ~content-width advance so digits just touch (the
                            # ~31px inset per side is the doubled padding cancelled)
ARGON_WIRE_ALPHA = 0.15     # dim "wireframes" template behind the live digits


def _draw_mod_icons(ctx, right_x: float, top_y: float) -> None:
    """lazer ModDisplay: active mods as flat-topped hexagons (mod_hex tinted
    by ModType), acronym centred in fill×0.1. Top-right, horizontal flow,
    rightmost on top, resting overlap. Native mania key-count is excluded."""
    s = ctx.scene
    mods = [m for m in (s.mod_acronyms or ()) if not is_keycount_acronym(m)]
    if not mods:
        return
    s2 = ctx.height / 768.0
    box = max(24, int(48 * s2))          # MOD_ICON_SIZE 80 × MOD_ICON_SCALE 0.6
    step = max(10, int(33 * s2))         # 48 − (25×0.6) overlap
    total_w = box + (len(mods) - 1) * step
    left = right_x - total_w
    for i, m in enumerate(mods):
        fx = left + i * step
        fill = mod_fill_colour(m)
        # Hexagon drawn into a square box (the letterbox encodes its aspect).
        _draw_tl(ctx, "mod_hex", fx, top_y, box, box, (*fill, 1.0))
        # Acronym centred, dark tint (lerp black→fill at 0.1).
        fg = (int(fill[0] * 0.1 * 255), int(fill[1] * 0.1 * 255),
              int(fill[2] * 0.1 * 255), 255)
        tex, tw, th = ctx.text(m, max(10, int(box * 0.42)), fg)
        cx = fx + box / 2.0
        cy_gl = ctx.height - top_y - box / 2.0      # hexagon centre (GL)
        ctx.draw_external(tex, int(cx - tw / 2), int(cy_gl - th / 2), tw, th, 1.0)


def _draw_tl(ctx, name_or_idx, left, top, w, h, tint, *, is_idx=False, direct=False):
    """Draw a sprite positioned by its TOP-LEFT corner in screen pixels
    (y measured downward from the top). Converts to the engine's GL
    bottom-left origin. `direct=True` draws via the full-res direct path
    (for wide sprites like the scorebar). Skips degenerate sizes."""
    if w <= 0 or h <= 0:
        return
    gl_y = ctx.height - top - h
    if direct:
        ctx.draw_direct(name_or_idx, int(left), int(gl_y), int(w), int(h), tint)
    elif is_idx:
        ctx.draw_sprite_idx(name_or_idx, int(left), int(gl_y), int(w), int(h), tint)
    else:
        ctx.draw_sprite(name_or_idx, int(left), int(gl_y), int(w), int(h), tint)


def _scorebar_fill_colour(hp: float) -> tuple[float, float, float]:
    """lazer LegacyHealthDisplay.getFillColour (new style): white at full,
    fading white→black approaching 0.5, then black→red approaching 0."""
    if hp >= 0.5:
        return (1.0, 1.0, 1.0)
    if hp >= 0.2:
        t = (0.5 - hp) / 0.3            # 0 at .5 → 1 at .2
        v = 1.0 - t
        return (v, v, v)
    t = (0.2 - hp) / 0.2                # 0 at .2 → 1 at 0
    return (t, 0.0, 0.0)


# --- R3D web-viewer chrome: not part of lazer. The wiki path reproduces
# lazer exactly (per r3drenderer_lazer_fidelity), so these are no-ops here.
# The legacy render_mania keeps the R3D look as the fallback renderer.
def hit_error_popups(*, element, skin, assets, variables, ctx) -> None:
    return  # lazer mania has no per-column hit-error number popups


def hit_strip(*, element, skin, assets, variables, ctx) -> None:
    return  # R3D rainbow UR strip; lazer uses BarHitErrorMeter (not drawn)


def _fmt_time(ms: float) -> str:
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}"


def _argon_song_progress(ctx) -> None:
    """lazer ArgonSongProgressBar: a thin bar across the bottom (0.9 width,
    centred) with elapsed time at the bottom-left and remaining at the right."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    A = rc.height / 1080.0
    p = max(0.0, min(1.0, s.song_progress))
    pad = 14 * A
    bar_h = max(3, int(7 * A))
    bw = rc.width * 0.9
    bx = (rc.width - bw) / 2.0
    top = rc.height - pad - bar_h          # from-top y of the bar
    _draw_tl(ctx, "column_bg", bx, top, bw, bar_h, (0.32, 0.33, 0.38, 0.55))
    _draw_tl(ctx, "column_bg", bx, top, bw * p, bar_h, (1, 1, 1, 0.95))
    # cyan head glow at the fill edge.
    _draw_tl(ctx, "column_bg", bx + bw * p, top, max(2, int(5 * A)), bar_h,
             (0.49, 0.84, 0.99, 0.6))
    # Elapsed (left) + remaining (right) time, just above the bar.
    elapsed = max(0.0, float(s.t_ms))
    total = elapsed / p if p > 0.01 else elapsed
    et, ew, eh = fr._cached_text(_fmt_time(elapsed), 20, (235, 235, 245, 255))
    fr._draw_external_texture(et, x=int(bx),
                              y=int(rc.height - top + 2 * A), w=ew, h=eh, alpha=0.9)
    rt, rw, rh = fr._cached_text("-" + _fmt_time(max(0.0, total - elapsed)), 20,
                                 (235, 235, 245, 255))
    fr._draw_external_texture(rt, x=int(bx + bw - rw),
                              y=int(rc.height - top + 2 * A), w=rw, h=rh, alpha=0.9)


def progress_bar(*, element, skin, assets, variables, ctx) -> None:
    if not ctx.options.show_progress_bar:
        return
    if is_argon_default(ctx, 0):
        _argon_song_progress(ctx)
    else:
        ctx.fr._draw_progress_bar(ctx.scene)


# Hit-error meter judgement windows (ms) → colour, centre-out (lazer mania).
_UR_BANDS = (
    (20.0, (108, 192, 255)),    # perfect  (blue)
    (43.0, (100, 220, 130)),    # great    (green)
    (76.0, (240, 220, 90)),     # good     (yellow)
    (106.0, (240, 160, 80)),    # ok       (orange)
    (127.0, (237, 73, 92)),     # meh/miss (red)
)


def _argon_hit_error(ctx) -> None:
    """lazer BarHitErrorMeter (Argon): a vertical meter on the right with the
    judgement windows as colour bands, white ticks for recent hit offsets, a
    white centre line, and an arrow at the rolling average."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    A = rc.height / 1080.0
    cx = rc.width - int(34 * A)
    cy = rc.height // 2                 # GL centre
    half = int(rc.height * 0.16)
    rng = _UR_BANDS[-1][0]              # ±ms mapped to the half-height
    band_w = max(3, int(6 * A))
    bx = cx - band_w // 2
    # Colour bands, centre outward, mirrored above/below the centre line.
    prev = 0.0
    for win, col in _UR_BANDS:
        y0 = int(prev / rng * half)
        y1 = int(win / rng * half)
        seg = max(1, y1 - y0)
        c = (col[0] / 255, col[1] / 255, col[2] / 255, 0.5)
        fr._draw_sprite("column_bg", bx, cy + y0, band_w, seg, c)
        fr._draw_sprite("column_bg", bx, cy - y1, band_w, seg, c)
        prev = win
    # White centre line.
    fr._draw_sprite("column_bg", cx - int(11 * A), cy - max(1, int(1 * A)),
                    int(22 * A), max(2, int(2 * A)), (1, 1, 1, 0.9))
    # Ticks for recent hit offsets (white), newest brightest.
    offs = (s.recent_offsets or ())[-20:]
    tick_w = int(20 * A)
    for i, off in enumerate(offs):
        frac = max(-1.0, min(1.0, off / rng))
        ty = cy + int(frac * half)
        a = 0.25 + 0.6 * (i + 1) / max(1, len(offs))
        fr._draw_sprite("column_bg", cx - tick_w // 2, ty - max(1, int(1 * A)),
                        tick_w, max(2, int(2 * A)), (1, 1, 1, a))
    # Rolling-average arrow.
    if offs:
        ay = cy + int(max(-1.0, min(1.0, s.avg_hit_offset_ms / rng)) * half)
        fr._draw_sprite("column_bg", cx + int(13 * A), ay - int(3 * A),
                        int(6 * A), int(6 * A), (1, 1, 1, 0.95))


def fail_overlay(*, element, skin, assets, variables, ctx) -> None:
    s = ctx.scene
    if s.hp <= 0.001 and s.results_opacity <= 0:
        ctx.fr._draw_fail_overlay()


def hp_bar(*, element, skin, assets, variables, ctx) -> None:
    """Health bar at the TOP — legacy scorebar when the skin ships it,
    else the Argon procedural capsule."""
    if not ctx.options.show_hp_bar:
        return
    # Argon default: the health line is drawn in _draw_argon_hud (after the
    # wedge, so it isn't covered). hp_bar runs before hud in RENDER_ORDER.
    if is_argon_default(ctx, 0):
        return
    hp = max(0.0, min(1.0, getattr(ctx.scene, "hp", 1.0)))
    if ctx.atlas.global_source("scorebar_bg") == "user":
        _legacy_scorebar(ctx, hp)
    else:
        _argon_health(ctx, hp)


def _legacy_scorebar(ctx, hp: float) -> None:
    """lazer LegacyHealthDisplay. Top-left, sprites scaled osu-px→render
    (s = height/480). scorebar-bg at (0,0); the colour fill sits at the
    old/new style offset and is clipped to HP×width (empties from the
    right); the marker rides the fill's right edge. New style = the skin
    ships scorebar-marker."""
    atlas = ctx.atlas
    # Legacy HUD textures render at native px × (height/768) — measured:
    # Vio scorebar-bg 40px → 59px at 1125h (×1.475).
    s = ctx.height / 768.0
    new_style = atlas.global_source("scorebar_marker") == "user"

    bg_w, bg_h = atlas.global_native_size("scorebar_bg")
    _draw_tl(ctx, "scorebar_bg", 0, 0, bg_w * s, bg_h * s, (1, 1, 1, 1), direct=True)

    col_w, col_h = atlas.global_native_size("scorebar_colour")
    if col_w <= 0:
        return
    # Fill offset in lazer's 768-space (osu value × 1.6): new (12,12.48),
    # old (4.8,16).
    off_x, off_y = (12.0, 12.48) if new_style else (4.8, 16.0)
    fill_w_full = col_w * s
    fill_h = col_h * s
    fill_w = hp * fill_w_full        # clip-approx: solid bars squish ≈ clip
    fr_tint = _scorebar_fill_colour(hp) if new_style else (1.0, 1.0, 1.0)
    _draw_tl(ctx, "scorebar_colour", off_x * s, off_y * s, fill_w, fill_h,
             (*fr_tint, 1.0), direct=True)

    # Marker at the right edge of the fill, centred (new) / top edge (old).
    right_x = off_x * s + fill_w
    if new_style:
        mk = "scorebar_marker"
        mw, mh = atlas.global_native_size(mk)
        mcx = right_x
        mcy = off_y * s + fill_h / 2.0
        _draw_tl(ctx, mk, mcx - mw * s / 2.0, mcy - mh * s / 2.0,
                 mw * s, mh * s, (*fr_tint, 1.0))
    else:
        # Old style: ki / kidanger / kidanger2 by HP, centred on the edge.
        mk = ("scorebar_kidanger2" if hp < 0.2
              else "scorebar_kidanger" if hp < 0.5 else "scorebar_ki")
        if atlas.global_source(mk) == "user":
            mw, mh = atlas.global_native_size(mk)
            _draw_tl(ctx, mk, right_x - mw * s / 2.0, off_y * s - mh * s / 2.0,
                     mw * s, mh * s, (1, 1, 1, 1))


def _argon_health(ctx, hp: float) -> None:
    """Argon procedural health: a rounded white capsule at (50,20), W=300,
    BarHeight=30 (lazer ArgonSkin coords, scaled by height/768), filling
    0→HP left→right over a dim track, cyan trailing glow."""
    s = ctx.height / 768.0
    x = 50 * s
    top = 20 * s
    w = 300 * s
    bh = max(3, int(30 * s))
    # Dim track (full width) then the white fill 0→HP.
    _draw_tl(ctx, "column_bg", x, top, w, bh, (0.10, 0.11, 0.13, 0.85))
    fill_w = hp * w
    # Trailing cyan glow just past the fill head.
    glow_w = max(2, int(6 * s))
    _draw_tl(ctx, "column_bg", x + fill_w, top, glow_w, bh,
             (0.49, 0.84, 0.99, 0.5))
    _draw_tl(ctx, "column_bg", x, top, fill_w, bh, (1, 1, 1, 0.95))


def banner(*, element, skin, assets, variables, ctx) -> None:
    return  # R3D song-title banner; lazer has no gameplay title banner


def _draw_fallback_hud(ctx) -> None:
    """Score + accuracy for skins WITHOUT a score font. lazer would use the
    default skin's font here; we render a clean PIL readout (no 4K pill, no
    zero-padding) top-right, plus the hexagon mod icons. Mirrors the skin-font
    layout so it sits in the same place."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    if not ctx.options.show_score:
        _draw_mod_icons(ctx, rc.width - max(8, int(rc.width * 0.012)),
                        max(8, int(rc.height * 0.03)))
        return
    if s.results_opacity > 0:
        disp_score, disp_acc = s.score, s.accuracy
    else:
        disp_score = s.score_smoothed if s.score_smoothed > 0 else s.score
        disp_acc = s.accuracy_smoothed
    right_pad = max(8, int(rc.width * 0.012))
    top_pad = max(8, int(rc.height * 0.02))
    right_x = rc.width - right_pad

    score_h = max(20, int(rc.height * 0.058))
    stex, sw, sh = fr._cached_text(f"{int(disp_score):d}", score_h, (255, 255, 255, 255))
    fr._draw_external_texture(stex, x=right_x - sw, y=rc.height - top_pad - sh,
                              w=sw, h=sh, alpha=1.0)
    acc_h = max(14, int(rc.height * 0.036))
    atex, aw, ah = fr._cached_text(f"{disp_acc:.2f}%", acc_h, (235, 235, 245, 255))
    acc_y = rc.height - top_pad - sh - ah - max(2, int(rc.height * 0.008))
    fr._draw_external_texture(atex, x=right_x - aw, y=acc_y, w=aw, h=ah, alpha=0.95)
    # lazer hexagon mod icons under the readout (key-count excluded).
    mods_top = (rc.height - acc_y) + max(6, int(rc.height * 0.012))
    _draw_mod_icons(ctx, right_x, mods_top)


def _argon_number(ctx, text, *, x, center_y, glyph_h, align, alpha=1.0,
                  tint=(1.0, 1.0, 1.0)):
    """Draw `text` in the argon-counter font with the dim wireframe template
    behind it (lazer's ArgonCounterTextComponent: live digits over the
    'wireframes' backing)."""
    ctx.draw_number(text, x=x, center_y=center_y, glyph_h=glyph_h,
                    overlap_px=ARGON_OVERLAP, align=align,
                    alpha=ARGON_WIRE_ALPHA, font="argon", wireframe=True,
                    tint=(0.7, 0.75, 0.85))
    ctx.draw_number(text, x=x, center_y=center_y, glyph_h=glyph_h,
                    overlap_px=ARGON_OVERLAP, align=align, alpha=alpha,
                    font="argon", tint=tint)


# NOTE: the old _draw_leaderboard (a hardcoded fake "#1" own-score card —
# grey placeholder avatar, name parsed off the banner string) was REMOVED
# 2026-08: no sibling renderer draws a gameplay scoreboard (std's
# render/scoreboard.py is an explicit accepted-but-no-op stub pending the
# osu!API hand-off; catch has no gameplay leaderboard element at all). The
# REAL per-map leaderboard now lives where the siblings put theirs: the
# results screen's flank cards (hud/lb_cards.py + hud/leaderboard.py, drawn
# by hud/lazer_results.py).


def _draw_argon_hud(ctx) -> None:
    """lazer's Argon default HUD: score in the top-left wedge, accuracy
    top-right (with an 'ACCURACY' label + pp under it), all in the
    argon-counter font. Layout from ArgonSkin (1080p design space, scaled
    by height/1080). Combo is drawn over the playfield in notes.py."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    A = rc.height / 1080.0          # argon design-space scale

    if s.results_opacity > 0:
        disp_score, disp_acc = s.score, s.accuracy
    else:
        disp_score = s.score_smoothed if s.score_smoothed > 0 else s.score
        disp_acc = s.accuracy_smoothed

    # ── Score: top-left wedge banner, number right-aligned inside it ──
    wedge_w, wedge_h = 380 * A, 72 * A
    if ctx.atlas.global_source("argon_wedge") in ("bundle", "user"):
        _draw_tl(ctx, "argon_wedge", 0, 0, wedge_w, wedge_h, (1, 1, 1, 1),
                 direct=True)

    # Health: lazer ArgonHealthDisplay — a THICK glossy white tube tracing the
    # top of the score wedge (MAIN_PATH_RADIUS=10 → ~20u thick, white #FFF,
    # cyan #7ED7FD glow). A dim track tube, the white fill 0→HP over it, and a
    # cyan glow at the fill head.
    if ctx.options.show_hp_bar:
        hp = max(0.0, min(1.0, getattr(s, "hp", 1.0)))
        hx, hw = 26 * A, 320 * A
        hh = max(5, int(20 * A))          # MAIN_PATH_RADIUS*2
        hy = 4 * A                        # ride the very top edge of the wedge
        # Soft cyan glow halo behind the tube.
        _draw_tl(ctx, "argon_hp", hx - 4 * A, hy - 4 * A, hw + 8 * A, hh + 8 * A,
                 (0.49, 0.84, 0.99, 0.18), direct=True)
        # Dim track tube (full width), then the bright white fill to HP.
        _draw_tl(ctx, "argon_hp", hx, hy, hw, hh, (0.32, 0.34, 0.40, 0.85),
                 direct=True)
        fw = max(hh, hp * hw)             # keep a round cap even near 0
        _draw_tl(ctx, "argon_hp", hx, hy, fw, hh, (1, 1, 1, 1.0), direct=True)
        # Cyan glow at the fill head.
        _draw_tl(ctx, "argon_hp", hx + fw - hh, hy - 3 * A, hh * 1.6, hh + 6 * A,
                 (0.49, 0.84, 0.99, 0.6), direct=True)
    if ctx.options.show_score:
        score_box = 52 * A
        # Origin TopRight at x = components_x_offset(50) + 200 = 250; the
        # number is vertically centred in the wedge.
        score_right = 250 * A
        # Sit below the HP tube (which rides the wedge top).
        score_cy_gl = rc.height - 46 * A
        _argon_number(ctx, f"{int(disp_score):d}", x=score_right,
                      center_y=score_cy_gl, glyph_h=score_box, align="right")

    # ── Accuracy: top-right, with the small 'ACCURACY' label above it ──
    acc_box = 42 * A
    acc_right = rc.width - 20 * A
    label_top = 14 * A
    ltex, lw, lh = fr._cached_text("ACCURACY", 15,  # 1080-ref (auto-scaled)
                                   (200, 205, 220, 235))
    fr._draw_external_texture(ltex, x=int(acc_right - lw),
                              y=int(rc.height - label_top - lh), w=lw, h=lh,
                              alpha=0.9)
    acc_cy_gl = rc.height - (label_top + lh + 6 * A + (acc_box * 0.74) / 2.0)
    _argon_number(ctx, f"{disp_acc:.2f}%", x=acc_right, center_y=acc_cy_gl,
                  glyph_h=acc_box, align="right")

    # ── PP: under the accuracy line ('PP' label + value), Argon-amber ──
    if ctx.options.show_pp_counter and s.max_pp > 0:
        pp_box = 30 * A
        pp_top = label_top + lh + 6 * A + acc_box * 0.74 + 8 * A
        pl, plw, plh = fr._cached_text("PP", 13,  # 1080-ref (auto-scaled)
                                       (200, 205, 220, 235))
        fr._draw_external_texture(pl, x=int(acc_right - plw),
                                  y=int(rc.height - pp_top - plh), w=plw, h=plh,
                                  alpha=0.85)
        pp_cy_gl = rc.height - (pp_top + plh + 4 * A + (pp_box * 0.74) / 2.0)
        _argon_number(ctx, f"{int(s.pp)}", x=acc_right, center_y=pp_cy_gl,
                      glyph_h=pp_box, align="right", tint=(1.0, 0.86, 0.55))

    # Active mods (hexagon icons) under the accuracy/pp block, top-right.
    _draw_mod_icons(ctx, acc_right, 110 * A)


def hud(*, element, skin, assets, variables, ctx) -> None:
    """Score + accuracy readout, top-right.

    When the user skin ships the score font (`score-0..9`, etc.) we compose
    the digits from those glyphs — lazer's `LegacyScoreCounter` /
    `LegacyAccuracyCounter`, both `LegacyFont.Score`, anchored top-right,
    overlap = skin.ini `[Fonts] ScoreOverlap` (default 0). When NO user skin
    is selected (Argon default), use lazer's Argon HUD (argon-counter font,
    score wedge top-left). Otherwise fall back to the clean PIL readout."""
    fr = ctx.fr
    s = ctx.scene
    if is_argon_default(ctx, 0) and ctx.has_argon_font():
        _draw_argon_hud(ctx)
        return
    if not ctx.has_score_font():
        # Skin ships no score font → lazer falls through to the default
        # skin's font. We don't have an Argon glyph font bundled, so render a
        # clean PIL readout (no R3D "4K" pill, no 8-digit zero-padding),
        # top-right, with the lazer hexagon mod icons — NOT fr._draw_hud
        # (that's the R3D viewer chrome).
        _draw_fallback_hud(ctx)
        return

    rc = fr.rc
    # Legacy HUD scale = height/768 (same as the scorebar). Score counter
    # scale 0.96, accuracy 0.6×0.96 (lazer LegacyScore/AccuracyCounter).
    hud_scale = rc.height / 768.0
    _gw, score_nh = ctx.atlas.global_native_size("score_0")
    score_nh = score_nh or 70
    right_pad = max(6, int(10 * hud_scale))      # LegacyScoreCounter Margin H=10
    top_pad = max(6, int(9 * hud_scale))
    overlap = ctx.skin_ini.score_overlap if ctx.skin_ini is not None else 0

    # Smoothed during gameplay (counter rolls up), authoritative on results.
    if s.results_opacity > 0:
        display_score, display_acc = s.score, s.accuracy
    else:
        display_score = s.score_smoothed if s.score_smoothed > 0 else s.score
        display_acc = s.accuracy_smoothed

    right_x = rc.width - right_pad
    # `mods_top` is a FROM-TOP y (what _draw_mod_icons/_draw_tl expect).
    mods_top = top_pad

    if ctx.options.show_score:
        score_h = score_nh * hud_scale * 0.96
        score_cy = rc.height - top_pad - score_h / 2.0
        ctx.draw_number(
            f"{int(display_score):d}", x=right_x, center_y=score_cy,
            glyph_h=score_h, overlap_px=overlap, align="right",
        )
        # Accuracy: scale 0.6×0.96, Margin H=17 (7px more indented than score).
        acc_h = score_nh * hud_scale * 0.576
        acc_right_x = rc.width - max(10, int(17 * hud_scale))
        acc_cy = (score_cy - score_h / 2.0) - max(4, int(9 * hud_scale)) - acc_h / 2.0
        ctx.draw_number(
            f"{display_acc:.2f}%", x=acc_right_x, center_y=acc_cy,
            glyph_h=acc_h, overlap_px=overlap, align="right", alpha=0.95,
        )
        # Below the accuracy line: from-top = height − (accuracy bottom GL).
        mods_top = (rc.height - (acc_cy - acc_h / 2.0)) + max(6, int(rc.height * 0.012))

    # Active mods as lazer-style hexagon icons, top-right under the readout.
    _draw_mod_icons(ctx, right_x, mods_top)

    if ctx.options.show_pp_counter and s.max_pp > 0:
        # PP readout, right-aligned below the mod-icon row. (Was `y=ay - ph - 14`
        # with `ay` undefined → NameError whenever a skin with no score font had
        # --show-pp on; positioned relative to `mods_top` instead.)
        pp_tex, pw, ph = fr._cached_text(f"{s.pp:.0f}pp", 44, (255, 220, 140, 255))
        pp_y = int(mods_top + max(28, int(rc.height * 0.045)))
        fr._draw_external_texture(
            pp_tex, x=rc.width - pw - right_pad,
            y=pp_y, w=pw, h=ph, alpha=0.95,
        )


# --- lazer ArgonKeyCounterDisplay geometry, in LAZER px (768-high UI space,
# scale_factor 1.5 already applied). Constants verbatim from STD's
# render/hud.py:255-261 (ARGON_KEY_* / ARGON_KEYS_POS) and :236 (BLUE0).
_AKEY_W, _AKEY_H = 52.5, 45.0        # cell 35×30 × scale_factor 1.5
_AKEY_SPACING = 2.0
_AKEY_LINE_H = 4.5                   # indicator line_height 3 × 1.5
_AKEY_PRESS_OFFSET = 4.0             # indicator drop on press (px)
_AKEY_NAME_H = 15.0                  # name 10 × 1.5
_AKEY_COUNT_H = 21.0                 # count 14 × 1.5
_AKEY_POS = (-60.0, -66.0)           # BottomRight (hitError.Width+10, 66)
_BLUE0 = (0x99 / 255.0, 0xDD / 255.0, 0xFF / 255.0)   # OsuColour.Blue0


# Easings (osu!framework Easing.*) — STD render/hud.py:483-499 verbatim.
def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _ease_out_quart(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 4


def _ease_out_quint(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 5


def key_counter(*, element, skin, assets, variables, ctx) -> None:
    """lazer's ArgonKeyCounterDisplay, bottom-right — 1:1 port of STD's
    _argon_key_overlay (std render/hud.py:2186-2240): one 52.5×45 lazer-px
    cell per column (spacing 2), row anchored BottomRight at (−60, −66)
    just above the song-progress strip. Per cell: the white indicator pill
    drops 4 px on press (60 ms OutQuint, alpha 0.5→1 over 10 ms) and
    returns over 250 ms OutQuart (alpha back to 0.5); the 'B{n}' name
    (lazer's generic KeyCounterActionTrigger label) flashes white while
    held and decays to Blue0 over 200 ms OutQuart; the comma-grouped press
    count sits below, bottom-anchored. Name + count are LEFT-aligned at
    x+3. Press/release edge ages come from scene.key_press_age_ms /
    key_release_age_ms (rising/falling edges of the replay's per-column
    key events — the same sourcing as STD's KeySeries.last_press_at /
    last_release_at, std render/hud.py:719-774)."""
    if not getattr(ctx.options, "show_key_counter", True):
        return
    fr = ctx.fr
    s = ctx.scene
    counts = getattr(s, "key_press_counts", ()) or ()
    if not counts:
        return
    rc = fr.rc
    lk = rc.height / 768.0             # lazer px → device px (std hud.py:184-185)
    ui_w_l = rc.width / lk             # screen width in lazer px
    n = len(counts)
    press_ages = getattr(s, "key_press_age_ms", ()) or ()
    release_ages = getattr(s, "key_release_age_ms", ()) or ()

    total_w = n * _AKEY_W + (n - 1) * _AKEY_SPACING
    right = ui_w_l + _AKEY_POS[0]
    bottom = 768.0 + _AKEY_POS[1]
    x0 = right - total_w
    top = bottom - _AKEY_H
    ind_h = _AKEY_LINE_H

    # White indicator capsule — size-constant per render, baked once.
    pill_w_px = max(2, int(round(_AKEY_W * lk)))
    pill_h_px = max(2, int(round(ind_h * lk)))
    pill = getattr(fr, "_key_pill_tex", None)
    if pill is None or pill[1] != pill_w_px or pill[2] != pill_h_px:
        pill = pill_to_texture(rc.ctx, pill_w_px, pill_h_px)
        fr._key_pill_tex = pill
    ptex = pill[0]

    # _cached_text sizes are 1080-reference (it rescales by height/1080);
    # lazer px → 1080-ref is ×(1080/768) = 1.40625.
    name_size = round(_AKEY_NAME_H * 1080.0 / 768.0)     # 15 → 21
    count_size = round(_AKEY_COUNT_H * 1080.0 / 768.0)   # 21 → 30

    for c in range(n):
        cx0 = x0 + c * (_AKEY_W + _AKEY_SPACING)
        pressed = c < len(s.keys_held) and s.keys_held[c]
        press_age = press_ages[c] if c < len(press_ages) else 99999
        release_age = release_ages[c] if c < len(release_ages) else 99999
        # Indicator pill y-offset + alpha, name whiteness — STD's tween
        # expressions verbatim (std render/hud.py:2212-2224). The 99999
        # "never" sentinel lands on the settled state through the clamped
        # easings, matching STD's age=inf branch.
        if pressed:
            dy = _AKEY_PRESS_OFFSET * _ease_out_quint(press_age / 60.0)
            ind_alpha = 0.5 + 0.5 * _clamp01(press_age / 10.0)
            name_white = _clamp01(press_age / 10.0)
        else:
            p = _ease_out_quart(release_age / 250.0)
            dy = _AKEY_PRESS_OFFSET * (1.0 - p)
            ind_alpha = 1.0 - 0.5 * p
            name_white = 1.0 - _ease_out_quart(release_age / 200.0)
        fr._draw_external_texture(
            ptex, x=int(round(cx0 * lk)),
            y=int(round(rc.height - (top + dy) * lk - pill_h_px)),
            w=pill_w_px, h=pill_h_px, alpha=ind_alpha)
        # Name: Blue0 → white lerp by name_white (quantized to 1/32 steps
        # so the text-texture cache isn't churned every frame; visually
        # identical to the continuous lerp).
        name_white = round(name_white * 32.0) / 32.0
        name_col = tuple(
            int(round((_BLUE0[i] + (1.0 - _BLUE0[i]) * name_white) * 255))
            for i in range(3))
        x_px = int(round((cx0 + 3.0) * lk))
        pad_top = ind_h + _AKEY_PRESS_OFFSET
        # text_to_texture pads 4 px of transparent border around the ink;
        # +4 / −4 below anchor the INK edge, not the texture edge.
        ntex, nw, nh = fr._cached_text(f"B{c + 1}", name_size, (*name_col, 255))
        name_top_px = (top + pad_top + 2.0) * lk
        fr._draw_external_texture(
            ntex, x=x_px, y=int(round(rc.height - name_top_px - nh + 4)),
            w=nw, h=nh, alpha=0.95)
        # Count: white, comma-grouped, ink bottom-anchored at bottom−1.
        ctex, cw, ch = fr._cached_text(
            f"{counts[c]:,}", count_size, (255, 255, 255, 255))
        count_bottom_px = (bottom - 1.0) * lk
        fr._draw_external_texture(
            ctex, x=x_px, y=int(round(rc.height - count_bottom_px - 4)),
            w=cw, h=ch, alpha=0.95)


def top_chrome(*, element, skin, assets, variables, ctx) -> None:
    return  # R3D title chrome; not in lazer


def ur_summary(*, element, skin, assets, variables, ctx) -> None:
    # lazer's hit-error meter (vertical, right side) for the Argon default.
    # The old R3D twin side bars stay stripped for legacy skins.
    if ctx.options.show_ur_bar and is_argon_default(ctx, 0):
        _argon_hit_error(ctx)


def results_overlay(*, element, skin, assets, variables, ctx) -> None:
    s = ctx.scene
    if s.results_opacity > 0 and ctx.options.show_result_screen:
        # Pass the live gameplay ctx so the results numbers render in the
        # argon-counter font (draw_number/_argon_number need the FrameContext).
        ctx.fr._draw_results_overlay(s, ctx)


def watermark(*, element, skin, assets, variables, ctx) -> None:
    if ctx.options.watermark_text:
        ctx.fr._draw_watermark(ctx.options.watermark_text)
