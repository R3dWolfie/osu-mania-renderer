"""Stage elements: background, stage decorations, column lanes.

Migration status (P2): elements are moving from thin delegation to
self-contained render_fns that draw through `ctx` (FrameContext) using the
atlas for image truth and SkinPair for skin.ini variables. Each migrated
element preserves byte-identical output (the parity test guards it).
"""
from __future__ import annotations

from osu_mania_renderer_v2.gpu.atlas import column_variant
from osu_mania_renderer_v2.gpu.renderer import LAZER_DEFAULT_COLUMN_LINE_WIDTH_REF
from osu_mania_renderer_v2.argon import argon_accent, is_argon_default

# Skins that ship 1×1 transparent placeholders register as "user" source
# but contribute zero visible art; reject them by native pixel area.
_PLACEHOLDER_THRESHOLD_PX2 = 100


def background(*, element, skin, assets, variables, ctx) -> None:
    # Background is the beatmap image (loaded once via fr.set_background),
    # not a skin element — keep delegating to the engine's bg pass.
    ctx.fr._draw_background(ctx.scene)


def stage_decorations(*, element, skin, assets, variables, ctx) -> None:
    """stage_left/right/bottom/hint + the side/column background dims.
    Ported from FrameRenderer._draw_stage_decorations: atlas image truth +
    precomputed playfield geometry; no skin.ini variables read here (dims
    are fixed conventions, sprite aspects come from the atlas)."""
    atlas = ctx.atlas
    h = ctx.height
    w = ctx.width
    region_w = int(h * 4.0 / 3.0)
    region_x0 = (w - region_w) // 2

    # Side-panel dim: translucent overlay outside the playfield column band.
    SIDE_DIM = (0.0, 0.0, 0.0, 0.55)
    if ctx.pf_x > 0:
        ctx.draw_sprite("column_bg", 0, 0, ctx.pf_x, h, SIDE_DIM)
    right_x = ctx.pf_x + ctx.pf_w
    if right_x < w:
        ctx.draw_sprite("column_bg", right_x, 0, w - right_x, h, SIDE_DIM)

    # Column-area dim: only when the skin ships no meaningful stage chrome.
    def _has_meaningful(name: str) -> bool:
        src = atlas.global_source(name)
        if src not in ("beatmap", "user"):
            return False
        sw, sh = atlas.global_native_size(name)
        return sw * sh > _PLACEHOLDER_THRESHOLD_PX2

    if not (_has_meaningful("stage_left") or _has_meaningful("stage_right")):
        COL_DIM = (0.0, 0.0, 0.0, 0.55)
        ctx.draw_sprite("column_bg", ctx.pf_x, 0, ctx.pf_w, h, COL_DIM)

    # Stage left/right (mania-stage-left/right) — ONE general rule for both a
    # thin edge border (Vio 4×768) and a full background panel (Night05
    # 1200×770): width = native_w(design) × (height/768), full stage height,
    # with stage-left's RIGHT edge at the playfield-left and stage-right's
    # LEFT edge at the playfield-right. Drawn behind the columns (background).
    tex_scale = h / 768.0
    if atlas.global_source("stage_left") in ("beatmap", "user"):
        lw, _lh = atlas.global_native_size("stage_left")
        sw = max(1, int(lw * tex_scale))
        ctx.draw_direct("stage_left", int(ctx.pf_x - sw), 0, sw, h, (1, 1, 1, 1))
    if atlas.global_source("stage_right") in ("beatmap", "user"):
        rw, _rh = atlas.global_native_size("stage_right")
        sw = max(1, int(rw * tex_scale))
        ctx.draw_direct("stage_right", int(ctx.pf_x + ctx.pf_w), 0, sw, h, (1, 1, 1, 1))

    # mania-stage-hint (LegacyHitTarget): centred on the hit line, width =
    # playfield, height = native × 1.44225 (lazer's 0.9 × 1.6025 scale).
    # This is BEHIND the notes/keys, so it stays in stage_decorations.
    # mania-stage-bottom is a FOREGROUND element (LegacyStageForeground),
    # drawn on top of the keys — see stage_foreground(), called late.
    rec_y = ctx.receptor_centre_y_gl
    tex_scale = h / 768.0
    if atlas.global_source("hit_light") in ("user", "beatmap"):
        _hw, hnh = atlas.global_native_size("hit_light")
        hint_h = max(2, int((hnh or 10) * 1.44225 * tex_scale))
        ctx.draw_sprite("hit_light", ctx.pf_x, rec_y - hint_h // 2,
                        ctx.pf_w, hint_h, (1, 1, 1, 1))


def stage_foreground(ctx) -> None:
    """mania-stage-bottom (lazer LegacyStageForeground) — drawn ON TOP of the
    notes and keys so it covers the lower portion of tall key images (that's
    why lazer's keys look short). Native sprite px × (height/768), centred on
    the playfield, anchored to the bottom edge."""
    atlas = ctx.atlas
    if atlas.global_source("playfield_frame") not in ("user", "beatmap"):
        return
    nw, nh = atlas.global_native_size("playfield_frame")
    if nw <= 0:
        return
    h = ctx.height
    # LegacyStageForeground applies POSITION_SCALE_FACTOR (×1.6), so
    # mania-stage-bottom renders at native px × (height/480), not /768 —
    # measured: Vio 50px → 113px at 1125h (×2.26). (The keys, by contrast,
    # are /768.)
    tex_scale = h / 480.0
    center_x = ctx.pf_x + ctx.pf_w / 2.0
    sb_w, sb_h = nw * tex_scale, nh * tex_scale
    sb_y_gl = (h - sb_h) if ctx.upside_down else 0
    ctx.draw_direct("playfield_frame", int(center_x - sb_w / 2),
                    int(sb_y_gl), int(sb_w), int(sb_h), (1, 1, 1, 1))


def columns(*, element, skin, assets, variables, ctx) -> None:
    """Per-column lane backgrounds + dividers. Ported from
    FrameRenderer._draw_columns: per-column Colour{N} from the parsed
    [Mania] section (the resolved skin.ini) else the default alternating
    palette; kiai lifts the tint; ColumnLineWidth/ColourColumnLine drive
    dividers. Byte-identical to the engine pass."""
    scene = ctx.scene
    h = ctx.height
    key_count = ctx.key_count
    col_w_uniform = ctx.col_w_uniform
    kiai_boost = 0.04 if (scene is not None and scene.is_kiai) else 0.0
    section = ctx.mania_section
    for c in range(key_count):
        # ── Argon default column: dark accent base + bottom-glow ──
        if is_argon_default(ctx, c):
            ac = argon_accent(c, key_count)
            r, g, b = ac[0] / 255, ac[1] / 255, ac[2] / 255
            # lazer ArgonColumnBackground: base = accent.Darken(3).Opacity(0.8)
            # (Darken(3) = accent/4). Then an ADDITIVE half-height glow at the
            # receptor side, gradient bright(accent.Opacity 0.6) → dim(0).
            ctx.draw_sprite("column_bg", ctx.col_x[c], 0, ctx.col_w[c], h,
                            (r * 0.25, g * 0.25, b * 0.25, 0.8))
            # lazer shows a LIGHTER region just above the hit line — a SMOOTH
            # gradient brightening (ArgonColumnBackground glow), bright at the
            # receptor fading up. The desaturated tint (mostly grey with ~28%
            # accent) is right, but it must be a gradient, not a flat solid
            # block — the hard top edge of the old block read as an ugly "chin",
            # especially under HD where the notes that hid it are gone.
            cw = ctx.col_w[c]
            x0 = ctx.col_x[c]
            rec_y = ctx.receptor_centre_y_gl
            grey = 0.50 + (0.10 if (scene is not None and scene.is_kiai) else 0.0)
            br = (0.28 * r + 0.72 * grey,
                  0.28 * g + 0.72 * grey,
                  0.28 * b + 0.72 * grey)
            a0 = 0.82
            # Single gradient glow from the hit line up — bright at rec_y,
            # fading to nothing (argon_col_glow is a bottom-bright gradient).
            glow_h = int(cw * 1.25)
            ctx.draw_sprite("argon_col_glow", x0, rec_y, cw, glow_h, (*br, a0))
            continue

        skin_colour = None
        if section is not None:
            skin_colour = section.colour.get(c + 1)
            if skin_colour is None:
                skin_colour = section.colour.get(c)
        if skin_colour is not None:
            sr, sg, sb, sa = skin_colour
            r, g, b, a = sr / 255, sg / 255, sb / 255, sa / 255
        else:
            variant = column_variant(c, key_count)
            if variant == "outer":
                r, g, b, a = 0.04, 0.04, 0.09, 0.55
            elif variant == "center":
                r, g, b, a = 0.07, 0.06, 0.12, 0.55
            else:
                r, g, b, a = 0.05, 0.05, 0.11, 0.45
        ctx.draw_sprite("column_bg", ctx.col_x[c], 0, ctx.col_w[c], h,
                        (r + kiai_boost, g + kiai_boost,
                         b + kiai_boost * 1.5, a))

    # Column dividers + outer borders. lazer's LegacyManiaSkinConfiguration
    # fills ColumnLineWidth[keys+1] with 2 by default, then overrides per
    # index from the skin.ini CSV — so a legacy skin shows hairline dividers
    # even with a single (or no) ColumnLineWidth value. The Argon default
    # has no stable dividers, so this is skipped there.
    if not is_argon_default(ctx, 0):
        line_widths = section.column_line_width if section else ()
        if section is not None and section.colour_column_line is not None:
            sr, sg, sb, sa = section.colour_column_line
            line_tint = (sr / 255, sg / 255, sb / 255, sa / 255)
        else:
            line_tint = (1.0, 1.0, 1.0, 0.9)

        def _divider_x(idx: int) -> int:
            if idx >= key_count:
                return ctx.col_x[-1] + ctx.col_w[-1]
            return ctx.col_x[idx]

        # ColumnLineWidth is the ONE geometry value lazer does NOT ×1.6
        # (LegacyManiaSkinDecoder parseArrayValue applyScaleFactor=false), so
        # it scales straight by height/768, not height/480 like ColumnWidth.
        line_scale = h / 768.0
        for c in range(key_count + 1):
            lw_ref = (line_widths[c] if c < len(line_widths)
                      else LAZER_DEFAULT_COLUMN_LINE_WIDTH_REF)
            if lw_ref <= 0:
                continue
            lw = max(1, int(round(lw_ref * line_scale)))
            x_centre = _divider_x(c)
            ctx.draw_sprite("column_bg", x_centre - lw // 2, 0, lw, h, line_tint)
    # (mania-stage-left/right are drawn in stage_decorations as background
    # panels — see there. The same rule covers thin borders and full panels.)
