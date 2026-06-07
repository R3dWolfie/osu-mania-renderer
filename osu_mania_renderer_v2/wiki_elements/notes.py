"""Note-field elements: receptors (under/over notes per KeysUnderNotes),
the scrolling notes (with the HD/FI flush window), and combo+judgment text.

receptors: self-contained (P2 full-decouple). notes/combo still delegate
pending their own decouple. All byte-identical to FrameRenderer.
"""
from __future__ import annotations

from osu_mania_renderer_v2.gpu.atlas import column_variant
from osu_mania_renderer_v2.wiki_elements._common import (
    HIT_LIGHT_DURATION_MS,
    JUDGMENT_LIGHT,
    RECEPTOR_HEIGHT_REL_COL,
    argon_accent,
    is_argon_default,
    note_anim_fps,
    stage_light_fps,
    stage_light_tint,
)

# osu!stable default mania note palette by column variant.
_NOTE_TINTS = {
    "outer":  (240 / 255, 240 / 255, 245 / 255, 1.0),
    "inner":  (70 / 255, 165 / 255, 255 / 255, 1.0),
    "center": (255 / 255, 210 / 255, 90 / 255, 1.0),
}


def _keys_under_notes(ctx) -> bool:
    ms = ctx.mania_section
    return bool(ms is not None and ms.keys_under_notes)


def _receptors(ctx) -> None:
    """Per-column receptor (on/off) + lighting_l (held) + hit lighting.
    Ported verbatim from FrameRenderer._draw_receptors."""
    scene = ctx.scene
    atlas = ctx.atlas
    centre_y = ctx.receptor_centre_y_gl
    for c in range(ctx.key_count):
        x0 = ctx.col_x[c]
        cw = ctx.col_w[c]
        held = scene.keys_held[c]

        # ── Argon default key area (lazer ArgonKeyArea) ──
        # Stage-space units (height/768). From the hit target DOWN to the
        # screen bottom: a grey hit-target line at the top, the hollow white
        # key pill 30u below it, and the 3 accent dots ~30u above the bottom.
        # On press the whole key area flashes a lightened accent and the line
        # goes white. Mirrors ArgonKeyArea exactly (icon_circle_size=8,
        # icon_spacing=7, top icon 22×14, hit line = CORNER_RADIUS*2 = 6.8u).
        if is_argon_default(ctx, c):
            ac = argon_accent(c, ctx.key_count)
            r, g, b = ac[0] / 255, ac[1] / 255, ac[2] / 255
            u = ctx.height / 768.0
            if held:
                # Background box: accent.Lighten(0.8), filling the key area
                # (hit target → bottom), masked corners (approx. plain rect).
                lr = r + (1.0 - r) * 0.8
                lg = g + (1.0 - g) * 0.8
                lb = b + (1.0 - b) * 0.8
                ctx.draw_sprite("column_bg", x0, 0, cw, int(centre_y),
                                (lr, lg, lb, 0.5))
            # Hit-target line (Circle, height CORNER_RADIUS*2): Gray(196) at
            # rest, white when held.
            hl_h = max(3, int(round(6.8 * u)))
            hl_tint = (1.0, 1.0, 1.0, 1.0) if held else (0.77, 0.77, 0.77, 0.95)
            ctx.draw_sprite("column_bg", x0, int(centre_y - hl_h / 2), cw, hl_h,
                            hl_tint)
            # Top icon: hollow white pill 22u×14u, 30u below the hit line;
            # shrinks to 0.9 on press.
            ps = 0.9 if held else 1.0
            pw, ph = int(round(22 * u * ps)), int(round(14 * u * ps))
            pcy = int(centre_y - 30 * u)
            ctx.draw_sprite("argon_key_pill", x0 + (cw - pw) // 2,
                            pcy - ph // 2, pw, ph, (1, 1, 1, 1.0))
            # Bottom icon: the 3 accent dots (22u×17u), near the bottom of the
            # key area (lazer leaves a big gap below the pill); white on press.
            dw, dh = int(round(22 * u)), int(round(17 * u))
            dcy = int(round(24 * u))
            dtint = (1.0, 1.0, 1.0, 1.0) if held else (r, g, b, 1.0)
            ctx.draw_sprite("argon_key_dots", x0 + (cw - dw) // 2,
                            dcy - dh // 2, dw, dh, dtint)
            # hit-light flash on the line.
            if c < len(scene.hit_light_age_ms):
                age = scene.hit_light_age_ms[c]
                jud = scene.hit_light_judgment[c] if c < len(scene.hit_light_judgment) else ""
                if 0 <= age < HIT_LIGHT_DURATION_MS and jud in JUDGMENT_LIGHT:
                    fade = 1.0 - (age / HIT_LIGHT_DURATION_MS)
                    glow_h = int(cw * 0.9)
                    ctx.draw_sprite("argon_col_glow", x0, centre_y - glow_h, cw, glow_h,
                                    (r, g, b, 0.6 * fade))
            continue

        # Legacy key area (lazer LegacyKeyArea): the KeyImage is stretched in
        # X to the column width, but its HEIGHT is the texture's native pixel
        # height in osu-pixels (NOT aspect-scaled to the column), anchored to
        # the BOTTOM edge of the stage. A tall key PNG therefore stays its own
        # height pinned to the bottom — it never balloons to a fraction of the
        # column. Pressing just swaps key→keyD (no offset/scale bump).
        kind = "receptor_on" if held else "receptor_off"
        slot_idx = atlas.column_slot_index(kind, c)
        nw, nh = atlas.column_native_size(kind, c)
        # Texture native px live in lazer's 768-internal space (osu 480 ×
        # POSITION_SCALE_FACTOR 1.6), so they scale to screen by height/768 —
        # NOT height/480 (which is for skin.ini osu-space values like
        # ColumnWidth). Verified: Vio key 107px → 157px at 1125h (×1.465).
        tex_scale = ctx.height / 768.0
        if nh > 0:
            rec_h = max(1, int(round(nh * tex_scale)))
        else:
            asp = atlas.column_aspect(kind, c)
            rec_h = max(1, int(cw / asp)) if asp > 0 else int(cw * RECEPTOR_HEIGHT_REL_COL)
        # Bottom-anchored (top-anchored when the stage is flipped upside-down).
        rec_y = (ctx.height - rec_h) if ctx.upside_down else 0
        ctx.draw_sprite_idx(slot_idx, x0, rec_y, cw, rec_h, (1, 1, 1, 1))

        # Lighting anchors at the HIT centre (cw × cw centred on centre_y).
        hit_h = cw
        hit_y = centre_y - cw // 2

        # lighting_l — sustained flash while held (skin-authored only).
        if held and atlas.global_source("lighting_l") in ("beatmap", "user"):
            ll_base = atlas.index_of("lighting_l")
            ll_frames = atlas.frame_count("lighting_l")
            if ll_frames > 1:
                fps = stage_light_fps(ctx, ll_frames)
                age_for_l = scene.key_press_age_ms[c] if c < len(scene.key_press_age_ms) else 0
                frame_idx = int(age_for_l * fps / 1000.0) % ll_frames
            else:
                frame_idx = 0
            tint = stage_light_tint(ctx, c)
            ctx.draw_sprite_idx(
                ll_base + frame_idx, x0, hit_y, cw, hit_h,
                (tint[0], tint[1], tint[2], 0.8),
            )

        # Hit lighting: colour flash growing outward from the receptor.
        if c < len(scene.hit_light_age_ms):
            age = scene.hit_light_age_ms[c]
            jud = scene.hit_light_judgment[c] if c < len(scene.hit_light_judgment) else ""
            if 0 <= age < HIT_LIGHT_DURATION_MS and jud in JUDGMENT_LIGHT:
                r, g, b = JUDGMENT_LIGHT[jud]
                fade = 1.0 - (age / HIT_LIGHT_DURATION_MS)
                scale = 1.4 + 0.3 * (1.0 - fade)
                lw = int(cw * scale)
                lh = int(hit_h * scale)
                ln_src = atlas.global_source("lighting_n")
                if ln_src in ("beatmap", "user"):
                    ln_base = atlas.index_of("lighting_n")
                    ln_frames = atlas.frame_count("lighting_n")
                    if ln_frames > 1:
                        f = min(int(age * 60.0 / 1000.0), ln_frames - 1)
                    else:
                        f = 0
                    ctx.draw_sprite_idx(
                        ln_base + f,
                        x0 + (cw - lw) // 2, hit_y + (hit_h - lh) // 2,
                        lw, lh, (r / 255, g / 255, b / 255, 0.7 * fade),
                    )
                else:
                    ctx.draw_sprite(
                        "note_circle",
                        x0 + (cw - lw) // 2, hit_y + (hit_h - lh) // 2,
                        lw, lh, (r / 255, g / 255, b / 255, 0.55 * fade),
                    )


def receptors_under(*, element, skin, assets, variables, ctx) -> None:
    # Drawn before notes only when the skin sets KeysUnderNotes.
    if _keys_under_notes(ctx):
        _receptors(ctx)


def _draw_notes_body(ctx) -> None:
    """Tap + hold notes (head/body/tail) with NoteBodyStyle tiling, skin
    per-column sprites + animation, and the falling tap trail. Ported
    verbatim from FrameRenderer._draw_notes."""
    scene = ctx.scene
    atlas = ctx.atlas
    key_count = ctx.key_count
    h = ctx.height
    receptor_y = ctx.receptor_centre_y_gl
    upside_down = ctx.upside_down

    def to_screen_y(yf: float) -> int:
        if upside_down:
            return int(yf * receptor_y)
        return int(receptor_y + (1.0 - yf) * (h - receptor_y))

    use_skin_notes = atlas.has_skin_notes()
    world_ms = scene.t_ms

    def _animated_idx(kind: str, col: int, note_time_ms: int = 0) -> int:
        base = atlas.column_slot_index(kind, col)
        n_frames = atlas.column_frame_count(kind, col)
        if n_frames <= 1:
            return base
        fps = note_anim_fps(ctx, n_frames)
        elapsed_ms = world_ms - note_time_ms
        return base + int(elapsed_ms * fps / 1000.0) % n_frames

    nb_asp = atlas.global_aspect("argon_note_body") or 1.667
    for n in scene.visible_notes:
        x0 = ctx.col_x[n.column]
        cw = ctx.col_w[n.column]

        # ── Argon default (no user skin for this column) ──
        if is_argon_default(ctx, n.column):
            ac = argon_accent(n.column, key_count)
            arr = (ac[0] / 255, ac[1] / 255, ac[2] / 255, 1.0)
            nh = max(6, int(cw / nb_asp))
            if n.is_hold:
                y_head = to_screen_y(n.head_y_fraction)
                y_tail = to_screen_y(n.tail_y_fraction)
                body_top = min(y_head, y_tail)
                body_h = abs(y_head - y_tail)
                inset = cw // 8
                ctx.draw_sprite("column_bg", x0 + inset, body_top,
                                cw - 2 * inset, body_h, (arr[0], arr[1], arr[2], 0.45))
                for yy in (y_tail, y_head):  # tail under head
                    ctx.draw_sprite("argon_note_body", x0, yy - nh // 2, cw, nh, arr)
                    ctx.draw_sprite("argon_note_glyph", x0, yy - nh // 2, cw, nh, (1, 1, 1, 1))
            else:
                y = to_screen_y(n.y_fraction)
                ctx.draw_sprite("argon_note_body", x0, y - nh // 2, cw, nh, arr)
                ctx.draw_sprite("argon_note_glyph", x0, y - nh // 2, cw, nh, (1, 1, 1, 1))
            continue

        tint = _NOTE_TINTS[column_variant(n.column, key_count)]
        col_has_skin = use_skin_notes and atlas.has_skin_note(n.column)
        if col_has_skin:
            note_asp = atlas.column_aspect("note_tap", n.column)
            local_note_h = max(1, int(cw / note_asp)) if note_asp > 0 else cw
        else:
            local_note_h = cw
        if col_has_skin:
            head_asp = atlas.column_aspect("note_hold_head", n.column)
            head_h = max(1, int(cw / head_asp)) if head_asp > 0 else cw
            tail_asp = atlas.column_aspect("note_hold_tail", n.column)
            tail_h = max(1, int(cw / tail_asp)) if tail_asp > 0 else cw
        else:
            head_h = cw
            tail_h = cw
        col_has_skin_hold = col_has_skin and atlas.has_skin_hold(n.column)
        if n.is_hold:
            y_head = to_screen_y(n.head_y_fraction)
            y_tail = to_screen_y(n.tail_y_fraction)
            body_top = min(y_head, y_tail)
            body_h = abs(y_head - y_tail)
            if col_has_skin_hold:
                body_idx = _animated_idx("note_hold_body", n.column, n.time_ms)
                head_idx = _animated_idx("note_hold_head", n.column, n.time_ms)
                tail_idx = _animated_idx("note_hold_tail", n.column, n.time_ms)
                body_style = (
                    ctx.mania_section.note_body_style
                    if ctx.mania_section is not None
                    and ctx.mania_section.note_body_style is not None
                    else 0
                )
                if body_style != 0:
                    body_aspect = atlas.column_aspect("note_hold_body", n.column)
                    tile_h = (
                        max(1, int(round(cw / body_aspect)))
                        if body_aspect > 0 else cw
                    )
                    seg_y = body_top
                    while seg_y < body_top + body_h:
                        seg_h = min(tile_h, body_top + body_h - seg_y)
                        ctx.draw_sprite_idx(body_idx, x0, seg_y, cw, seg_h, (1, 1, 1, 1))
                        seg_y += tile_h
                else:
                    ctx.draw_sprite_idx(body_idx, x0, body_top, cw, body_h, (1, 1, 1, 1))
                ctx.draw_sprite_idx(head_idx, x0, y_head - head_h // 2, cw, head_h, (1, 1, 1, 1))
                ctx.draw_sprite_idx(tail_idx, x0, y_tail - tail_h // 2, cw, tail_h, (1, 1, 1, 1))
            else:
                pad = cw // 6
                ctx.draw_sprite("column_bg", x0 + pad, body_top, cw - 2 * pad, body_h, tint)
                ctx.draw_sprite("note_circle", x0, y_head - local_note_h // 2, cw, local_note_h, tint)
                ctx.draw_sprite("note_circle", x0, y_tail - local_note_h // 2, cw, local_note_h, tint)
        else:
            y = to_screen_y(n.y_fraction)
            if col_has_skin:
                tap_idx = _animated_idx("note_tap", n.column, n.time_ms)
                trail_step = max(4, local_note_h // 4)
                for k in (2, 1):
                    ghost_y = y + k * trail_step
                    ghost_alpha = 0.20 / k
                    ctx.draw_sprite_idx(
                        tap_idx, x0, ghost_y - local_note_h // 2,
                        cw, local_note_h, (1, 1, 1, ghost_alpha),
                    )
                ctx.draw_sprite_idx(tap_idx, x0, y - local_note_h // 2, cw, local_note_h, (1, 1, 1, 1))
            else:
                trail_step = max(4, local_note_h // 4)
                for k in (2, 1):
                    ghost_y = y + k * trail_step
                    ghost_alpha = 0.20 / k
                    ghost_tint = (tint[0], tint[1], tint[2], ghost_alpha)
                    ctx.draw_sprite("note_circle", x0, ghost_y - local_note_h // 2, cw, local_note_h, ghost_tint)
                ctx.draw_sprite("note_circle", x0, y - local_note_h // 2, cw, local_note_h, tint)


def notes(*, element, skin, assets, variables, ctx) -> None:
    # HD/FI uniforms apply only to scrolling notes — flush around them so
    # queued sprites aren't drawn with the wrong HD state (mirrors draw()).
    scene = ctx.scene
    ctx.flush()
    ctx.set_note_fx(scene.visual_mods.hidden, scene.visual_mods.fade_in)
    _draw_notes_body(ctx)
    ctx.flush()
    ctx.set_note_fx(False, False)


# Argon judgement text + colour by judged value (lazer ArgonJudgementPiece).
_ARGON_JUDGE_TEXT = {
    "geki": ("PERFECT", (108, 192, 255)),
    "300":  ("GREAT",   (108, 192, 255)),
    "katu": ("GOOD",    (100, 220, 130)),
    "100":  ("OK",      (240, 220, 90)),
    "50":   ("MEH",     (240, 160, 80)),
    "miss": ("MISS",    (237, 73, 92)),
}


def _argon_combo_and_judgment(ctx) -> None:
    """Argon default combo + judgement over the playfield. Combo in the
    argon-counter font (with wireframe backing), the judgement as Argon
    text below it. Positioned at the legacy stage combo line (ComboPosition,
    default 111 in 480-space) — where lazer's Argon mania combo sits."""
    from osu_mania_renderer_v2.wiki_elements.hud import _argon_number, ARGON_OVERLAP
    fr = ctx.fr
    s = ctx.scene
    h = fr.rc.height
    px = h / 480.0
    A = h / 1080.0
    center_x = fr.pf_x + fr.pf_w / 2.0

    section = ctx.mania_section
    combo_pos = (section.combo_position
                 if section is not None and section.combo_position is not None
                 else 111.0)
    centre_y_gl = h - combo_pos * px

    # Judgement text near mid-playfield (lazer's Argon mania judgement floats
    # around the centre, well below the combo line — ~56% from the top). lazer
    # spaces the letters out, so render per-character with letter tracking.
    if s.active_judgments:
        j = s.active_judgments[-1]
        info = _ARGON_JUDGE_TEXT.get(j.judgment)
        if info is not None:
            text, col = info
            alpha = max(0.0, 1.0 - j.age_ms / 500.0)
            jpop = 1.0 + 0.25 * max(0.0, 1.0 - j.age_ms / 120.0)
            jh = int(40 * jpop)        # 1080-ref; _cached_text scales it
            tracking = int(jh * 0.40)
            glyphs = [fr._cached_text(ch, jh, (*col, 255)) for ch in text]
            total = sum(g[1] for g in glyphs) + tracking * (len(glyphs) - 1)
            jcy_gl = h * 0.44
            gx = center_x - total / 2.0
            for gtex, gw, ghh in glyphs:
                fr._draw_external_texture(gtex, x=int(gx),
                                          y=int(jcy_gl - ghh / 2), w=gw, h=ghh,
                                          alpha=alpha)
                gx += gw + tracking

    if s.combo <= 0:
        return
    pop = 1.0
    if s.combo_age_ms < 180:
        t = s.combo_age_ms / 180.0
        pop = 1.0 + 0.14 * (1.0 - t) ** 2
    # ArgonComboCounter Scale 1.3; box height in 1080-space ≈ 72px.
    combo_box = 72 * A * pop
    _argon_number(ctx, f"{s.combo}", x=center_x, center_y=centre_y_gl,
                  glyph_h=combo_box, align="center")


def combo_and_judgment(*, element, skin, assets, variables, ctx) -> None:
    """Judgment burst (atlas sprite) + centred combo counter.

    When the skin ships the score font, the combo is composed from those
    glyphs — lazer's mania combo uses `LegacyFont.Combo`, which defaults to
    the `score` prefix; ComboOverlap default 0; centred above the playfield.
    Otherwise the legacy PIL combo (with its colour tiers / pop) is used.
    """
    fr = ctx.fr
    s = ctx.scene
    if is_argon_default(ctx, 0) and ctx.has_argon_font():
        _argon_combo_and_judgment(ctx)
        return
    if not ctx.has_score_font():
        fr._draw_combo_and_judgment(s)
        return

    atlas = ctx.atlas
    h = fr.rc.height
    px = h / 480.0                       # osu-space positions → render px
    tex_scale = h / 768.0                # texture native px → render px
    center_x = fr.pf_x + fr.pf_w / 2.0

    # Combo position (ComboPosition, osu Y from top, 480-space; default 111).
    section = ctx.mania_section
    combo_pos = (section.combo_position
                 if section is not None and section.combo_position is not None
                 else 111.0)
    centre_y_gl = h - combo_pos * px
    # Combo glyph height: the mania combo counter lives in the stage space, so
    # its glyphs get the ×1.6 POSITION_SCALE_FACTOR like positions — native px
    # × (height/480), NOT the plain texture scale. Measured: Night05 combo
    # native 37px → 79px at 1076h (×2.14 ≈ height/480).
    _gw, combo_native_h = atlas.global_native_size("combo_0")
    combo_h = (combo_native_h or 70) * px
    # Sanity clamp — keep the counter readable and above the hit line even when
    # a skin ships an oversized combo font or an extreme ComboPosition (e.g.
    # ComboPosition:460 drops it onto the receptors). Cap height to ~one column
    # (and ≤11% of frame), then lift it so it never sits below the hit line.
    combo_h = min(combo_h, ctx.col_w_uniform * 1.0, h * 0.11)
    centre_y_gl = max(centre_y_gl, ctx.receptor_centre_y_gl + combo_h)

    # ── Judgement burst — drawn just BELOW the combo (lazer mania stacks the
    # combo above the hit-result). Native sprite px × texture scale, fading
    # out over 500ms; animation frames at 60fps.
    if s.active_judgments:
        j = s.active_judgments[-1]
        name = f"judgment_{j.judgment}"
        if atlas.global_source(name) in ("user", "beatmap", "bundle"):
            nw, nh = atlas.global_native_size(name)
            if nw > 0:
                base = atlas.index_of(name)
                fc = atlas.frame_count(name)
                idx = base + (min(int(j.age_ms * 60.0 / 1000.0), fc - 1) if fc > 1 else 0)
                alpha = max(0.0, 1.0 - j.age_ms / 500.0)
                # Judgement is a stage-space element (like the combo), so it
                # gets the ×1.6 POSITION_SCALE_FACTOR → native px × (height/480),
                # not the plain texture scale. (Aspect is the sprite's own —
                # Night05's mania-hit300-0 is a wide 256×72 animation frame.)
                jw, jh = nw * px, nh * px
                # Centre below the combo: combo bottom − gap − half judgment.
                jcy = centre_y_gl - combo_h / 2.0 - jh / 2.0 - max(4, int(h * 0.01))
                fr._draw_sprite_idx(
                    idx, int(center_x - jw / 2), int(jcy - jh / 2),
                    int(jw), int(jh), (1, 1, 1, alpha),
                )

    if s.combo <= 0:
        return
    # Pop animation: digits scale up briefly on each increment, settle back.
    pop = 1.0
    if s.combo_age_ms < 180:
        t = s.combo_age_ms / 180.0
        pop = 1.0 + 0.18 * (1.0 - t) ** 2
    overlap = ctx.skin_ini.combo_overlap if ctx.skin_ini is not None else 0
    ctx.draw_number(
        f"{s.combo}", x=center_x, center_y=centre_y_gl,
        glyph_h=combo_h * pop, overlap_px=overlap, align="center", alpha=0.95,
        font="combo",
    )


def receptors_over(*, element, skin, assets, variables, ctx) -> None:
    # Default: keys on top of notes.
    if not _keys_under_notes(ctx):
        _receptors(ctx)
    # mania-stage-bottom is a foreground element: drawn ON TOP of the keys so
    # it covers their lower portion (lazer's keys look short for this reason).
    from osu_mania_renderer_v2.wiki_elements.stage import stage_foreground
    stage_foreground(ctx)
