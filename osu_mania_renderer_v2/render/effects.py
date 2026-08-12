"""Effect elements: stage lights, flashlight, combo-break red wash, and the
full-screen fade-to-black. Self-contained (P2 full-decouple); gates + math
mirror FrameRenderer exactly (byte-identical).
"""
from __future__ import annotations

from osu_mania_renderer_v2.render.element_common import (
    STAGE_LIGHT_DURATION_MS,
    stage_light_fps,
    stage_light_tint,
)


def stage_lights(*, element, skin, assets, variables, ctx) -> None:
    if not ctx.options.show_key_overlay:
        return
    scene = ctx.scene
    if not scene.key_press_age_ms:
        return
    atlas = ctx.atlas
    src = atlas.global_source("stage_light")
    # Only draw the column light when the skin ships a REAL mania-stage-light.
    # Many skins (Night05) include a 1×1 transparent placeholder that resolves
    # as "user" but is meaningless — scaling that 1×1 pixel up paints a solid
    # colour block. lazer's LegacyColumnBackground only lights up when a real
    # LightImage exists, so skip placeholders (and never synthesize a light).
    sw, sh = atlas.global_native_size("stage_light")
    use_skin_sprite = src in ("beatmap", "user") and sw * sh > 16
    if not use_skin_sprite:
        return
    sl_frames = atlas.frame_count("stage_light")
    sl_base_idx = atlas.index_of("stage_light")
    for c in range(ctx.key_count):
        if c >= len(scene.key_press_age_ms):
            break
        age = scene.key_press_age_ms[c]
        if age >= STAGE_LIGHT_DURATION_MS:
            continue
        t = age / STAGE_LIGHT_DURATION_MS
        alpha = 0.35 * (1.0 - t)
        tint = stage_light_tint(ctx, c)
        tint_rgba = (tint[0], tint[1], tint[2], alpha)
        if sl_frames > 1:
            fps = stage_light_fps(ctx, sl_frames)
            frame_idx = int(age * fps / 1000.0) % sl_frames
        else:
            frame_idx = 0
        sl_top = ctx.receptor_centre_y_gl + ctx.col_w[c]
        ctx.draw_sprite_idx(
            sl_base_idx + frame_idx,
            ctx.col_x[c], 0, ctx.col_w[c], sl_top, tint_rgba,
        )


def flashlight(*, element, skin, assets, variables, ctx) -> None:
    if ctx.scene.visual_mods.flashlight:
        # v1 approximation: semi-transparent dark vignette over the frame.
        ctx.draw_sprite("bg_vignette", 0, 0, ctx.width, ctx.height, (0, 0, 0, 0.65))


def miss_break_wash(*, element, skin, assets, variables, ctx) -> None:
    s = ctx.scene
    if s.miss_break_age_ms < 300 and s.results_opacity <= 0:
        t = s.miss_break_age_ms / 300.0
        alpha = max(0.0, 0.35 * (1.0 - t))
        ctx.fr._draw_sprite(
            "column_bg", 0, 0, ctx.fr.rc.width, ctx.fr.rc.height,
            (0.95, 0.20, 0.20, alpha),
        )


def break_overlay(*, element, skin, assets, variables, ctx) -> None:
    """lazer's BreakOverlay (gpu/break_overlay.py, the catch d8ccb60
    rollout) — a LATER overlay-component child than HUDOverlay in lazer's
    Player, so it draws above every HUD element and under the
    miss-flash/fade/results/watermark layers. Delegates to the engine's
    stateful lower-level overlay primitive; None on no-break maps and zero GL
    calls outside break windows."""
    if ctx.fr._break_overlay is not None:
        ctx.fr._break_overlay.draw(ctx.scene)


def fade_to_black(*, element, skin, assets, variables, ctx) -> None:
    s = ctx.scene
    if s.fade_to_black > 0:
        ctx.fr._draw_sprite(
            "column_bg", 0, 0, ctx.fr.rc.width, ctx.fr.rc.height,
            (0, 0, 0, s.fade_to_black),
        )


def intro_logo(*, element, skin, assets, variables, ctx) -> None:
    """R3D intro splash (show_logo): the shared 'R' tile + red glow, fading
    out exactly as the first note spawns — parity with std/catch. Delegates
    to the engine's lower-level texture primitive; no-op unless
    options.show_logo is on."""
    # scene.t_ms is the gameplay/map clock the splash envelope is authored
    # against (fade-out ends at first_note − approach). ctx.t_ms is the raw
    # VIDEO clock, which on the pre-roll leads map-time by logo_preroll_ms —
    # using it makes the splash fade ~preroll early on the wiki/prod path.
    ctx.fr.draw_logo_splash(ctx.scene.t_ms)
