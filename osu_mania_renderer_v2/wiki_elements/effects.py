"""Effect elements: stage lights, flashlight, combo-break red wash, and the
full-screen fade-to-black. Self-contained (P2 full-decouple); gates + math
mirror FrameRenderer exactly (byte-identical).
"""
from __future__ import annotations


def stage_lights(*, element, skin, assets, variables, ctx) -> None:
    if not ctx.fr._is_argon_default():
        return
    if not ctx.options.show_key_overlay:
        return
    # The shared renderer consumes exact replay-derived release ages, keeping
    # sub-frame taps, direct seeks, and out-of-order samples byte-identical.
    ctx.fr._draw_stage_lights(ctx.scene)


def legacy_hit_lighting(*, element, skin, assets, variables, ctx) -> None:
    """Stable hit effects: all keys/foreground below, legacy judgement above."""
    # Drawn by the shared per-stage legacy gameplay compositor.
    return


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
    miss-flash/fade/results/watermark layers, mirroring
    FrameRenderer.draw(). Delegates to the engine so the legacy and wiki
    paths draw identical pixels; None on no-break maps and zero GL calls
    outside break windows."""
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
    to the engine so the legacy and wiki paths draw identical pixels; no-op
    unless options.show_logo is on."""
    ctx.fr.draw_logo_splash(ctx.t_ms)
