"""Canonical render-pipeline element registry.

Importing this module populates ``render.compositor.RENDER_ORDER`` and
``ELEMENTS``. This list is the sole painter-order authority. Elements use the
shared ``FrameRenderer`` only for the lower-level GPU primitives they need.
"""
from __future__ import annotations

from osu_mania_renderer_v2.hud import elements as hud
from osu_mania_renderer_v2.render import effects, notes, stage
from osu_mania_renderer_v2.render.compositor import ElementSpec, register_element

# (name, render_fn) in canonical painter order.
_ORDER = [
    ("background", stage.background),
    ("stage_decorations", stage.stage_decorations),
    ("columns", stage.columns),
    ("stage_lights", effects.stage_lights),
    ("receptors_under", notes.receptors_under),
    ("notes", notes.notes),
    ("combo_and_judgment", notes.combo_and_judgment),
    ("receptors_over", notes.receptors_over),
    ("hit_error_popups", hud.hit_error_popups),
    ("hit_strip", hud.hit_strip),
    ("progress_bar", hud.progress_bar),
    ("hp_bar", hud.hp_bar),
    ("banner", hud.banner),
    ("hud", hud.hud),
    ("key_counter", hud.key_counter),
    ("top_chrome", hud.top_chrome),
    ("flashlight", effects.flashlight),
    ("ur_summary", hud.ur_summary),
    # lazer BreakOverlay: above every HUD element, below the wash/fade/
    # results/watermark layers (Player.createOverlayComponents z-order).
    ("break_overlay", effects.break_overlay),
    ("miss_break_wash", effects.miss_break_wash),
    ("fade_to_black", effects.fade_to_black),
    # R3D intro splash — topmost intro element (over the start fade), gone
    # before results ever fade in.
    ("intro_logo", effects.intro_logo),
    ("results_overlay", hud.results_overlay),
    ("watermark", hud.watermark),
]

for _name, _fn in _ORDER:
    register_element(
        _name,
        ElementSpec(
            name=_name, asset_basenames=(), variables=(), render_fn=_fn,
        ),
    )
