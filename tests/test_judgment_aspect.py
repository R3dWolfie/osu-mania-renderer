"""#72: judgement popup preserves its sprite aspect ratio (custom skins).

Non-GL unit tests. They exercise `FrameRenderer._draw_combo_and_judgment`
directly on a bare (un-__init__'d) renderer with a stub atlas, capturing the
size handed to `_draw_sprite_idx`. This mirrors the stage_left/right
global_aspect handling: `jud_w` stays the width anchor while `jud_h` follows
the sprite's native aspect (width / height). Before the fix `jud_h = jud_w`
forced a square quad, stretching a custom skin's non-square mania-hit300 into
the "horror game" distortion reported for "OT!skin collab".

Deliberately a standalone, dependency-light module: the legacy slow GL test
files (test_gpu_renderer_hud.py etc.) import from the pre-refactor
`osu_mania_renderer_v2.scene` / `.models` paths and no longer collect on this
branch, so this file uses the current `render.scene` path instead.
"""
from types import SimpleNamespace

from osu_mania_renderer_v2.gpu.renderer import FrameRenderer
from osu_mania_renderer_v2.render.scene import JudgmentPopup


def _run_judgment_draw(sprite_aspect):
    """Invoke the real _draw_combo_and_judgment with a stub atlas whose
    global_aspect() returns `sprite_aspect`; return (jud_w, jud_h) as passed
    to _draw_sprite_idx."""
    fr = object.__new__(FrameRenderer)
    fr.pf_x = 0
    fr.pf_w = 1000
    fr.combo_baseline_y_gl = 500
    fr.atlas = SimpleNamespace(
        global_aspect=lambda name: sprite_aspect,
        index_of=lambda name: 0,
        frame_count=lambda name: 1,
    )
    captured = {}

    def _fake_draw_sprite_idx(idx, x, y, w, h, color):
        captured["w"] = w
        captured["h"] = h

    fr._draw_sprite_idx = _fake_draw_sprite_idx
    scene = SimpleNamespace(
        active_judgments=(JudgmentPopup(column=2, judgment="300", age_ms=100),),
        combo=0,
    )
    FrameRenderer._draw_combo_and_judgment(fr, scene, draw_combo=False)
    return captured["w"], captured["h"]


def test_judgment_square_sprite_unchanged():
    # Default skins ship 384x384 squares -> aspect 1.0 -> jud_h == jud_w.
    w, h = _run_judgment_draw(1.0)
    assert w == int(1000 * 0.55)
    assert h == w


def test_judgment_wide_sprite_not_stretched():
    # A wide custom sprite (aspect 2.0, e.g. 768x384) must draw SHORTER than
    # wide, not forced square. Pre-fix this was jud_h == jud_w (stretched).
    w, h = _run_judgment_draw(2.0)
    assert w == int(1000 * 0.55)
    assert h == int(w / 2.0)
    assert h < w


def test_judgment_tall_sprite_not_stretched():
    # A tall custom sprite (aspect 0.5, e.g. 192x384) draws TALLER than wide.
    w, h = _run_judgment_draw(0.5)
    assert h == int(w / 0.5)
    assert h > w
