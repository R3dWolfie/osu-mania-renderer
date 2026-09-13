"""Legacy score/accuracy sprite-font compositor regression tests."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, legacy_text_layout


def test_fixed_digits_use_five_width_while_punctuation_stays_variable():
    layout = legacy_text_layout(
        "15.5%",
        {
            "1": (10, 40), "5": (20, 40),
            ".": (5, 40), "%": (30, 40),
        },
        scale=1,
        overlap=2,
    )

    assert layout.width == 87
    assert [glyph.width for glyph in layout.glyphs] == [10, 20, 5, 20, 30]
    assert layout.glyphs[0].x == 5  # narrow '1' centred in a score-5 cell
    assert layout.glyphs[-1].x == 57  # percent keeps its authored width


def test_score_prefix_controls_score_sprite_resolution(tmp_path):
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (15, 30), (20, 40, 60, 255)).save(skin / "neon-0.png")

    frames, source = SpriteAtlas._resolve_global(
        "score_0",
        skin_dir=skin,
        beatmap_dir=None,
        section=None,
        score_prefix="neon",
    )

    assert source == "user"
    assert frames[0].size == (15, 30)


class _HudAtlas:
    def global_source(self, slot):
        return "user" if slot.startswith("score_") else "missing"

    def global_native_size(self, slot):
        if slot == "score_dot":
            return 10.0, 40.0
        if slot == "score_percent":
            return 30.0, 40.0
        return 20.0, 40.0

    def global_has_visible_pixels(self, _slot):
        return True

    def index_of(self, slot):
        return slot


def test_custom_hud_draws_actual_score_assets_with_source_geometry():
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1024, height=768)
    renderer.options = SimpleNamespace(
        show_score=True, show_pp_counter=False,
    )
    renderer.skin_ini = SimpleNamespace(score_overlap=0)
    renderer.atlas = _HudAtlas()
    renderer.draws = []
    renderer._draw_sprite_idx = lambda *args: renderer.draws.append(args)
    renderer._draw_legacy_mod_icons = lambda _scene, fallback_anchor_y: fallback_anchor_y
    renderer._cached_text = lambda *_args: (_ for _ in ()).throw(
        AssertionError("PIL score/accuracy path used"),
    )
    scene = SimpleNamespace(max_pp=0, replay_mods=0)

    FrameRenderer._draw_custom_legacy_hud(renderer, scene, 6430, 98.31)

    # 8 padded score digits, then five accuracy digits + dot + percent.
    assert [draw[0] for draw in renderer.draws[:8]] == [
        "score_0", "score_0", "score_0", "score_0",
        "score_6", "score_4", "score_3", "score_0",
    ]
    assert "score_dot" in [draw[0] for draw in renderer.draws]
    assert "score_percent" in [draw[0] for draw in renderer.draws]
    # Score top/right: margin 10, scale .96. Accuracy: margin 17, scale .576.
    score_draws = renderer.draws[:8]
    accuracy_draws = renderer.draws[8:]
    assert max(draw[1] + draw[3] for draw in score_draws) == 1014
    assert max(draw[1] + draw[3] for draw in accuracy_draws) == 1007
    assert max(draw[2] + draw[4] for draw in score_draws) == 768
    assert max(draw[2] + draw[4] for draw in accuracy_draws) == 721


def test_transparent_percent_placeholder_has_no_layout_advance():
    layout = legacy_text_layout(
        "98.31%",
        {"9": (20, 40), "8": (20, 40), ".": (5, 40),
         "3": (20, 40), "1": (20, 40), "5": (20, 40)},
        scale=1,
        overlap=0,
    )

    assert all(glyph.slot != "score_percent" for glyph in layout.glyphs)
    assert layout.width == 85
