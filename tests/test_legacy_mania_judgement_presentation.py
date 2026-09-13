"""Focused source-contract tests for custom legacy Mania judgments."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    legacy_judgment_alpha,
    legacy_judgment_frame,
    legacy_judgment_scale,
    legacy_mania_position_gl,
)
from osu_mania_renderer_v2.render.scene import JudgmentPopup


def test_score_and_combo_positions_are_independent_and_flip_for_upscroll():
    assert legacy_mania_position_gl(250, 720, False) == 345
    assert legacy_mania_position_gl(280, 720, False) == 300
    assert legacy_mania_position_gl(250, 720, True) == 375
    assert legacy_mania_position_gl(280, 720, True) == 420

    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720, key_count=4)
    renderer.mania_section = ManiaSection(
        keys=4,
        column_width=(70,),
        combo_position=250,
        score_position=280,
        upside_down=True,
    )
    renderer._is_argon_default = lambda: False
    FrameRenderer._compute_playfield_geometry(renderer)

    assert renderer.combo_baseline_y_gl == 375
    assert renderer.score_popup_y_gl == 420


def test_judgment_frame_is_looped_at_20fps_from_frame_zero():
    assert legacy_judgment_frame(0, 3) == 0
    assert legacy_judgment_frame(49, 3) == 0
    assert legacy_judgment_frame(50, 3) == 1
    assert legacy_judgment_frame(100, 3) == 2
    assert legacy_judgment_frame(150, 3) == 0


def test_judgment_220ms_fade_envelope():
    assert legacy_judgment_alpha(-1) == 0
    assert legacy_judgment_alpha(0) == 0
    assert legacy_judgment_alpha(10) == pytest.approx(0.5)
    assert legacy_judgment_alpha(20) == 1
    assert legacy_judgment_alpha(179) == 1
    assert legacy_judgment_alpha(200) == pytest.approx(0.5)
    assert legacy_judgment_alpha(220) == 0


def test_success_and_miss_scale_phases_match_legacy_source():
    assert legacy_judgment_scale(0, miss=False) == pytest.approx(0.8)
    assert legacy_judgment_scale(39.999, miss=False) == pytest.approx(
        1.0, abs=0.00001,
    )
    assert legacy_judgment_scale(40, miss=False) == pytest.approx(0.85)
    assert legacy_judgment_scale(80, miss=False) == pytest.approx(0.7)
    assert legacy_judgment_scale(180, miss=False) == pytest.approx(0.7)
    assert legacy_judgment_scale(220, miss=False) == pytest.approx(0.4)

    assert legacy_judgment_scale(0, miss=True) == pytest.approx(1.2)
    assert legacy_judgment_scale(50, miss=True) == pytest.approx(1.05)
    assert legacy_judgment_scale(100, miss=True) == pytest.approx(1.0)


def test_custom_judgment_uses_native_design_size_and_score_position():
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=768)
    renderer.options = SimpleNamespace(show_judgment=True)
    renderer.pf_x = 400
    renderer.pf_w = 320
    renderer.score_popup_y_gl = 300
    renderer.atlas = SimpleNamespace(
        global_native_size=lambda _slot: (80.0, 40.0),
        index_of=lambda _slot: 7,
        frame_count=lambda _slot: 1,
    )
    draws = []
    renderer._draw_sprite_idx = lambda *args: draws.append(args)
    scene = SimpleNamespace(active_judgments=(
        JudgmentPopup(column=0, judgment="miss", age_ms=100),
    ))

    FrameRenderer._draw_custom_legacy_judgment(renderer, scene)

    assert draws == [(7, 520, 280, 80, 40, (1.0, 1.0, 1.0, 1.0))]


def test_hit200_default_uses_mania_hit200_not_old_100k_alias(tmp_path):
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (20, 20), (255, 0, 0, 255)).save(
        skin / "mania-hit100k.png",
    )
    Image.new("RGBA", (20, 20), (0, 255, 0, 255)).save(
        skin / "mania-hit200.png",
    )

    frames, source = SpriteAtlas._resolve_global(
        "judgment_katu", skin_dir=skin, beatmap_dir=None, section=None,
    )

    assert source == "user"
    assert frames[0].getpixel((0, 0)) == (0, 255, 0, 255)


def test_explicit_hit200_override_remains_authoritative(tmp_path):
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (20, 20), (0, 255, 0, 255)).save(
        skin / "mania-hit200.png",
    )
    Image.new("RGBA", (20, 20), (0, 0, 255, 255)).save(
        skin / "custom-hit.png",
    )

    frames, source = SpriteAtlas._resolve_global(
        "judgment_katu",
        skin_dir=skin,
        beatmap_dir=None,
        section=ManiaSection(keys=4, hit_200="custom-hit"),
    )

    assert source == "user"
    assert frames[0].getpixel((0, 0)) == (0, 0, 255, 255)
