"""Stable-style custom legacy gameplay mod icon regression tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.mods import Mod, legacy_mod_icons
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas, legacy_mod_slot_name
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    legacy_mod_icon_geometry,
)


def test_raw_replay_mods_drive_icons_without_synthetic_keycount():
    assert legacy_mod_icons(0) == ()
    assert [(icon.acronym, icon.asset_name) for icon in legacy_mod_icons(Mod.HD)] == [
        ("HD", "hidden"),
    ]


def test_nightcore_and_perfect_suppress_implied_icons():
    icons = legacy_mod_icons(Mod.DT | Mod.NC | Mod.SD | Mod.PF)
    assert [icon.acronym for icon in icons] == ["NC", "PF"]


def test_mod_icon_resolver_prefers_beatmap_then_at2x(tmp_path):
    skin = tmp_path / "skin"
    beatmap = tmp_path / "beatmap"
    skin.mkdir()
    beatmap.mkdir()
    Image.new("RGBA", (100, 100), (255, 0, 0, 255)).save(
        skin / "selection-mod-hidden.png",
    )
    Image.new("RGBA", (200, 200), (0, 255, 0, 255)).save(
        skin / "selection-mod-hidden@2x.png",
    )

    image, source = SpriteAtlas._resolve_mod_icon(
        "hidden", skin_dir=skin, beatmap_dir=beatmap,
    )
    assert source == "user"
    assert image.size == (200, 200)
    assert image.info["scale_adjust"] == 2

    Image.new("RGBA", (60, 60), (0, 0, 255, 255)).save(
        beatmap / "selection-mod-hidden.png",
    )
    image, source = SpriteAtlas._resolve_mod_icon(
        "hidden", skin_dir=skin, beatmap_dir=beatmap,
    )
    assert source == "beatmap"
    assert image.size == (60, 60)


def test_mod_icon_position_and_texture_spaces_are_distinct():
    geometry = legacy_mod_icon_geometry(1280, 720, 1, (100, 100))

    assert geometry.position_scale == 1.5
    assert geometry.texture_scale == pytest.approx(0.9375)
    assert geometry.center == (1160, 579)
    assert geometry.rect == pytest.approx((1113.125, 532.125, 93.75, 93.75))


class _ModAtlas:
    def global_source(self, slot):
        return "user" if slot == legacy_mod_slot_name("hidden") else "missing"

    def global_native_size(self, _slot):
        return 100.0, 100.0


def test_skin_asset_is_used_and_modern_missing_asset_uses_fallback():
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720)
    renderer.atlas = _ModAtlas()
    renderer.direct = []
    renderer.fallback = []
    renderer._draw_direct = lambda *args: renderer.direct.append(args)
    renderer._draw_generated_mod_fallback = lambda icon, **kwargs: (
        renderer.fallback.append((icon, kwargs)) or (0, 0, 45, 45)
    )
    scene = SimpleNamespace(replay_mods=int(Mod.HD | Mod.MR))

    FrameRenderer._draw_legacy_mod_icons(
        renderer, scene, fallback_anchor_y=500,
    )

    assert renderer.direct[0][0] == legacy_mod_slot_name("hidden")
    assert renderer.fallback[0][0].acronym == "MR"
