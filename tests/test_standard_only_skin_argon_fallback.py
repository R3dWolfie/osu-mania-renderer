"""Standard-only skins must retain the complete Argon Mania presentation."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.beatmap.mods import Mod, actual_mod_acronyms
from osu_mania_renderer_v2.beatmap.skin_ini import parse_skin_ini
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    LegacyScorebarLayout,
    RenderContext,
)
from osu_mania_renderer_v2.render.scene import SceneState, VisibleNote
from osu_mania_renderer_v2.wiki_elements.hud import (
    _draw_argon_hud,
    _draw_mod_icons,
)


class _ResolvedSkinAtlas:
    """Resolve real temporary skin files without requiring a GL context."""

    def __init__(self, skin_dir, mania_section, key_count=4):
        self.skin_dir = skin_dir
        self.mania_section = mania_section
        self.key_count = key_count
        self._globals = {}

    def column_source(self, kind, column):
        _frames, source = SpriteAtlas._resolve_column(
            kind=kind,
            col=column,
            key_count=self.key_count,
            skin_dir=self.skin_dir,
            beatmap_dir=None,
            section=self.mania_section,
        )
        return source

    def _global(self, slot):
        if slot not in self._globals:
            self._globals[slot] = SpriteAtlas._resolve_global(
                slot,
                skin_dir=self.skin_dir,
                beatmap_dir=None,
                section=self.mania_section,
            )
        return self._globals[slot]

    def global_source(self, slot):
        return self._global(slot)[1]

    def global_native_size(self, slot):
        image = self._global(slot)[0][0]
        scale_adjust = float(image.info.get("scale_adjust", 1) or 1)
        return image.width / scale_adjust, image.height / scale_adjust

    def direct_image(self, slot):
        return self._global(slot)[0][0]


def _write_standard_only_skin(path):
    path.mkdir()
    (path / "skin.ini").write_text(
        "[General]\nName: Synthetic standard-only skin\n"
        "[Fonts]\nScorePrefix: score\n",
        encoding="utf-8",
    )
    # Standard global assets are deliberately present. None is structural
    # evidence that the skin supplies a Mania presentation.
    Image.new("RGBA", (720, 420), (255, 255, 255, 255)).save(
        path / "scorebar-bg.png",
    )
    Image.new("RGBA", (650, 30), (255, 255, 255, 255)).save(
        path / "scorebar-colour.png",
    )
    Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(
        path / "selection-mod-hidden.png",
    )
    for digit in range(10):
        Image.new("RGBA", (20, 30), (255, 255, 255, 255)).save(
            path / f"score-{digit}.png",
        )


def _renderer_for_skin(path):
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(key_count=4, width=1280, height=720)
    renderer.options = SimpleNamespace(hud_opacity=1.0, show_hp_bar=True)
    renderer.skin_ini = parse_skin_ini(path)
    renderer.mania_section = renderer.skin_ini.mania_for_keycount(4)
    renderer.atlas = _ResolvedSkinAtlas(path, renderer.mania_section)
    return renderer


def test_standard_only_skin_dispatches_every_gameplay_pass_to_argon(tmp_path):
    skin = tmp_path / "standard-only"
    _write_standard_only_skin(skin)
    renderer = _renderer_for_skin(skin)

    assert renderer.mania_section is None
    assert renderer._is_argon_default() is True
    assert renderer._has_custom_health_bar_assets() is True
    # Even a pair that would look like a standard composite in isolation must
    # not enter the legacy Mania scorebar classifier without Mania evidence.
    assert renderer._classify_legacy_scorebar_layout() is None

    calls = []
    renderer._draw_argon_stage_decorations = lambda scene: calls.append("stage")
    renderer._draw_argon_columns = lambda scene: calls.append("columns")
    renderer._draw_argon_notes = lambda scene: calls.append("notes")
    renderer._draw_argon_receptors = lambda scene: calls.append("receptors")
    renderer._draw_argon_combo_and_judgment = (
        lambda scene, draw_combo=True: calls.append("combo-judgment")
    )
    renderer._draw_argon_hud = lambda scene: calls.append("hud")
    renderer._draw_mania_health_bar = lambda hp: calls.append("mania-hp")
    renderer._draw_standard_legacy_health_bar = lambda hp: calls.append("standard-hp")
    renderer._draw_sprite = lambda *args: calls.append("procedural-hp")
    scene = SimpleNamespace(
        t_ms=100,
        hp=0.5,
        key_press_age_ms=(0, 0, 0, 0),
    )

    renderer._draw_stage_decorations(scene)
    renderer._draw_columns(scene)
    renderer._draw_stage_lights(scene)
    renderer._draw_notes(scene)
    renderer._draw_combo_and_judgment(scene)
    renderer._draw_receptors(scene)
    renderer._draw_hp_bar(scene)
    renderer._draw_hud(scene)

    assert calls == [
        "stage",
        "columns",
        "notes",
        "combo-judgment",
        "receptors",
        "hud",
    ]


def test_genuine_legacy_mania_skin_stays_on_legacy_path(tmp_path):
    skin = tmp_path / "legacy-mania"
    skin.mkdir()
    (skin / "skin.ini").write_text(
        "[General]\nName: Synthetic legacy Mania skin\n"
        "[Mania]\nKeys: 4\nColumnWidth: 30,30,30,30\n",
        encoding="utf-8",
    )
    Image.new("RGBA", (48, 24), (255, 255, 255, 255)).save(
        skin / "mania-note1.png",
    )
    Image.new("RGBA", (48, 48), (255, 255, 255, 255)).save(
        skin / "mania-key1.png",
    )
    Image.new("RGBA", (712, 100), (255, 255, 255, 255)).save(
        skin / "scorebar-bg.png",
    )
    Image.new("RGBA", (695, 53), (255, 255, 255, 255)).save(
        skin / "scorebar-colour.png",
    )
    renderer = _renderer_for_skin(skin)

    assert renderer.mania_section is not None
    assert renderer._is_argon_default() is False
    classification = renderer._classify_legacy_scorebar_layout()
    assert classification is not None
    assert classification.layout is LegacyScorebarLayout.MANIA_SIDE

    calls = []
    renderer._draw_argon_hud = lambda scene: calls.append("argon")
    renderer._draw_custom_legacy_hud = (
        lambda scene, score, accuracy: calls.append("legacy")
    )
    renderer._draw_hud(SimpleNamespace(
        results_opacity=0,
        score=100,
        score_smoothed=90,
        accuracy=99.0,
        accuracy_smoothed=98.5,
    ))
    assert calls == ["legacy"]


def test_argon_hud_owns_the_health_presentation(monkeypatch):
    from osu_mania_renderer_v2.wiki_elements import hud as hud_module

    direct = []
    health = []
    monkeypatch.setattr(
        hud_module,
        "_draw_argon_health_display",
        lambda ctx, geometry, hp: health.append(hp),
    )
    monkeypatch.setattr(hud_module, "_draw_argon_wedges", lambda ctx: None)
    renderer = SimpleNamespace(
        rc=SimpleNamespace(
            width=1280,
            height=720,
            ctx=SimpleNamespace(blend_func=None),
        ),
        _banner_text="",
        _cached_text=lambda text, size, colour: (text, 20, 10),
        _draw_external_texture=lambda *args, **kwargs: None,
        _draw_direct=lambda name, *args, **kwargs: direct.append(name),
        _flush_sprite_batch=lambda: None,
    )
    ctx = SimpleNamespace(
        fr=renderer,
        height=720,
        options=SimpleNamespace(
            show_hp_bar=True,
            show_score=False,
            show_mods=True,
            show_scoreboard=True,
            show_pp_counter=False,
        ),
        scene=SimpleNamespace(
            results_opacity=0,
            score=0,
            score_smoothed=0,
            accuracy=100.0,
            accuracy_smoothed=100.0,
            hp=0.5,
            max_pp=0,
            pp=0,
            combo=0,
            replay_mods=0,
        ),
        atlas=SimpleNamespace(global_source=lambda slot: "bundle"),
        draw_direct=lambda name, *args: direct.append(name),
        draw_sprite=lambda *args: None,
        draw_number=lambda *args, **kwargs: None,
        text=lambda text, size, colour: (text, 20, 10),
        draw_external=lambda *args: None,
    )

    _draw_argon_hud(ctx)

    assert health == [0.5]
    assert "argon_hp" not in direct
    assert "argon_wedge" not in direct


def test_argon_mod_display_uses_only_actual_replay_flags():
    assert actual_mod_acronyms(0) == ()
    assert actual_mod_acronyms(int(Mod.K4)) == ("4K",)

    sprites = []
    labels = []
    ctx = SimpleNamespace(
        height=768,
        options=SimpleNamespace(show_mods=True),
        scene=SimpleNamespace(
            replay_mods=0,
            # The older scene field still carries the synthetic native 4K
            # label; Argon must not consume it.
            mod_acronyms=("4K",),
        ),
        draw_sprite=lambda *args: sprites.append(args),
        text=lambda label, size, colour: (labels.append(label) or (label, 12, 12)),
        draw_external=lambda *args: None,
    )

    _draw_mod_icons(ctx, right_x=1200, top_y=100)
    assert sprites == []
    assert labels == []

    ctx.scene.replay_mods = int(Mod.K4)
    _draw_mod_icons(ctx, right_x=1200, top_y=100)
    assert sprites[0][0] == "mod_hex"
    assert labels == ["4K"]


@pytest.mark.slow
def test_standard_only_skin_full_argon_gpu_smoke(tmp_path):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    skin = tmp_path / "standard-only"
    _write_standard_only_skin(skin)

    with HeadlessGl(width=320, height=240) as gl:
        renderer = FrameRenderer(
            RenderContext(
                ctx=gl.ctx,
                fbo=gl.fbo,
                width=320,
                height=240,
                key_count=4,
            ),
            skin_dir=skin,
        )
        assert renderer._is_argon_default() is True
        assert renderer._legacy_scorebar_classification is None
        renderer.draw(SceneState(
            t_ms=100,
            visible_notes=(VisibleNote(
                column=1,
                is_hold=False,
                y_fraction=0.5,
                head_y_fraction=0.5,
                tail_y_fraction=0.5,
            ),),
            keys_held=(False, False, False, False),
            visual_mods=VisualMods(),
        ))

        assert max(gl.fbo.read(components=3)) > 50
