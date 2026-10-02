"""Stable source-backed topology, global skin indexing, and split draw parity."""

from __future__ import annotations

import os
from types import SimpleNamespace

import moderngl
import numpy as np
from PIL import Image
import pytest

from osu_mania_renderer_v2.beatmap.models import RenderOptions, VisualMods
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection, parse_skin_ini
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.legacy_stage_geometry import (
    legacy_stage_layout,
    legacy_stage_topology,
)
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import JudgmentPopup, SceneState, VisibleNote
from osu_mania_renderer_v2.wiki_elements import notes, stage
from osu_mania_renderer_v2.wiki_elements.context import FrameContext


@pytest.mark.parametrize(
    "keys,counts,specials",
    [
        (1, (1,), (0,)),
        (4, (4,), ()),
        (7, (7,), (3,)),
        (9, (9,), (4,)),
        (10, (10,), ()),
        (11, (6, 5), (8,)),
        (12, (6, 6), ()),
        (13, (7, 6), (3,)),
        (14, (7, 7), (3, 10)),
        (16, (8, 8), ()),
        (18, (9, 9), (4, 13)),
    ],
)
def test_stable_auto_topology_and_global_mapping(keys, counts, specials):
    layout = legacy_stage_topology(keys)
    assert tuple(s.column_count for s in layout.stages) == counts
    assert sum(counts) == keys
    assert tuple(c for c in range(keys) if layout.column_kind(c) == "S") == specials
    offset = 0
    for s in layout.stages:
        assert s.first_column == offset
        for local in range(s.column_count):
            assert layout.locate(offset + local) == (s, local)
        offset += s.column_count
    for invalid in (-1, keys):
        with pytest.raises(IndexError):
            layout.locate(invalid)


@pytest.mark.parametrize("keys", [2, 4, 7, 9, 10, 11, 12, 13, 18])
def test_skin_can_force_both_directions_even_with_coop(keys):
    for mods in (0, int(Mod.KC)):
        single = legacy_stage_topology(keys, ManiaSection(keys, split_stages=False), mods=mods)
        split = legacy_stage_topology(keys, ManiaSection(keys, split_stages=True), mods=mods)
        assert tuple(s.column_count for s in single.stages) == (keys,)
        assert tuple(s.column_count for s in split.stages) == ((keys + 1) // 2, keys // 2)
    assert (
        len(legacy_stage_topology(1, ManiaSection(1, split_stages=True), mods=int(Mod.KC)).stages)
        == 1
    )


def test_native_coop_never_doubles_columns_and_resolved_convert_is_not_doubled_again():
    # Stable ColumnsWithMods returns native CS before checking KC. A standard
    # 6K+KC conversion instead supplies an already-resolved total of 12.
    native = legacy_stage_topology(6, mods=int(Mod.KC))
    converted = legacy_stage_topology(12, mods=int(Mod.K6 | Mod.KC))
    assert tuple(s.column_count for s in native.stages) == (3, 3)
    assert tuple(s.column_count for s in converted.stages) == (6, 6)
    assert native.locate(5)[1] == 2
    assert converted.locate(11)[1] == 5


@pytest.mark.parametrize(
    "keys,counts",
    [(1, (1,)), (2, (1, 1)), (9, (5, 4)), (10, (5, 5)), (11, (6, 5)),
     (12, (6, 6)), (18, (9, 9))],
)
def test_keycoop_split_and_global_mapping_remain_unchanged(keys, counts):
    layout = legacy_stage_topology(keys, mods=int(Mod.KC))
    assert tuple(s.column_count for s in layout.stages) == counts
    assert sum(s.column_count for s in layout.stages) == keys
    for c in range(keys):
        stage, local = layout.locate(c)
        assert stage.first_column + local == c


@pytest.mark.parametrize("style,specials", [(0, ()), (1, (0,)), (2, (9,))])
def test_default_10k_special_style_uses_one_stage(style, specials):
    layout = legacy_stage_topology(10, ManiaSection(10, special_style=style))
    assert tuple(s.column_count for s in layout.stages) == (10,)
    assert tuple(c for c in range(10) if layout.column_kind(c) == "S") == specials


@pytest.mark.parametrize(
    "keys,style,expected",
    [
        (12, 1, (0, 11)),
        (12, 2, (5, 6)),
        (11, 1, (0, 8)),
        (11, 2, (5, 8)),
        (13, 1, (3, 12)),
        (13, 2, (3, 7)),
        (10, 1, (2, 7)),
        (16, 1, (0, 15)),
    ],
)
def test_special_style_is_local_and_mirrored(keys, style, expected):
    # Keep 10K mirror coverage through an explicit split, not auto topology.
    layout = legacy_stage_topology(
        keys, ManiaSection(keys, special_style=style, split_stages=True)
    )
    assert tuple(c for c in range(keys) if layout.column_kind(c) == "S") == expected
    assert layout.stages[1].special_style == 3 - style
    # In an active left-special stage, the non-special even lanes use kind 2.
    if keys == 12 and style == 1:
        assert tuple(layout.column_kind(c) for c in range(keys)) == (
            "S",
            "1",
            "2",
            "1",
            "2",
            "1",
            "1",
            "2",
            "1",
            "2",
            "1",
            "S",
        )


def test_parser_retains_nullable_overrides_and_total_key_lookup(tmp_path):
    (tmp_path / "skin.ini").write_text(
        "\ufeff[Mania]\nKeys:6\nSplitStages:1\nStageSeparation:12.5\nSeparateScore:0\n[Mania]\nKeys:12\nSplitStages:0\n"
    )
    ini = parse_skin_ini(tmp_path)
    section = ini.mania_for_keycount(6)
    assert section.split_stages is True
    assert section.stage_separation == 12.5
    assert section.separate_score is False
    assert ini.mania_for_keycount(12).split_stages is False
    assert ini.mania_for_keycount(10) is None  # never borrow the half-key block
    assert ManiaSection(12).separate_score is None  # consumer default true
    assert len(legacy_stage_topology(12, section).stages) == 2


def test_stage_gap_reference_units_boundary_spacing_and_global_widths():
    section = ManiaSection(
        12,
        column_start=100,
        column_right=20,
        column_width=tuple(range(20, 32)),
        column_spacing=(1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
    )
    layout = legacy_stage_layout(12, section, render_width=1920, render_height=1080)
    left, right = layout.stages
    assert layout.width_scale == 1
    assert left.x == 225
    assert layout.separation == 90
    assert right.x - left.right == 90
    assert layout.column_x[6] - right.x == 7 * 2.25
    assert layout.column_x[6] - (layout.column_x[5] + layout.column_width[5]) == 47 * 2.25
    assert layout.column_width[5:7] == (25 * 2.25, 26 * 2.25)
    assert right.right == layout.column_x[-1] + layout.column_width[-1]


def test_fractional_geometry_parser_preserves_global_csv_positions(tmp_path):
    # stable Components.Section.ConvertString recursively keeps a failed
    # float list item as 0; it never drops/shifts the next column's value.
    (tmp_path / "skin.ini").write_text(
        "[Mania]\nKeys:12\nColumnStart:136.5\nColumnRight:19.5\nColumnWidth:20.5,bad,31.5\nColumnSpacing:1.5,bad,3.5\nColumnLineWidth:0.5,bad,2.5\n"
    )
    section = parse_skin_ini(tmp_path).mania_for_keycount(12)
    assert (section.column_start, section.column_right) == (136.5, 19.5)
    assert section.column_width == (20.5, 0, 31.5)
    assert section.column_spacing == (1.5, 0, 3.5)
    assert section.column_line_width == (0.5, 0, 2.5)
    layout = legacy_stage_layout(12, section, render_width=1920, render_height=1080)
    assert layout.column_width[:3] == (20.5 * 2.25, 5 * 2.25, 31.5 * 2.25)


def test_stable_csv_prefix_clamps_and_fit_scale_every_stage_equally():
    section = ManiaSection(12, column_width=(500,), column_spacing=(-999,), stage_separation=1)
    layout = legacy_stage_layout(12, section, render_width=640, render_height=480)
    assert layout.column_width == (100,) + (30,) * 11  # one entry is NOT broadcast
    assert layout.column_x[1] == layout.column_x[0] + 70  # clamp spacing to -next width
    assert layout.separation == 5
    wide = legacy_stage_layout(
        18,
        ManiaSection(18, column_start=136, column_right=19, column_width=(100,) * 18),
        render_width=640,
        render_height=480,
    )
    assert wide.width_scale == pytest.approx((640 - 136 - 19) / 1840)
    assert wide.stages[-1].right == pytest.approx(621)
    assert wide.separation == pytest.approx(40 * wide.width_scale)
    assert len(set(wide.column_width)) == 1


def test_fit_uses_stable_ceiling_of_screen_reference_width():
    layout = legacy_stage_layout(
        18, ManiaSection(18, column_width=(100,) * 18), render_width=1920, render_height=1080
    )
    assert layout.width_scale == pytest.approx((854 - 136 - 19) / 1840)
    assert layout.stages[-1].right == pytest.approx((854 - 19) * 2.25)


@pytest.mark.parametrize("keys", [4, 7, 9, 10])
def test_single_stage_renderer_geometry_stays_unchanged(keys):
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(width=1280, height=720, key_count=keys)
    fr.mania_section = ManiaSection(keys, column_start=267, column_width=(30,))
    fr._is_argon_default = lambda: False
    fr._compute_playfield_geometry()
    assert fr.col_w == (45,) * keys  # preserve existing singleton shorthand
    assert fr.col_x == tuple(round((1280 - 45 * keys) / 2) + c * 45 for c in range(keys))
    assert len(fr.stage_layout.stages) == 1


@pytest.mark.parametrize(
    "split,expected_colours",
    [(True, ("yellow", "red", "green", "yellow")),
     (False, ("red", "red", "green", "red"))],
)
def test_atlas_uses_stage_local_fallback_and_global_authored_override(
    tmp_path, split, expected_colours
):
    for kind, colour in (("1", "red"), ("2", "blue"), ("S", "yellow")):
        Image.new("RGBA", (8, 8), colour).save(tmp_path / f"mania-note{kind}.png")
    Image.new("RGBA", (8, 8), "green").save(tmp_path / "global6.png")
    section = ManiaSection(10, note_image={6: "global6"}, split_stages=split)
    for c, expected in zip((2, 5, 6, 7), expected_colours):
        frames, source = SpriteAtlas._resolve_column(
            kind="note_tap",
            col=c,
            key_count=10,
            skin_dir=tmp_path,
            beatmap_dir=None,
            section=section,
        )
        assert source == "user"
        assert frames[0].getpixel((4, 4)) == Image.new("RGBA", (1, 1), expected).getpixel((0, 0))


@pytest.mark.parametrize("separate", [None, True, False])
@pytest.mark.parametrize("wiki", [False, True])
def test_separate_score_only_routes_judgment_and_combo_is_shared(separate, wiki):
    fr = object.__new__(FrameRenderer)
    fr.mania_section = ManiaSection(12, separate_score=separate)
    fr.stage_layout = legacy_stage_layout(
        12, fr.mania_section, render_width=1920, render_height=1080
    )
    fr._is_argon_default = lambda: False
    judgments, combos = [], []
    fr._draw_custom_legacy_judgment = lambda scene, **kw: judgments.append(
        (scene.active_judgments, kw["center_x"])
    )
    fr._draw_custom_legacy_combo = lambda scene, **kw: combos.append((scene, kw))
    scene = SceneState(
        1000,
        (),
        (False,) * 12,
        VisualMods(),
        active_judgments=(JudgmentPopup(5, "300", 60), JudgmentPopup(6, "50", 20)),
        combo=123,
    )
    if wiki:
        # Exercise the real wiki dispatch, not just the engine helper.
        notes.combo_and_judgment(
            element=None,
            skin=None,
            assets=None,
            variables=None,
            ctx=SimpleNamespace(fr=fr, scene=scene),
        )
    else:
        fr._draw_combo_and_judgment(scene)
    assert len(judgments) == len(combos) == 2
    if separate is False:
        assert judgments[0][0] == judgments[1][0] == scene.active_judgments
    else:
        assert [tuple(j.column for j in js) for js, x in judgments] == [(5,), (6,)]
    assert [x for js, x in judgments] == [s.center_x for s in fr.stage_layout.stages]
    assert all(s is scene and s.combo == 123 for s, kw in combos)


def _skin(tmp_path):
    widths = ",".join(["10"] * 12)
    lines = ",".join(["0"] * 13)
    (tmp_path / "skin.ini").write_text(
        f"[General]\nVersion:2.7\n[Mania]\nKeys:12\nColumnStart:100\nColumnWidth:{widths}\nColumnLineWidth:{lines}\nHitPosition:400\n"
    )
    for name, size, colour in (
        ("mania-stage-left", (4, 768), (0, 0, 255, 255)),
        ("mania-stage-right", (4, 768), (255, 0, 0, 255)),
        ("mania-stage-hint", (20, 4), (255, 255, 0, 255)),
        ("mania-stage-bottom", (20, 4), (255, 0, 255, 255)),
        ("mania-key1", (10, 60), (0, 255, 0, 255)),
        ("mania-key2", (10, 60), (0, 255, 0, 255)),
        ("mania-note1", (10, 10), (0, 255, 255, 255)),
        ("mania-note2", (10, 10), (0, 255, 255, 255)),
    ):
        Image.new("RGBA", size, colour).save(tmp_path / f"{name}.png")


@pytest.mark.slow
def test_split_gl_chrome_gap_note_key_alignment_and_path_parity(tmp_path):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 for real GL")
    _skin(tmp_path)
    scene = SceneState(
        1000,
        tuple(VisibleNote(c, False, 0.7, 0.7, 0.7) for c in (5, 6)),
        (False,) * 12,
        VisualMods(),
    )
    outputs = []
    with HeadlessGl(width=640, height=480) as gl:
        fr = FrameRenderer(
            RenderContext(gl.ctx, gl.fbo, 640, 480, 12),
            RenderOptions(resolution=(640, 480), fps=60),
            skin_dir=tmp_path,
        )
        gl.ctx.enable(moderngl.BLEND)
        gl.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        for wiki in (False, True):
            gl.fbo.use()
            gl.fbo.clear(0, 0, 0, 1)
            if wiki:
                ctx = FrameContext(fr, None, gl.ctx, gl.fbo, 640, 480, 12)
                ctx.scene = scene
                stage.stage_decorations(
                    element=None, skin=None, assets=None, variables=None, ctx=ctx
                )
                stage.columns(element=None, skin=None, assets=None, variables=None, ctx=ctx)
                notes._draw_notes_body(ctx)
                notes._receptors(ctx)
                stage.stage_foreground(ctx)
            else:
                fr._draw_stage_decorations(scene)
                fr._draw_columns(scene)
                fr._draw_notes(scene)
                fr._draw_receptors(scene)
                fr._draw_legacy_stage_foreground()
            fr._flush_sprite_batch()
            pixels = np.frombuffer(gl.fbo.read(components=3), np.uint8).reshape(480, 640, 3)
            outputs.append(pixels.copy())
            left, right = fr.stage_layout.stages
            assert (left.right, right.x) == (160, 200)
            assert np.array_equal(pixels[300, 161], (255, 0, 0))  # right of first
            assert np.array_equal(pixels[300, 198], (0, 0, 255))  # left of second
            assert np.array_equal(pixels[100:450, 175:185], np.zeros((350, 10, 3), np.uint8))
            for c in (5, 6):
                center = int(fr.col_x[c] + fr.col_w[c] / 2)
                assert np.array_equal(pixels[30, center], (0, 255, 0))
                assert np.array_equal(pixels[205, center], (0, 255, 255))
            assert fr.col_x[6] - (fr.col_x[5] + fr.col_w[5]) == 40
            for s in fr.stage_layout.stages:
                assert np.array_equal(pixels[2, int(s.center_x)], (255, 0, 255))
        assert np.array_equal(*outputs)


@pytest.mark.slow
def test_split_boundary_holds_and_global_hit_light_order_gl(tmp_path):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 for real GL")
    _skin(tmp_path)
    for kind in ("1", "2"):
        Image.new("RGBA", (10, 20), (100, 0, 0, 255)).save(tmp_path / f"mania-note{kind}L.png")
        Image.new("RGBA", (10, 10), (0, 0, 255, 255)).save(tmp_path / f"mania-note{kind}T.png")
    Image.new("RGBA", (36, 600), (100, 0, 0, 255)).save(tmp_path / "mania-lightingN.png")
    scene = SceneState(
        1000,
        tuple(VisibleNote(c, True, 0.7, 0.7, 0.4) for c in (5, 6)),
        (False,) * 12,
        VisualMods(),
        hit_light_age_ms=(9999,) * 5 + (80, 80) + (9999,) * 5,
        hit_light_judgment=("",) * 5 + ("300", "300") + ("",) * 5,
    )
    outputs = []
    with HeadlessGl(width=640, height=480) as gl:
        fr = FrameRenderer(RenderContext(gl.ctx, gl.fbo, 640, 480, 12), skin_dir=tmp_path)
        gl.ctx.enable(moderngl.BLEND)
        gl.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        for wiki in (False, True):
            gl.fbo.use()
            gl.fbo.clear(0, 0, 0, 1)
            if wiki:
                ctx = FrameContext(fr, None, gl.ctx, gl.fbo, 640, 480, 12, scene=scene)
                notes._draw_notes_body(ctx)
                notes._receptors(ctx)
            else:
                fr._draw_notes(scene)
                fr._draw_receptors(scene)
            # Both frame orchestrators consume this same global layer after keys.
            fr._draw_legacy_hit_lighting(scene)
            fr._flush_sprite_batch()
            pixels = np.frombuffer(gl.fbo.read(components=3), np.uint8).reshape(480, 640, 3).copy()
            outputs.append(pixels)
            for c in (5, 6):
                center = int(fr.col_x[c] + fr.col_w[c] / 2)
                assert np.array_equal(pixels[206, center], (0, 255, 255))  # head over body
                assert np.array_equal(pixels[250, center], (100, 0, 0))
                assert np.array_equal(pixels[315, center], (0, 0, 255))  # rear over body
                assert pixels[30, center, 0] > 20 and pixels[30, center, 1] == 255  # light over key
            assert np.count_nonzero(pixels[:, 175:185]) == 0
        assert np.array_equal(*outputs)


def test_split_stage_bottom_retains_native_animation_frames(tmp_path):
    for frame, colour in ((0, "red"), (1, "blue")):
        Image.new("RGBA", (20, 4), colour).save(tmp_path / f"mania-stage-bottom-{frame}.png")

    class Probe:
        def texture_array(self, **kwargs):
            return SimpleNamespace(filter=None, build_mipmaps=lambda: None)

    atlas = SpriteAtlas.load(Probe(), key_count=12, skin_dir=tmp_path)
    assert atlas.frame_count("playfield_frame") == atlas.direct_frame_count("playfield_frame") == 2
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(height=480)
    fr.stage_layout = legacy_stage_layout(12, None, render_width=640, render_height=480)
    fr.atlas, fr.upside_down = atlas, False
    calls = []
    fr._draw_direct = lambda *args, **kwargs: calls.append((args, kwargs))
    for clock, expected in ((0, 0), (17, 1), (34, 0), (17, 1)):
        calls.clear()
        fr._stage_clock_ms = clock
        fr._draw_split_stage_foreground()
        assert len(calls) == 2
        assert all(kw["frame_index"] == expected for args, kw in calls)
