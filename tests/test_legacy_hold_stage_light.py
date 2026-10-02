"""Legacy lane, hold-body, and stage-light fidelity regressions."""
from __future__ import annotations

import os
from types import SimpleNamespace

import moderngl
import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.models import KeyEvent, VisualMods
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection, parse_skin_ini
from osu_mania_renderer_v2.gpu import atlas as atlas_module
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.legacy_mania import (
    LEGACY_NOTE_BODY_REPEAT_BOTTOM,
    LEGACY_NOTE_BODY_REPEAT_TOP,
    LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM,
    LEGACY_NOTE_BODY_STRETCH,
    legacy_disallow_zero_alpha_colour,
    legacy_doubled_alpha_colour,
    legacy_hold_body_segments,
    legacy_note_body_style,
    legacy_stage_light_fps,
    legacy_stage_light_geometry,
    legacy_stage_light_presentation,
)
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.render import (
    _key_edges_per_col,
    _key_release_ages_at,
    build_frame_state,
)
from osu_mania_renderer_v2.render.scene import VisibleNote


def test_legacy_doubled_alpha_keeps_authored_rgb() -> None:
    assert legacy_doubled_alpha_colour((12, 34, 56, 255)) == pytest.approx(
        (12 / 255, 34 / 255, 56 / 255, 1.0),
    )
    assert legacy_doubled_alpha_colour((12, 34, 56, 128)) == pytest.approx(
        (12 / 255, 34 / 255, 56 / 255, (128 / 255) ** 2),
    )
    assert legacy_doubled_alpha_colour((12, 34, 56, 0)) == pytest.approx(
        (12 / 255, 34 / 255, 56 / 255, 0.0),
    )
    assert legacy_disallow_zero_alpha_colour((12, 34, 56, 128)) == pytest.approx(
        (12 / 255, 34 / 255, 56 / 255, 128 / 255),
    )
    assert legacy_disallow_zero_alpha_colour((12, 34, 56, 0))[-1] == 1.0


def _column_renderer(section: ManiaSection | None) -> FrameRenderer:
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=640, height=480, key_count=1)
    renderer.col_x = (100,)
    renderer.col_w = (30,)
    renderer.col_w_uniform = 30
    renderer.pf_x = 100
    renderer.pf_w = 30
    renderer.mania_section = section
    renderer._is_argon_default = lambda: False
    renderer.draws = []
    renderer._draw_sprite = lambda *args: renderer.draws.append(args)
    return renderer


def test_authored_column_and_line_ignore_kiai_and_double_alpha() -> None:
    section = ManiaSection(
        keys=1,
        colour={1: (20, 40, 60, 128)},
        colour_column_line=(80, 100, 120, 128),
        column_line_width=(2, 2),
    )
    renderer = _column_renderer(section)

    FrameRenderer._draw_columns(renderer, SimpleNamespace(is_kiai=True))

    assert renderer.draws[0][-1] == pytest.approx(
        (20 / 255, 40 / 255, 60 / 255, (128 / 255) ** 2),
    )
    assert renderer.draws[1][-1] == pytest.approx(
        (80 / 255, 100 / 255, 120 / 255, (128 / 255) ** 2),
    )


def test_fallback_column_retains_kiai_palette_boost() -> None:
    normal = _column_renderer(None)
    kiai = _column_renderer(None)
    FrameRenderer._draw_columns(normal, SimpleNamespace(is_kiai=False))
    FrameRenderer._draw_columns(kiai, SimpleNamespace(is_kiai=True))

    assert normal.draws[0][-1] == pytest.approx((0.07, 0.06, 0.12, 0.55))
    assert kiai.draws[0][-1] == pytest.approx((0.11, 0.10, 0.18, 0.55))


def test_parser_and_versioned_body_style_precedence(tmp_path) -> None:
    skin = tmp_path / "skin"
    skin.mkdir()
    (skin / "skin.ini").write_text(
        """
[General]
Version: 2.7
[Mania]
Keys: 4
NoteBodyStyle: 4
NoteBodyStyle2: 2
ColourLight1: 255,113,165,150
""".strip(),
        encoding="utf-8",
    )
    parsed = parse_skin_ini(skin)
    section = parsed.mania_for_keycount(4)

    assert section is not None
    assert section.note_body_style_by_column == {2: 2}
    assert section.colour_light[1] == (255, 113, 165, 150)
    assert legacy_note_body_style(section, 2, 2.4) == LEGACY_NOTE_BODY_STRETCH
    assert legacy_note_body_style(section, 2, 2.7) == LEGACY_NOTE_BODY_REPEAT_TOP
    assert (
        legacy_note_body_style(section, 1, 2.7)
        == LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM
    )
    assert legacy_note_body_style(ManiaSection(keys=4), 0, 2.7) == 3
    assert legacy_note_body_style(
        ManiaSection(keys=4, note_body_style=1), 0, 2.7,
    ) == LEGACY_NOTE_BODY_REPEAT_BOTTOM


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        (LEGACY_NOTE_BODY_REPEAT_TOP, [(0, 10, 0, 1), (10, 10, 0, 1),
                                          (20, 4, 0, 0.4)]),
        (LEGACY_NOTE_BODY_REPEAT_BOTTOM, [(0, 4, 0.6, 1), (4, 10, 0, 1),
                                       (14, 10, 0, 1)]),
        (LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM,
         [(0, 7, 0.3, 1), (7, 10, 0, 1), (17, 7, 0, 0.7)]),
    ],
)
def test_repeat_alignment_uses_uv_crops_not_squashed_tiles(style, expected) -> None:
    segments = legacy_hold_body_segments(0, 24, 10, style)
    actual = [
        (segment.y, segment.height, segment.source_bottom, segment.source_top)
        for segment in segments
    ]
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed == pytest.approx(wanted)
    # A partial destination segment samples the same fraction of the source;
    # it is not the whole pattern resized into a shorter rectangle.
    assert any(
        segment.height < 10
        and segment.source_top - segment.source_bottom < 1
        for segment in segments
    )


def test_stretch_is_one_full_source_draw() -> None:
    segments = legacy_hold_body_segments(5, 24, 10, LEGACY_NOTE_BODY_STRETCH)
    assert len(segments) == 1
    assert (
        segments[0].y,
        segments[0].height,
        segments[0].source_bottom,
        segments[0].source_top,
    ) == pytest.approx((5, 24, 0, 1))


@pytest.mark.slow
def test_uv_crop_samples_pattern_slice_instead_of_squashing(tmp_path) -> None:
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    skin = tmp_path / "skin"
    skin.mkdir()
    (skin / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 1\nNoteImage0L: body\n",
        encoding="utf-8",
    )
    pattern = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    for y in range(2, 4):
        for x in range(4):
            pattern.putpixel((x, y), (0, 0, 255, 255))
    pattern.save(skin / "body.png")

    with HeadlessGl(width=32, height=32) as gl:
        renderer = FrameRenderer(
            RenderContext(
                ctx=gl.ctx,
                fbo=gl.fbo,
                width=32,
                height=32,
                key_count=1,
            ),
            skin_dir=skin,
        )
        body_idx = renderer.atlas.column_slot_index("note_hold_body", 0)

        def sample(source_bottom, source_top):
            gl.fbo.use()
            gl.fbo.clear(0, 0, 0, 1)
            gl.ctx.enable(moderngl.BLEND)
            renderer._draw_sprite_idx_cropped_y(
                body_idx,
                0,
                0,
                32,
                32,
                (1, 1, 1, 1),
                source_bottom=source_bottom,
                source_top=source_top,
            )
            renderer._flush_sprite_batch()
            return tuple(gl.fbo.read(
                viewport=(16, 16, 1, 1),
                components=3,
            ))

        # Source fractions are bottom-to-top: bottom half is blue, top red.
        assert sample(0.0, 0.5)[2] > 240
        assert sample(0.5, 1.0)[0] > 240


class _StageLightAtlas:
    source = "user"

    def global_source(self, _slot):
        return self.source

    def frame_count(self, _slot):
        return 3

    def index_of(self, _slot):
        return 10

    def global_native_size(self, _slot):
        return 20.0, 40.0


def _stage_renderer(*, key_count=1, upside_down=False) -> FrameRenderer:
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720, key_count=key_count)
    renderer.atlas = _StageLightAtlas()
    renderer.mania_section = ManiaSection(keys=key_count, light_position=400)
    renderer.col_x = tuple(100 + 60 * c for c in range(key_count))
    renderer.col_w = tuple(50 for _ in range(key_count))
    renderer.upside_down = upside_down
    renderer._is_argon_default = lambda: False
    renderer._stage_light_fps = lambda _frames: 10.0
    renderer._stage_light_tint = lambda c: (1.0, 0.5 + c * 0.1, 0.25)
    renderer.draws = []
    renderer._draw_sprite_idx = lambda *args: renderer.draws.append(args)
    return renderer


def _stage_scene(time_ms, *held, release_ages=None):
    if release_ages is None:
        release_ages = tuple(-1 for _ in held)
    return SimpleNamespace(
        t_ms=time_ms,
        keys_held=tuple(held),
        key_release_age_ms=tuple(release_ages),
    )


def test_stage_light_press_hold_release_and_repress_are_source_driven() -> None:
    renderer = _stage_renderer()
    initial = _stage_scene(0, False)
    FrameRenderer._draw_stage_lights(renderer, initial)
    assert renderer.draws == []

    # Animation phase is global-clock driven: pressing at 250ms starts on
    # frame 2 rather than resetting to frame 0.
    pressed = _stage_scene(250, True)
    original_keys = pressed.keys_held
    FrameRenderer._draw_stage_lights(renderer, pressed)
    assert renderer.draws[-1][0] == 12
    assert renderer.draws[-1][1:5] == pytest.approx((100, 120, 50, 37.5))
    assert renderer.draws[-1][-1] == pytest.approx((1.0, 0.5, 0.25, 1.0))
    assert pressed.keys_held is original_keys

    # Held animation continues on the same absolute clock.
    renderer.draws.clear()
    FrameRenderer._draw_stage_lights(renderer, _stage_scene(350, True))
    assert renderer.draws[-1][0] == 10
    assert renderer.draws[-1][-1][-1] == 1.0

    # Release changes only alpha/Y scale; the frame keeps progressing.
    renderer.draws.clear()
    FrameRenderer._draw_stage_lights(
        renderer, _stage_scene(360, False, release_ages=(0,)),
    )
    assert renderer.draws[-1][0] == 10
    FrameRenderer._draw_stage_lights(
        renderer, _stage_scene(485, False, release_ages=(125,)),
    )
    midpoint = renderer.draws[-1]
    assert midpoint[0] == 11
    assert midpoint[2:5] == pytest.approx((120, 50, 18.75))
    assert midpoint[-1][-1] == pytest.approx(0.5)

    # Re-press cancels the envelope at the current clock phase, not frame 0.
    renderer.draws.clear()
    FrameRenderer._draw_stage_lights(renderer, _stage_scene(500, True))
    assert renderer.draws[-1][0] == 12
    assert renderer.draws[-1][4] == pytest.approx(37.5)
    assert renderer.draws[-1][-1][-1] == 1.0

    FrameRenderer._draw_stage_lights(
        renderer, _stage_scene(510, False, release_ages=(0,)),
    )
    renderer.draws.clear()
    FrameRenderer._draw_stage_lights(
        renderer, _stage_scene(760, False, release_ages=(250,)),
    )
    assert renderer.draws == []


_SUBFRAME_TAP = (
    KeyEvent(time_ms=1005, keys_held=1),
    KeyEvent(time_ms=1020, keys_held=0),
)


def _subframe_tap_scene(time_ms: int):
    _presses, releases = _key_edges_per_col(_SUBFRAME_TAP, 1)
    mask = 0
    for event in _SUBFRAME_TAP:
        if event.time_ms > time_ms:
            break
        mask = event.keys_held
    return _stage_scene(
        time_ms,
        bool(mask & 1),
        release_ages=_key_release_ages_at(releases, time_ms, 1),
    )


def _subframe_tap_pipeline_scene(time_ms: int):
    presses, releases = _key_edges_per_col(_SUBFRAME_TAP, 1)
    replay = SimpleNamespace(
        key_events=_SUBFRAME_TAP,
        mods=0,
        max_combo=0,
        accuracy=100.0,
        count_geki=0,
        count_300=0,
        count_katu=0,
        count_100=0,
        count_50=0,
        count_miss=0,
        mania_acc_weight=300,
    )
    plan = SimpleNamespace(
        replay=replay,
        modded=SimpleNamespace(notes=()),
        key_count=1,
        gameplay_end_ms=2000,
        results_start_ms=3000,
        effective_approach_ms=600,
        visual_mods=VisualMods(),
        judged_hits={},
        sv_for_note={},
        timing_points=(),
        sv_table=(),
        note_times=(),
        max_hold_dur_ms=0,
        judgment_timeline=[],
        hit_error_windows=(0, 0, 0, 0, 0),
        total_quality=0,
        kiai_ranges=[],
        per_column_ur=(0,),
        miss_break_times=[],
        press_iters=presses,
        release_iters=releases,
        acronyms=(),
        player_pp=0,
        max_pp=0,
        stars=0,
        mania_mw=300,
        n_scoring=0,
        max_combo_portion=0,
        score_scale=1,
        score_final=None,
    )
    scene, _score, _accuracy = build_frame_state(plan, time_ms, 0, 100)
    return scene


def _last_subframe_draw(sample_times: tuple[int, ...]):
    renderer = _stage_renderer()
    for time_ms in sample_times:
        FrameRenderer._draw_stage_lights(
            renderer, _subframe_tap_scene(time_ms),
        )
    return renderer.draws[-1] if renderer.draws else None


def test_subframe_tap_uses_exact_release_timestamp_without_held_sample() -> None:
    presses, releases = _key_edges_per_col(_SUBFRAME_TAP, 1)
    assert presses == [[1005]]
    assert releases == [[1020]]
    assert _key_release_ages_at(releases, 1000, 1) == (-1,)
    assert _key_release_ages_at(releases, 1033, 1) == (13,)

    before = _subframe_tap_pipeline_scene(1000)
    released = _subframe_tap_pipeline_scene(1033)
    assert before.keys_held == (False,)
    assert before.key_release_age_ms == (-1,)
    assert released.keys_held == (False,)
    assert released.key_press_age_ms == (28,)
    assert released.key_release_age_ms == (13,)

    renderer = _stage_renderer()
    FrameRenderer._draw_stage_lights(renderer, before)
    assert renderer.draws == []
    FrameRenderer._draw_stage_lights(renderer, released)

    draw = renderer.draws[-1]
    amount = 1.0 - 13.0 / 250.0
    assert draw[0] == 11
    assert draw[1:5] == pytest.approx((100, 120, 50, 37.5 * amount))
    assert draw[-1][-1] == pytest.approx(amount)


def test_stage_light_direct_seek_and_output_fps_are_history_independent() -> None:
    direct = _last_subframe_draw((1033,))
    fps_30 = _last_subframe_draw((1000, 1033))
    fps_60 = _last_subframe_draw((1000, 1017, 1033))
    out_of_order = _last_subframe_draw((1100, 1033))

    assert direct is not None
    assert direct == fps_30 == fps_60 == out_of_order


def test_stage_light_has_no_synthetic_fallback() -> None:
    renderer = _stage_renderer()
    renderer.atlas.source = "bundle"
    renderer.named_draws = []
    renderer._draw_sprite = lambda *args: renderer.named_draws.append(args)

    FrameRenderer._draw_stage_lights(renderer, _stage_scene(0, True))

    assert renderer.draws == []
    assert renderer.named_draws == []


def test_stage_light_accepts_distinct_classic_fallback_tier() -> None:
    renderer = _stage_renderer()
    renderer.atlas.source = "classic"

    FrameRenderer._draw_stage_lights(renderer, _stage_scene(0, True))

    assert len(renderer.draws) == 1


def test_stage_light_columns_have_independent_release_state_but_shared_clock_phase() -> None:
    renderer = _stage_renderer(key_count=2)
    FrameRenderer._draw_stage_lights(renderer, _stage_scene(100, True, False))
    renderer.draws.clear()
    FrameRenderer._draw_stage_lights(
        renderer,
        _stage_scene(200, False, True, release_ages=(0, -1)),
    )
    renderer.draws.clear()
    FrameRenderer._draw_stage_lights(
        renderer,
        _stage_scene(325, False, True, release_ages=(125, -1)),
    )

    assert [draw[0] for draw in renderer.draws] == [10, 10]
    assert renderer.draws[0][-1][-1] == pytest.approx(0.5)
    assert renderer.draws[1][-1][-1] == 1.0


def test_stage_light_geometry_uses_native_height_and_orientation_anchor() -> None:
    down = legacy_stage_light_geometry(
        column_x=100,
        column_width=50,
        native_size=(20, 40),
        light_position=400,
        render_height=720,
        upside_down=False,
        vertical_scale=0.5,
    )
    up = legacy_stage_light_geometry(
        column_x=100,
        column_width=50,
        native_size=(20, 40),
        light_position=400,
        render_height=720,
        upside_down=True,
        vertical_scale=0.5,
    )
    wide = legacy_stage_light_geometry(
        column_x=100,
        column_width=100,
        native_size=(20, 40),
        light_position=400,
        render_height=720,
        upside_down=False,
        vertical_scale=0.5,
    )
    assert (down.y, down.height, down.anchor_y) == pytest.approx(
        (120, 18.75, 120),
    )
    assert (up.y, up.height, up.anchor_y) == pytest.approx(
        (581.25, 18.75, 600),
    )
    assert wide.width == 100
    assert wide.height == pytest.approx(down.height)


def test_stage_light_backwards_clock_keeps_deterministic_absolute_phase() -> None:
    presentation = legacy_stage_light_presentation(
        held=True, release_age_ms=None, time_ms=400, frame_count=3, fps=10,
    )
    assert presentation.frame == 1
    presentation = legacy_stage_light_presentation(
        held=True, release_age_ms=None, time_ms=250, frame_count=3, fps=10,
    )
    # Out-of-order evaluation uses the absolute clock and no renderer history.
    assert presentation.frame == 2
    assert presentation.alpha == 1


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 60.0), (120, 120.0), (0, 24.0), (-1, 24.0)],
)
def test_stage_light_fps_matches_lazer_decoder(configured, expected) -> None:
    section = ManiaSection(keys=4, light_frame_per_second=configured)
    assert legacy_stage_light_fps(section) == expected


def test_general_animation_framerate_does_not_affect_stage_light_fps() -> None:
    renderer = object.__new__(FrameRenderer)
    renderer.mania_section = ManiaSection(keys=4)
    renderer.skin_ini = SimpleNamespace(animation_framerate=7)

    assert FrameRenderer._stage_light_fps(renderer, frame_count=13) == 60.0


def test_stage_light_lookup_precedence_and_explicit_transparency(tmp_path) -> None:
    skin = tmp_path / "skin"
    beatmap = tmp_path / "beatmap"
    skin.mkdir()
    beatmap.mkdir()
    Image.new("RGBA", (2, 3), (0, 255, 0, 255)).save(
        skin / "mania-stage-light.png",
    )
    Image.new("RGBA", (4, 5), (255, 0, 0, 255)).save(
        beatmap / "mania-stage-light.png",
    )

    frames, source = SpriteAtlas._resolve_global(
        "stage_light", skin_dir=skin, beatmap_dir=beatmap, section=None,
    )
    assert source == "beatmap"
    assert frames[0].getpixel((0, 0)) == (255, 0, 0, 255)

    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(skin / "blank.png")
    frames, source = SpriteAtlas._resolve_global(
        "stage_light",
        skin_dir=skin,
        beatmap_dir=None,
        section=ManiaSection(keys=4, stage_light="blank"),
    )
    assert source == "user"
    assert frames[0].getchannel("A").getbbox() is None


def test_missing_stage_light_uses_authoritative_classic_asset() -> None:
    frames, source = SpriteAtlas._resolve_global(
        "stage_light", skin_dir=None, beatmap_dir=None, section=None,
    )

    assert source == "classic"
    assert frames[0].size == (100, 500)
    assert frames[0].info["scale_adjust"] == 2
    assert frames[0].getpixel((50, 250)) == (255, 255, 255, 128)


def test_synthetic_r3d_stage_light_is_never_a_classic_fallback(
    tmp_path, monkeypatch,
) -> None:
    Image.new("RGBA", (128, 32), (180, 220, 255, 90)).save(
        tmp_path / "stage_light.png",
    )
    monkeypatch.setattr(atlas_module, "SPRITES_DIR", tmp_path)

    frames, source = SpriteAtlas._resolve_global(
        "stage_light", skin_dir=None, beatmap_dir=None, section=None,
    )

    assert source == "missing"
    assert frames[0].getchannel("A").getbbox() is None


def test_hold_body_animation_uses_30ms_active_clock_and_resets() -> None:
    renderer = object.__new__(FrameRenderer)
    renderer._legacy_hold_body_started_ms = {}
    renderer._legacy_hold_body_last_ms = {}
    note = SimpleNamespace(
        column=0,
        time_ms=100,
        head_y_fraction=1.0,
        tail_y_fraction=0.5,
    )

    assert renderer._legacy_hold_body_frame_index(
        _stage_scene(100, False), note, 3,
    ) == 0
    assert renderer._legacy_hold_body_frame_index(
        _stage_scene(110, True), note, 3,
    ) == 0
    assert renderer._legacy_hold_body_frame_index(
        _stage_scene(140, True), note, 3,
    ) == 1
    assert renderer._legacy_hold_body_frame_index(
        _stage_scene(170, True), note, 3,
    ) == 2
    assert renderer._legacy_hold_body_frame_index(
        _stage_scene(175, False), note, 3,
    ) == 0
    assert renderer._legacy_hold_body_frame_index(
        _stage_scene(180, True), note, 3,
    ) == 0


class _HoldAtlas:
    def has_skin_notes(self):
        return True

    def has_skin_note(self, _column):
        return True

    def has_skin_hold(self, _column):
        return True

    def column_aspect(self, kind, _column):
        return 1.0 if kind == "note_hold_body" else 2.0

    def column_native_size(self, kind, _column):
        return (40, 51.2) if kind == "note_hold_body" else (40, 20)

    def column_slot_index(self, kind, _column):
        return {
            "note_hold_body": 10,
            "note_hold_head": 20,
            "note_hold_tail": 30,
        }[kind]

    def column_frame_count(self, _kind, _column):
        return 1


def _hold_renderer(style: int) -> FrameRenderer:
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=800, height=600, key_count=1)
    renderer.atlas = _HoldAtlas()
    renderer.mania_section = ManiaSection(keys=1, note_body_style=style)
    renderer.skin_ini = SimpleNamespace(legacy_version=2.7)
    renderer.col_x = (100,)
    renderer.col_w = (40,)
    renderer.col_w_uniform = 40
    renderer.pf_x = 100
    renderer.pf_w = 40
    renderer.receptor_centre_y_gl = 100
    renderer.upside_down = False
    renderer._is_argon_default = lambda: False
    renderer._legacy_hold_body_started_ms = {}
    renderer._legacy_hold_body_last_ms = {}
    renderer.normal_draws = []
    renderer.direct_draws = []
    renderer._draw_legacy_column_direct = lambda *args, **kw: renderer.direct_draws.append((args, kw))
    renderer.cropped_draws = []
    renderer._draw_sprite_idx = lambda *args: renderer.normal_draws.append(args)
    renderer._draw_sprite_idx_cropped_y = (
        lambda *args, **kwargs: renderer.cropped_draws.append((args, kwargs))
    )
    return renderer


def _hold_scene():
    return SimpleNamespace(
        t_ms=100,
        keys_held=(False,),
        visible_notes=(VisibleNote(
            column=0,
            is_hold=True,
            y_fraction=1.0,
            head_y_fraction=1.0,
            tail_y_fraction=0.5,
            time_ms=100,
        ),),
    )


def test_hold_draw_path_uses_native_direct_stretch_or_repeat_slices() -> None:
    stretch = _hold_renderer(LEGACY_NOTE_BODY_STRETCH)
    FrameRenderer._draw_notes(stretch, _hold_scene())
    assert stretch.cropped_draws == []
    assert len(stretch.normal_draws) == 2  # caps keep their atlas batching
    assert stretch.direct_draws[0][0] == (
        "column/note_hold_body/0", 100, 110, 40, 230,
    )
    assert stretch.direct_draws[0][1]["repeat_y"] is False

    repeated = _hold_renderer(LEGACY_NOTE_BODY_REPEAT_BOTTOM)
    FrameRenderer._draw_notes(repeated, _hold_scene())
    assert len(repeated.normal_draws) == 2
    assert len(repeated.direct_draws) == 6
    first_args, first_kwargs = repeated.direct_draws[0]
    assert first_args[1:5] == pytest.approx((100, 110, 40, 30))
    assert first_kwargs["source_bottom"] == pytest.approx(0.25)
    assert first_kwargs["source_top"] == 1.0
    assert first_kwargs["frame_index"] == 0
    assert first_kwargs["repeat_y"] is True
