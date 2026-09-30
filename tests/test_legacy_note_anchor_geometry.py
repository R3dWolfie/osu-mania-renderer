"""Regression coverage for legacy/custom mania note edge anchoring."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
from osu_mania_renderer_v2.gpu.legacy_mania import (
    LEGACY_NOTE_BODY_REPEAT_BOTTOM,
    LEGACY_NOTE_BODY_REPEAT_TOP,
    LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM,
    LEGACY_NOTE_BODY_STRETCH,
)
from osu_mania_renderer_v2.gpu.legacy_note_geometry import (
    legacy_hold_geometry,
    legacy_note_draw_y,
)
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer
from osu_mania_renderer_v2.render.scene import VisibleNote
from osu_mania_renderer_v2.wiki_elements.notes import _draw_notes_body


@pytest.mark.parametrize(
    ("upside_down", "is_tail", "expected"),
    [
        (False, False, 400),
        (False, True, 360),
        (True, False, 360),
        (True, True, 400),
    ],
    ids=("down-tap-head", "down-tail", "up-tap-head", "up-tail"),
)
def test_legacy_note_draw_y_uses_lazer_edge_anchors(
    upside_down, is_tail, expected,
):
    assert legacy_note_draw_y(
        400, 40, upside_down=upside_down, is_tail=is_tail,
    ) == expected


@pytest.mark.parametrize(
    ("upside_down", "is_tail", "expected"),
    [
        (False, False, 400),
        (False, True, 361),
        (True, False, 361),
        (True, True, 400),
    ],
)
def test_legacy_note_draw_y_truncates_non_integer_geometry_deterministically(
    upside_down, is_tail, expected,
):
    assert legacy_note_draw_y(
        400.75, 39.5, upside_down=upside_down, is_tail=is_tail,
    ) == expected


@pytest.mark.parametrize(
    (
        "upside_down",
        "y_tail",
        "expected_head_y",
        "expected_tail_y",
        "expected_body_y",
        "expected_body_h",
    ),
    [
        (False, 450, 400, 410, 420, 10),
        (True, 200, 360, 200, 220, 160),
    ],
    ids=("downscroll", "upscroll"),
)
def test_legacy_hold_geometry_joins_edge_anchored_cap_centres(
    upside_down,
    y_tail,
    expected_head_y,
    expected_tail_y,
    expected_body_y,
    expected_body_h,
):
    geometry = legacy_hold_geometry(
        400, y_tail, 40, 40, upside_down=upside_down,
    )

    assert geometry.head_draw_y == expected_head_y
    assert geometry.tail_draw_y == expected_tail_y
    assert geometry.body_y == expected_body_y
    assert geometry.body_height == expected_body_h


@pytest.mark.parametrize(
    ("upside_down", "y_tail"),
    [(False, 500), (True, 100)],
    ids=("downscroll", "upscroll"),
)
def test_legacy_hold_body_overlaps_each_separated_cap_by_half(
    upside_down, y_tail,
):
    geometry = legacy_hold_geometry(
        400, y_tail, 40, 40, upside_down=upside_down,
    )
    body_top = geometry.body_y + geometry.body_height

    def body_overlap(cap_y):
        return max(
            0,
            min(body_top, cap_y + 40) - max(geometry.body_y, cap_y),
        )

    assert body_overlap(geometry.head_draw_y) == 20
    assert body_overlap(geometry.tail_draw_y) == 20
    assert body_overlap(geometry.head_draw_y) < 40
    assert body_overlap(geometry.tail_draw_y) < 40


class _LegacyNoteAtlas:
    _indices = {
        "note_tap": 10,
        "note_hold_body": 20,
        "note_hold_head": 30,
        "note_hold_tail": 40,
    }

    def has_skin_notes(self):
        return True

    def has_skin_note(self, _column):
        return True

    def has_skin_hold(self, _column):
        return True

    def column_aspect(self, kind, _column):
        return 2.0 if kind != "note_hold_body" else 1.0

    def column_slot_index(self, kind, _column):
        return self._indices[kind]

    def column_frame_count(self, _kind, _column):
        return 1

    def global_aspect(self, _kind):
        return 1.667


def _scene(note):
    return SimpleNamespace(
        t_ms=0,
        visible_notes=(note,),
        keys_held=(False,),
    )


def _record_legacy_draws(
    render_path, note, *, upside_down, note_body_style=0,
):
    indexed_draws = []
    named_draws = []
    common = {
        "atlas": _LegacyNoteAtlas(),
        "height": 500,
        "key_count": 1,
        "receptor_centre_y_gl": 400,
        "upside_down": upside_down,
        "col_x": (100,),
        "col_w": (80,),
        "mania_section": ManiaSection(
            keys=1, note_body_style=note_body_style,
        ),
        "skin_ini": SimpleNamespace(legacy_version=2.7),
    }

    if render_path == "monolithic":
        renderer = object.__new__(FrameRenderer)
        renderer.rc = SimpleNamespace(width=800, height=500, key_count=1)
        renderer.pf_x = 100
        renderer.pf_w = 80
        renderer.col_w_uniform = 80
        for name, value in common.items():
            if name not in {"height", "key_count"}:
                setattr(renderer, name, value)
        renderer._is_argon_default = lambda: False
        renderer._draw_sprite_idx = lambda *args: indexed_draws.append(args)
        renderer._draw_sprite_idx_cropped_y = (
            lambda *args, **_kwargs: indexed_draws.append(args)
        )
        renderer._draw_sprite = lambda *args: named_draws.append(args)
        FrameRenderer._draw_notes(renderer, _scene(note))
    else:
        renderer = object.__new__(FrameRenderer)
        renderer._draw_sprite_idx_cropped_y = (
            lambda *args, **_kwargs: indexed_draws.append(args)
        )
        ctx = SimpleNamespace(
            **common,
            fr=renderer,
            scene=_scene(note),
            persistent={},
            draw_sprite_idx=lambda *args: indexed_draws.append(args),
            draw_sprite=lambda *args: named_draws.append(args),
        )
        _draw_notes_body(ctx)

    return indexed_draws, named_draws


@pytest.mark.parametrize("render_path", ["monolithic", "wiki-elements"])
@pytest.mark.parametrize(
    ("upside_down", "expected_y"),
    [
        (False, [420, 410, 400]),
        (True, [380, 370, 360]),
    ],
    ids=("downscroll", "upscroll"),
)
def test_legacy_tap_and_ghost_draws_use_their_edge_anchors(
    render_path, upside_down, expected_y,
):
    note = VisibleNote(
        column=0,
        is_hold=False,
        y_fraction=1.0,
        head_y_fraction=1.0,
        tail_y_fraction=1.0,
    )

    indexed_draws, named_draws = _record_legacy_draws(
        render_path, note, upside_down=upside_down,
    )

    assert [draw[0] for draw in indexed_draws] == [10, 10, 10]
    assert [draw[2] for draw in indexed_draws] == expected_y
    assert named_draws == []


@pytest.mark.parametrize("render_path", ["monolithic", "wiki-elements"])
@pytest.mark.parametrize(
    "body_style",
    [
        LEGACY_NOTE_BODY_STRETCH,
        LEGACY_NOTE_BODY_REPEAT_TOP,
        LEGACY_NOTE_BODY_REPEAT_BOTTOM,
        LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM,
    ],
    ids=("stretch", "repeat-top", "repeat-bottom", "repeat-both"),
)
@pytest.mark.parametrize(
    (
        "upside_down",
        "y_tail",
        "expected_head_y",
        "expected_tail_y",
        "expected_body_y",
        "expected_body_h",
    ),
    [
        (False, 450, 400, 410, 420, 10),
        (True, 200, 360, 200, 220, 160),
    ],
    ids=("downscroll", "upscroll"),
)
def test_legacy_hold_caps_keep_edge_anchors_while_body_joins_cap_centres(
    render_path,
    body_style,
    upside_down,
    y_tail,
    expected_head_y,
    expected_tail_y,
    expected_body_y,
    expected_body_h,
):
    note = VisibleNote(
        column=0,
        is_hold=True,
        y_fraction=1.0,
        head_y_fraction=1.0,
        tail_y_fraction=0.5,
    )
    y_head = 400

    indexed_draws, named_draws = _record_legacy_draws(
        render_path,
        note,
        upside_down=upside_down,
        note_body_style=body_style,
    )

    *body_draws, head, tail = indexed_draws
    assert body_draws
    assert all(draw[0] == 20 for draw in body_draws)
    assert all(draw[3] == 80 for draw in body_draws)
    assert body_draws[0][2] == expected_body_y
    assert sum(draw[4] for draw in body_draws) == expected_body_h
    assert body_draws[-1][2] + body_draws[-1][4] == (
        expected_body_y + expected_body_h
    )

    assert head[0] == 30
    assert tail[0] == 40
    assert head[2] == expected_head_y
    assert tail[2] == expected_tail_y
    assert head[2] == legacy_note_draw_y(
        y_head, head[4], upside_down=upside_down,
    )
    assert tail[2] == legacy_note_draw_y(
        y_tail, tail[4], upside_down=upside_down, is_tail=True,
    )

    body_endpoints = {
        expected_body_y,
        expected_body_y + expected_body_h,
    }
    assert body_endpoints == {
        head[2] + head[4] / 2,
        tail[2] + tail[4] / 2,
    }
    assert abs((head[2] + head[4] / 2) - y_head) == head[4] / 2
    assert abs((tail[2] + tail[4] / 2) - y_tail) == tail[4] / 2
    assert named_draws == []
