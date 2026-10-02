"""Native key/body sources and stable's hold-body source-coordinate contract."""
from __future__ import annotations

import os
from types import SimpleNamespace

import moderngl
import numpy as np
import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection, parse_skin_ini
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas, column_direct_name
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.legacy_mania import (
    LegacyHoldBodySegment,
    legacy_clip_y_segment,
    legacy_hold_body_segments,
    legacy_note_body_style,
)
from osu_mania_renderer_v2.gpu.legacy_note_geometry import legacy_note_height
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    RenderContext,
    configure_direct_texture_sampling,
)
from osu_mania_renderer_v2.render.scene import VisibleNote
from osu_mania_renderer_v2.wiki_elements.notes import _draw_notes_body, _receptors


class _TextureProbe:
    def build_mipmaps(self):
        self.mipmaps = True


class _ContextProbe:
    info = {"GL_MAX_TEXTURE_SIZE": 65536}

    def __init__(self):
        self.uploads = []

    def texture_array(self, *, size, components, data):
        self.uploads.append((size, components, data))
        return _TextureProbe()


@pytest.fixture
def native_skin(tmp_path):
    # End artwork is in L itself. The tail is deliberately transparent.
    section = ManiaSection(
        keys=1, key_image={0: "key"}, key_image_d={0: "pressed"},
        note_image={0: "cap"}, note_image_h={0: "cap"},
        note_image_l={0: "body"}, note_image_t={0: "transparent"},
    )
    key = Image.new("RGBA", (150, 768), (0, 0, 0, 0))
    for row in range(768):
        key.putpixel((70, row), (row % 256, 40, 50, 255))
    key.save(tmp_path / "key.png")
    pressed = key.copy()
    pressed.putpixel((70, 300), (10, 220, 250, 255))
    pressed.save(tmp_path / "pressed.png")
    body = Image.new("RGBA", (7, 2048), (0, 200, 0, 255))
    body.paste((240, 0, 0, 255), (0, 0, 7, 64))
    body.paste((0, 0, 240, 255), (0, 1984, 7, 2048))
    body.save(tmp_path / "body.png")
    Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(tmp_path / "cap.png")
    Image.new("RGBA", (1, 1)).save(tmp_path / "transparent.png")
    return tmp_path, section, key, body


def _atlas(fixture, **kwargs):
    path, section, *_ = fixture
    return SpriteAtlas.load(
        _ContextProbe(), skin_dir=path, key_count=1, mania_section=section, **kwargs,
    )


def _renderer(atlas, section, *, up=False, raw=None, version=2.7):
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(width=1024, height=768, key_count=1, ctx=_ContextProbe())
    fr.atlas = atlas
    fr.col_x = (100,)
    fr.col_w = (140,)
    fr.col_w_uniform = 140
    fr.pf_x, fr.pf_w = 100, 140
    fr.receptor_centre_y_gl = 668 if up else 100
    fr.upside_down = up
    fr.mania_section = ManiaSection(**{**section.__dict__, "note_body_style": raw})
    fr.skin_ini = SimpleNamespace(legacy_version=version)
    fr._is_argon_default = lambda: False
    fr._draw_custom_legacy_lighting = lambda *a, **kw: None
    fr._legacy_hold_body_started_ms = {}
    fr._legacy_hold_body_last_ms = {}
    fr.direct_draws = []
    fr.indexed_draws = []
    fr.cropped_draws = []
    fr._draw_direct = lambda *a, **kw: fr.direct_draws.append((a, kw))
    fr._draw_sprite_idx = lambda *a: fr.indexed_draws.append(a)
    fr._draw_sprite_idx_cropped_y = lambda *a, **kw: fr.cropped_draws.append((a, kw))
    fr._draw_sprite = lambda *a: None
    return fr


def _wiki_context(fr, scene):
    return SimpleNamespace(
        fr=fr, scene=scene, atlas=fr.atlas, width=fr.rc.width, height=fr.rc.height,
        key_count=1, col_x=fr.col_x, col_w=fr.col_w,
        receptor_centre_y_gl=fr.receptor_centre_y_gl, upside_down=fr.upside_down,
        mania_section=fr.mania_section, skin_ini=fr.skin_ini, persistent={},
        draw_sprite_idx=fr._draw_sprite_idx, draw_sprite=fr._draw_sprite,
    )


def test_native_sources_retain_every_pixel_and_animation_frame(native_skin):
    path, section, key, body = native_skin
    body.save(path / "body-0.png")
    second = Image.new("RGBA", body.size, (10, 20, 30, 255))
    second.save(path / "body-1.png")
    atlas = _atlas(native_skin)
    for kind in ("receptor_off", "receptor_on"):
        name = column_direct_name(kind, 0)
        assert atlas.direct_image(name).size == (150, 768)
        expected = key if kind == "receptor_off" else Image.open(path / "pressed.png")
        assert atlas.direct_image(name).tobytes() == expected.tobytes()
    assert atlas.direct_image(column_direct_name("receptor_off", 0)).tobytes() != \
        atlas.direct_image(column_direct_name("receptor_on", 0)).tobytes()
    name = column_direct_name("note_hold_body", 0)
    assert atlas.direct_frame_count(name) == 2
    assert atlas.direct_frame_image(name, 0).tobytes() == body.tobytes()
    assert atlas.direct_frame_image(name, 1).tobytes() == second.tobytes()
    with pytest.raises(IndexError):
        atlas.direct_frame_image(name, 2)
    assert atlas.column_native_size("note_hold_body", 0) == (7, 2048)


def test_2x_pixels_are_retained_while_design_size_stays_the_same(native_skin):
    path, section, key, body = native_skin
    key.resize((300, 1536)).save(path / "key@2x.png")
    body.resize((14, 4096)).save(path / "body@2x.png")
    atlas = _atlas(native_skin)
    assert atlas.direct_image(column_direct_name("receptor_off", 0)).size == (300, 1536)
    assert atlas.column_native_size("receptor_off", 0) == (150, 768)
    assert atlas.direct_image(column_direct_name("note_hold_body", 0)).size == (14, 4096)
    assert atlas.column_native_size("note_hold_body", 0) == (7, 2048)
    fr = _renderer(atlas, section)
    fr._draw_legacy_key(0, held=False)
    assert fr.direct_draws[0][0][1:] == (100, 0, 140, 768)


def test_native_source_priority_is_the_existing_resolver_priority(native_skin, tmp_path):
    path, section, *_ = native_skin
    beatmap = tmp_path / "map"
    beatmap.mkdir()
    Image.new("RGBA", (10, 500), (20, 30, 40, 255)).save(beatmap / "mania-keyS.png")
    atlas = _atlas(native_skin, beatmap_dir=beatmap)
    assert atlas.column_source("receptor_off", 0) == "user"  # explicit named override
    assert atlas.direct_image(column_direct_name("receptor_off", 0)).size == (150, 768)
    atlas = SpriteAtlas.load(_ContextProbe(), skin_dir=path, key_count=1, beatmap_dir=beatmap)
    assert atlas.column_source("receptor_off", 0) == "beatmap"
    assert atlas.direct_image(column_direct_name("receptor_off", 0)).size == (10, 500)
    atlas = SpriteAtlas.load(_ContextProbe(), key_count=1)
    for kind in ("receptor_off", "receptor_on", "note_hold_body"):
        assert atlas.direct_image(column_direct_name(kind, 0)) is not None


@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("path", ["monolithic", "wiki"])
def test_key_draws_use_native_authority_and_swaps_keep_geometry(native_skin, up, path):
    _, section, *_ = native_skin
    fr = _renderer(_atlas(native_skin), section, up=up)
    for held in (False, True):
        scene = SimpleNamespace(keys_held=(held,))
        if path == "monolithic":
            fr._draw_receptors(scene)
        else:
            _receptors(_wiki_context(fr, scene))
    assert fr.indexed_draws == []  # never consumes the 256-square receptor layer
    off, on = fr.direct_draws
    assert off[0][0] == "column/receptor_off/0"
    assert on[0][0] == "column/receptor_on/0"
    assert off[0][1:] == on[0][1:] == (100, 0, 140, 768)
    assert off[1]["source_bottom"] == (1.0 if up else 0.0)
    assert off[1]["source_top"] == (0.0 if up else 1.0)


@pytest.mark.parametrize("raw,expected", [(None, 3), (0, 0), (1, 3), (2, 2), (3, 3), (4, 4), (99, 3)])
def test_stable_raw_contract_and_version_gate(raw, expected):
    section = ManiaSection(keys=2, note_body_style=raw)
    assert legacy_note_body_style(section, 0, 2.5) == expected
    assert legacy_note_body_style(section, 0, 2.4) == 0


def test_parser_preserves_stable_numeric_and_named_values(tmp_path):
    (tmp_path / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 4\nNoteBodyStyle: RepeatTopAndBottom\n"
        "NoteBodyStyle0: 1\nNoteBodyStyle1: Stretch\nNoteBodyStyle2: RepeatTop\n"
        "NoteBodyStyle3: invalid\nWidthForNoteHeightScale: 25.5\n",
    )
    section = parse_skin_ini(tmp_path).mania_for_keycount(4)
    assert section.note_body_style == 4
    assert section.note_body_style_by_column == {0: 1, 1: 0, 2: 2}
    assert [legacy_note_body_style(section, col, 2.7) for col in range(4)] == [3, 0, 2, 4]
    assert section.width_for_note_height_scale == 25.5


@pytest.mark.parametrize("value,raw", [
    (" repeatbottom ", 3), ("+2", 2), ("-1", -1),
    ("Stretch, RepeatTop", 2), ("RepeatTop, RepeatBottom", 3),
    ("RepeatTopAndBottom, RepeatTop", 6),
    ("2147483647", 2147483647), ("2147483648", None),
    ("-2147483649", None), ("2_0", None), ("2.0", None),
    ("RepeatTop, unknown", None),
])
def test_parser_matches_stable_enum_parse_and_invalid_override_fallback(tmp_path, value, raw):
    (tmp_path / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 1\nNoteBodyStyle: 4\n"
        f"NoteBodyStyle0: {value}\n",
    )
    section = parse_skin_ini(tmp_path).mania_for_keycount(1)
    assert section.note_body_style_by_column.get(0) == raw
    expected = 4 if raw is None else (raw if raw in (0, 2, 3, 4) else 3)
    assert legacy_note_body_style(section, 0, 2.7) == expected


@pytest.mark.parametrize("raw", [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("up", [False, True])
def test_tall_body_source_end_and_both_renderer_paths(native_skin, raw, up):
    _, section, *_ = native_skin
    atlas = _atlas(native_skin)
    note = VisibleNote(column=0, is_hold=True, y_fraction=1, head_y_fraction=1,
                       tail_y_fraction=0.5, time_ms=100)
    scene = SimpleNamespace(t_ms=100, visible_notes=(note,), keys_held=(False,))
    outputs = []
    for path in ("monolithic", "wiki"):
        fr = _renderer(atlas, section, up=up, raw=raw)
        if path == "monolithic":
            fr._draw_notes(scene)
        else:
            _draw_notes_body(_wiki_context(fr, scene))
        assert all(draw[0] != atlas.column_slot_index("note_hold_body", 0)
                   for draw in fr.indexed_draws)
        assert len(fr.direct_draws) == 1  # native tile is 2048 high, NOT 140 / (7/2048)
        args, kw = fr.direct_draws[0]
        assert args[0] == "column/note_hold_body/0"
        length = args[4]
        bottom, top = kw["source_bottom"], kw["source_top"]
        if up:
            bottom, top = top, bottom
        if raw == 0:
            assert (bottom, top) == (0, 1)
        elif raw == 2:
            assert bottom == 0
            assert top == pytest.approx(length / 2048)
        elif raw == 4:
            assert bottom == pytest.approx((1 - length / 2048) / 2)
            assert top == pytest.approx((1 + length / 2048) / 2)
        else:
            # Tail edge selects image row zero, including its authored red marker.
            assert top == 1
            assert bottom == pytest.approx(1 - length / 2048)
        assert kw["note_cover"] is True
        outputs.append((fr.direct_draws, fr.indexed_draws))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("style", [0, 2, 3, 4])
def test_repeated_segments_mirror_without_losing_or_squashing_partial_rows(style):
    down = legacy_hold_body_segments(10, 250, 100, style)
    up = legacy_hold_body_segments(10, 250, 100, style, upside_down=True)
    assert sum(s.height for s in down) == pytest.approx(250)
    for a, b in zip(down, reversed(up), strict=True):
        assert a.height == b.height
        assert a.y - 10 == pytest.approx(260 - b.y - b.height)
        assert a.source_bottom == b.source_top
        assert a.source_top == b.source_bottom
        if style:
            assert a.height == pytest.approx((a.source_top - a.source_bottom) * 100)


@pytest.mark.parametrize("configured,expected", [(None, 30), (0, 30), (-1, 30), (20, 15), (80, 60)])
def test_width_for_note_height_scale_only_changes_y(configured, expected):
    assert legacy_note_height(2, minimum_column_width=60, configured_width=configured,
                              render_height=720) == expected


def test_direct_column_upload_is_cached_and_uses_stable_linear_filter(native_skin):
    _, section, *_ = native_skin
    fr = _renderer(_atlas(native_skin), section)
    fr._direct_arr_cache = {}
    name = column_direct_name("note_hold_body", 0)
    texture = fr._direct_texture_array(name, frame_index=0)
    assert fr._direct_texture_array(name, frame_index=0) is texture
    assert len(fr.rc.ctx.uploads) == 1
    assert fr.rc.ctx.uploads[0][0] == (7, 2048, 1)
    assert texture.filter == (moderngl.LINEAR, moderngl.LINEAR)
    assert texture.repeat_x is False and texture.repeat_y is False
    assert not hasattr(texture, "mipmaps")
    configure_direct_texture_sampling("column/receptor_off/0", texture)
    assert texture.filter == (moderngl.LINEAR, moderngl.LINEAR)


@pytest.mark.parametrize("up", [False, True])
def test_native_tall_upload_splits_at_gpu_limit_without_resampling(native_skin, up):
    _, section, *_ = native_skin
    fr = _renderer(_atlas(native_skin), section, up=up)
    fr.rc.ctx.info = {"GL_MAX_TEXTURE_SIZE": 128}
    fr._direct_arr_cache = {}
    fr._draw_legacy_column_direct(
        "column/note_hold_body/0", 100, 10, 140, 300,
        frame_index=0, source_bottom=1 if up else 0, source_top=0 if up else 1,
        repeat_y=True,
    )
    draws = fr.direct_draws
    assert len(draws) > 1
    assert sum(args[4] for args, _ in draws) == pytest.approx(300)
    assert draws[0][0][2] == pytest.approx(10)
    for args, kw in draws:
        tex = fr._direct_texture_array(args[0], frame_index=0, source_region=kw["source_region"])
        assert tex.filter == (moderngl.LINEAR, moderngl.LINEAR)
    assert all(size[0] == 7 and size[1] <= 128 for size, _, _ in fr.rc.ctx.uploads)
    first_rows = fr.rc.ctx.uploads[0][2]
    assert len(first_rows) > 0


@pytest.mark.slow
@pytest.mark.parametrize("up", [False, True])
def test_gl_native_body_marker_survives_at_tail_and_note_cover(native_skin, up):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    path, _, *_ = native_skin
    (path / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 1\nNoteImage0L: body\n",
    )
    with HeadlessGl(width=64, height=768) as gl:
        fr = FrameRenderer(RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=64, height=768,
                                        key_count=1), skin_dir=path)
        fr.upside_down = up
        note = VisibleNote(column=0, is_hold=True, time_ms=0, y_fraction=1, head_y_fraction=1,
                           tail_y_fraction=0.5)
        scene = SimpleNamespace(t_ms=0, keys_held=(False,))
        gl.fbo.use()
        gl.ctx.enable(moderngl.BLEND)
        gl.fbo.clear(0, 0, 0, 1)
        fr._draw_legacy_hold_body(scene, note, x0=0, cw=64, body_y=100, body_height=300)
        pixels = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 64, 3)
        marker_y = 105 if up else 395
        assert tuple(pixels[marker_y, 32]) == (240, 0, 0)
        assert tuple(pixels[250, 32]) == (0, 200, 0)
        # Native direct notes participate in the existing HD cover.
        fr.apply_note_cover(True, False, 0)
        fr._cov_recep = 0
        gl.fbo.clear(0, 0, 0, 1)
        fr._draw_legacy_hold_body(scene, note, x0=0, cw=64, body_y=0, body_height=100)
        covered = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 64, 3)
        assert tuple(covered[50, 32]) == (0, 0, 0)
        assert fr.programs["sprite"]["u_hd"].value == 0  # HUD/key draws don't inherit it


@pytest.mark.parametrize("configured,expected_height", [(40, 64), (None, 100), (0, 100)])
@pytest.mark.parametrize("path", ["monolithic", "wiki"])
@pytest.mark.parametrize("hold", [False, True])
def test_width_setting_and_minimum_column_width_reach_both_draw_paths(
    native_skin, configured, expected_height, path, hold,
):
    _, section, *_ = native_skin
    fr = _renderer(_atlas(native_skin), section)
    fr.col_w = (140, 100)
    fr.mania_section = ManiaSection(**{
        **fr.mania_section.__dict__, "width_for_note_height_scale": configured,
    })
    note = VisibleNote(column=0, is_hold=hold, y_fraction=1, head_y_fraction=1,
                       tail_y_fraction=0.5)
    scene = SimpleNamespace(t_ms=0, keys_held=(False,), visible_notes=(note,))
    if path == "monolithic":
        fr._draw_notes(scene)
    else:
        _draw_notes_body(_wiki_context(fr, scene))
    assert fr.indexed_draws
    assert all(args[3] == 140 for args in fr.indexed_draws)  # column X size is preserved
    assert all(args[4] == expected_height for args in fr.indexed_draws)
    if hold:
        args, kw = fr.direct_draws[0]
        assert (1 - kw["source_bottom"]) * 2048 == pytest.approx(args[4])


def test_animated_native_body_draw_selects_30ms_source_frames(native_skin):
    path, section, _, body = native_skin
    body.save(path / "body-0.png")
    body.save(path / "body-1.png")
    fr = _renderer(_atlas(native_skin), section)
    note = VisibleNote(column=0, is_hold=True, time_ms=0, y_fraction=1,
                       head_y_fraction=1, tail_y_fraction=0.5)
    frames = []
    for t_ms, held in [(0, True), (29, True), (30, True), (60, True), (65, False)]:
        fr._draw_legacy_hold_body(
            SimpleNamespace(t_ms=t_ms, keys_held=(held,)), note,
            x0=100, cw=140, body_y=100, body_height=300,
        )
        frames.append(fr.direct_draws[-1][1]["frame_index"])
    assert frames == [0, 0, 1, 0, 0]


@pytest.mark.slow
@pytest.mark.parametrize("up", [False, True])
def test_gl_split_and_full_native_sources_sample_the_same_pixels(native_skin, up):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    path, section, *_ = native_skin
    atlas = _atlas(native_skin)
    with HeadlessGl(width=64, height=768) as gl:
        fr = FrameRenderer(RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=64, height=768,
                                        key_count=1), skin_dir=path)
        fr.atlas = atlas
        kwargs = dict(frame_index=0, source_bottom=1 if up else 0,
                      source_top=0 if up else 1, repeat_y=True)
        gl.fbo.use()
        gl.ctx.enable(moderngl.BLEND)
        gl.fbo.clear(0, 0, 0, 1)
        fr._draw_legacy_column_direct("column/note_hold_body/0", 0, 50, 64, 640, **kwargs)
        texture = fr._direct_arr_cache[("column/note_hold_body/0", 0)]
        assert texture.repeat_x is True and texture.repeat_y is True
        full = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).copy()
        # Force the same strip fallback used by a GPU with a smaller native limit.
        previous = gl.ctx.info["GL_MAX_TEXTURE_SIZE"]
        gl.ctx.info["GL_MAX_TEXTURE_SIZE"] = 128
        try:
            gl.fbo.clear(0, 0, 0, 1)
            fr._draw_legacy_column_direct("column/note_hold_body/0", 0, 50, 64, 640, **kwargs)
            strips = [texture for key, texture in fr._direct_arr_cache.items()
                      if isinstance(key, tuple) and len(key) == 3]
            assert strips and all(texture.repeat_x and not texture.repeat_y for texture in strips)
            split = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).copy()
        finally:
            gl.ctx.info["GL_MAX_TEXTURE_SIZE"] = previous
        assert np.max(np.abs(full.astype(int) - split.astype(int))) <= 1


@pytest.mark.parametrize("reverse", [False, True])
def test_consumption_crop_keeps_original_source_slope(reverse):
    source = (0.9, 0.2) if reverse else (0.2, 0.9)
    original = LegacyHoldBodySegment(10, 100, *source)
    clipped = legacy_clip_y_segment(original, minimum_y=40, maximum_y=90)
    assert clipped.y == 40 and clipped.height == 50
    assert clipped.source_bottom == pytest.approx(source[0] + (source[1] - source[0]) * 0.3)
    assert clipped.source_top == pytest.approx(source[0] + (source[1] - source[0]) * 0.8)
    assert legacy_clip_y_segment(original, minimum_y=110) is None
    assert legacy_clip_y_segment(original, maximum_y=10) is None


@pytest.mark.parametrize("raw", [0, None, 2, 4])
@pytest.mark.parametrize("up", [False, True])
def test_both_paths_mask_original_body_and_tail_but_leave_head_unmasked(native_skin, raw, up):
    _, section, *_ = native_skin
    outputs = []
    note = VisibleNote(
        column=0, is_hold=True, y_fraction=1, head_y_fraction=1,
        tail_y_fraction=0.7, time_ms=1000,
        hold_head_hit=True, hold_active=True,
        body_head_y_fraction=1.4, hold_clip_head_y_fraction=1,
    )
    for path in ("monolithic", "wiki"):
        fr = _renderer(_atlas(native_skin), section, up=up, raw=raw)
        scene = SimpleNamespace(t_ms=1240, visible_notes=(note,), keys_held=(True,))
        if path == "monolithic":
            fr._draw_notes(scene)
        else:
            _draw_notes_body(_wiki_context(fr, scene))
        boundary = 598 if up else 170  # receptor +/- half the 140px head
        args, kw = fr.direct_draws[0]
        assert args[2] + args[4] == pytest.approx(boundary if up else 230)
        assert args[2] == pytest.approx(537 if up else boundary)
        full_y, full_height = (537, 328) if up else (-97, 327)
        style = legacy_note_body_style(fr.mania_section, 0, 2.7)
        original = legacy_hold_body_segments(full_y, full_height, 2048, style, upside_down=up)[0]
        expected = legacy_clip_y_segment(
            original, minimum_y=None if up else boundary, maximum_y=boundary if up else None,
        )
        assert kw["source_bottom"] == pytest.approx(expected.source_bottom)
        assert kw["source_top"] == pytest.approx(expected.source_top)
        # Source slope is cropped, including Stretch; it is never resized to [0, 1].
        assert abs(kw["source_top"] - kw["source_bottom"]) < 1
        head = fr.indexed_draws[0]
        assert head[2:5] == (528 if up else 100, 140, 140)
        assert len(fr.indexed_draws) == 1  # tail goes through the crop primitive
        tail, tail_kw = fr.cropped_draws[0]
        assert tail[2] >= boundary if not up else tail[2] + tail[4] <= boundary
        assert (tail_kw["source_bottom"], tail_kw["source_top"]) != (0, 1)
        outputs.append((fr.direct_draws, fr.indexed_draws, fr.cropped_draws))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("up", [False, True])
def test_missed_and_dropped_holds_use_factual_mask_position_not_key_state(native_skin, up):
    _, section, *_ = native_skin
    for clip_fraction in (None, 1.2):
        note = VisibleNote(
            column=0, is_hold=True, y_fraction=1.4 if clip_fraction is None else 1.2,
            head_y_fraction=1.4 if clip_fraction is None else 1.2,
            tail_y_fraction=0.7, time_ms=1000, hold_head_hit=clip_fraction is not None,
            hold_active=False, body_head_y_fraction=1.4,
            hold_clip_head_y_fraction=clip_fraction,
        )
        outputs = []
        for path in ("monolithic", "wiki"):
            fr = _renderer(_atlas(native_skin), section, up=up)
            scene = SimpleNamespace(t_ms=1240, visible_notes=(note,), keys_held=(True,))
            if path == "monolithic":
                fr._draw_notes(scene)
            else:
                _draw_notes_body(_wiki_context(fr, scene))
            body, _kw = fr.direct_draws[0]
            if clip_fraction is None:
                assert body[4] == (328 if up else 327)  # missed full body passes
                assert not fr.cropped_draws
            else:
                assert body[4] > 60  # moving drop boundary reveals more than a frozen mask
                assert body[2] < 170 if not up else body[2] + body[4] > 598
            outputs.append((fr.direct_draws, fr.indexed_draws, fr.cropped_draws))
        assert outputs[0] == outputs[1]


@pytest.mark.slow
@pytest.mark.parametrize("raw", [0, 3, 2])
@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("force_strips", [False, True])
def test_gl_consumption_crops_original_pixels_as_marker_crosses_boundary(native_skin, raw, up, force_strips):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    path, *_ = native_skin
    (path / "skin.ini").write_text(
        f"[General]\nVersion: 2.7\n[Mania]\nKeys: 1\nNoteBodyStyle: {raw}\nNoteImage0L: body\n",
    )
    with HeadlessGl(width=64, height=768) as gl:
        fr = FrameRenderer(RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=64, height=768,
                                        key_count=1), skin_dir=path)
        fr.upside_down = up
        if force_strips:
            gl.ctx.info["GL_MAX_TEXTURE_SIZE"] = 128
        note = VisibleNote(column=0, is_hold=True, time_ms=0,
                           y_fraction=1, head_y_fraction=1, tail_y_fraction=0.5)
        scene = SimpleNamespace(t_ms=0, keys_held=(True,))
        gl.fbo.use()
        gl.ctx.enable(moderngl.BLEND)
        marker_counts = []
        for down_y in (150, 110, 50):
            body_y = 768 - down_y - 300 if up else down_y
            gl.fbo.clear(0, 0, 0, 1)
            fr._draw_legacy_hold_body(scene, note, x0=0, cw=64, body_y=body_y, body_height=300)
            reference = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 64, 3)
            gl.fbo.clear(0, 0, 0, 1)
            fr._draw_legacy_hold_body(
                scene, note, x0=0, cw=64, body_y=body_y, body_height=300,
                clip_min_y=None if up else 350, clip_max_y=418 if up else None,
            )
            clipped = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 64, 3)
            visible = slice(0, 418) if up else slice(350, 768)
            consumed = slice(418, 768) if up else slice(0, 350)
            assert np.max(np.abs(reference[visible].astype(int) - clipped[visible].astype(int))) <= 1
            assert not np.any(clipped[consumed])
            marker_counts.append(np.count_nonzero(np.all(clipped[:, 32] == (240, 0, 0), axis=1)))
        if raw in (0, 3):
            assert marker_counts[0] >= marker_counts[1] > marker_counts[2] == 0


@pytest.mark.slow
@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("opaque_tail", [False, True])
def test_gl_tail_uses_same_mask_while_head_and_transparent_tail_are_preserved(native_skin, up, opaque_tail):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    path, *_ = native_skin
    if opaque_tail:
        tail = Image.new("RGBA", (20, 20), (0, 0, 240, 255))
        tail.paste((240, 0, 0, 255), (0, 0, 20, 10))
        tail.save(path / "transparent.png")
    (path / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 1\nNoteImage0: cap\n"
        "NoteImage0H: cap\nNoteImage0L: body\nNoteImage0T: transparent\n",
    )
    with HeadlessGl(width=64, height=768) as gl:
        fr = FrameRenderer(RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=64, height=768,
                                        key_count=1), skin_dir=path)
        fr.upside_down = up
        fr.receptor_centre_y_gl = 668 if up else 100
        note = VisibleNote(column=0, is_hold=True, time_ms=0,
                           y_fraction=1, head_y_fraction=1, tail_y_fraction=0.9,
                           hold_head_hit=True, hold_active=True,
                           body_head_y_fraction=1, hold_clip_head_y_fraction=1)
        scene = SimpleNamespace(t_ms=0, keys_held=(True,))
        gl.fbo.use()
        gl.ctx.enable(moderngl.BLEND)
        gl.fbo.clear(0, 0, 0, 1)
        fr._draw_legacy_hold_note(
            scene, note, x0=0, cw=64, y_head=668 if up else 100,
            y_tail=638 if up else 130, head_h=40, tail_h=80,
            head_idx=fr.atlas.column_slot_index("note_hold_head", 0),
            tail_idx=fr.atlas.column_slot_index("note_hold_tail", 0),
        )
        fr._flush_sprite_batch()
        pixels = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 64, 3)
        row = lambda down_row: 767 - down_row if up else down_row
        assert tuple(pixels[row(80), 32]) == (0, 0, 0)  # clipped tail's far side
        assert tuple(pixels[row(110), 32]) == (255, 255, 255)  # unmasked head
        # The opaque cropped tail overlaps the head here. Stable's head
        # depth is higher than rear depth, so both cases retain white.
        assert tuple(pixels[row(125), 32]) == (255, 255, 255)


@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("path", ["monolithic", "wiki"])
@pytest.mark.parametrize("state", ["approaching", "held", "dropped", "missed"])
def test_shared_hold_part_order_in_both_paths(native_skin, up, path, state):
    _, section, *_ = native_skin
    fr = _renderer(_atlas(native_skin), section, up=up, raw=0)
    calls = []
    fr._draw_legacy_hold_body = lambda *a, **kw: calls.append(("body", kw))
    fr._draw_sprite_idx = lambda idx, *a: calls.append((idx, a))
    fr._draw_sprite_idx_cropped_y = lambda idx, *a, **kw: calls.append((idx, kw))
    clipped = state in ("held", "dropped")
    head_fraction = 1.05 if state == "dropped" else 1
    note = VisibleNote(
        column=0, is_hold=True, y_fraction=head_fraction,
        head_y_fraction=head_fraction, tail_y_fraction=0.8, time_ms=1000,
        hold_head_hit=clipped, hold_active=state == "held",
        body_head_y_fraction=1.2 if clipped else None,
        hold_clip_head_y_fraction=head_fraction if clipped else None,
    )
    scene = SimpleNamespace(t_ms=1240, visible_notes=(note,), keys_held=(clipped,))
    if path == "monolithic":
        fr._draw_notes(scene)
    else:
        _draw_notes_body(_wiki_context(fr, scene))
    assert [call[0] for call in calls] == [
        "body", fr.atlas.column_slot_index("note_hold_tail", 0),
        fr.atlas.column_slot_index("note_hold_head", 0),
    ]
    assert calls[0][1]["body_height"] >= 0
    tail = calls[1][1]
    if clipped:
        assert isinstance(tail, dict)  # cropped tail, still before full head
        assert 0 < tail["source_top"] - tail["source_bottom"] < 1
    else:
        assert not isinstance(tail, dict)  # complete tail path
    assert not isinstance(calls[-1][1], dict)  # head always unmasked


@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("gap", [150, 130, 101, 100])
def test_short_hold_overlap_order_and_nonnegative_body(native_skin, up, masked, gap):
    _, section, *_ = native_skin
    fr = _renderer(_atlas(native_skin), section, up=up, raw=0)
    fr.receptor_centre_y_gl = 468 if up else 300
    calls = []
    fr._draw_legacy_hold_body = lambda *a, **kw: calls.append(("body", kw))
    fr._draw_sprite_idx = lambda idx, *a: calls.append((idx, a))
    fr._draw_sprite_idx_cropped_y = lambda idx, *a, **kw: calls.append((idx, kw))
    note = VisibleNote(
        column=0, is_hold=True, time_ms=0,
        y_fraction=1, head_y_fraction=1, tail_y_fraction=0.8,
        hold_clip_head_y_fraction=1 if masked else None,
    )
    fr._draw_legacy_hold_note(
        SimpleNamespace(t_ms=0), note, x0=0, cw=64,
        y_head=468 if up else 300, y_tail=468 - gap if up else 300 + gap,
        head_h=100, tail_h=100, head_idx=1, tail_idx=2,
    )
    assert [call[0] for call in calls] == ["body", 2, 1]
    assert calls[0][1]["body_height"] == pytest.approx(gap - 100)
    assert calls[-1][1][1:4] == (368 if up else 300, 64, 100)


@pytest.mark.slow
@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("gap", [130, 101, 100])
def test_gl_overlapping_tail_stays_behind_head_with_original_crop(tmp_path, up, masked, gap):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    # Transparent right edges expose body-only and tail-only pixels even
    # where the large cap rectangles almost completely overlap.
    head = Image.new("RGBA", (64, 100))
    head.paste((0, 255, 0, 255), (0, 0, 32, 100))
    head.save(tmp_path / "green.png")
    tail = Image.new("RGBA", (64, 100))
    tail.paste((255, 0, 0, 255), (0, 0, 48, 100))
    # Two asymmetrical stripes make a full-source squash observable.
    tail.paste((255, 255, 0, 255), (0, 10, 48, 20))
    tail.paste((255, 0, 255, 255), (0, 70, 48, 80))
    tail.save(tmp_path / "red.png")
    Image.new("RGBA", (7, 2048), (0, 0, 255, 255)).save(tmp_path / "blue.png")
    (tmp_path / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 1\nNoteBodyStyle: 0\n"
        "NoteImage0H: green\nNoteImage0T: red\nNoteImage0L: blue\n",
    )
    with HeadlessGl(width=64, height=768) as gl:
        fr = FrameRenderer(RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=64, height=768,
                                        key_count=1), skin_dir=tmp_path)
        fr.upside_down = up
        fr.receptor_centre_y_gl = 468 if up else 300
        gl.fbo.use()
        gl.ctx.enable(moderngl.BLEND)

        def draw(clip):
            note = VisibleNote(
                column=0, is_hold=True, time_ms=0,
                y_fraction=1, head_y_fraction=1, tail_y_fraction=0.8,
                hold_head_hit=clip, hold_active=clip,
                hold_clip_head_y_fraction=1 if clip else None,
            )
            gl.fbo.clear(0, 0, 0, 1)
            fr._draw_legacy_hold_note(
                SimpleNamespace(t_ms=0, keys_held=(clip,)), note, x0=0, cw=64,
                y_head=468 if up else 300, y_tail=468 - gap if up else 300 + gap,
                head_h=100, tail_h=100,
                head_idx=fr.atlas.column_slot_index("note_hold_head", 0),
                tail_idx=fr.atlas.column_slot_index("note_hold_tail", 0),
            )
            fr._flush_sprite_batch()
            pixels = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 64, 3)
            return pixels[::-1] if up else pixels

        reference = draw(False)
        pixels = draw(masked)
        # Green head covers the opaque tail, including its coloured bands.
        assert tuple(pixels[365, 16]) == (0, 255, 0)
        assert tuple(pixels[315, 16]) == (0, 255, 0)  # head extends outside mask
        assert tuple(pixels[395, 40]) == (255, 0, 0)  # only tail art here
        if gap == 130:
            assert tuple(pixels[425, 16]) == (255, 0, 0)  # outside head rectangle
            assert tuple(pixels[365, 56]) == (0, 0, 255)  # body-only artwork
        elif gap == 101:
            assert tuple(pixels[350, 56]) == (0, 0, 255)  # native one-pixel body
            assert np.count_nonzero(np.any(pixels[:, 56], axis=1)) == 1
        else:
            assert not np.any(pixels[:, 56])  # zero body leaves no stray quad
        if masked:
            # Only the tail occupies this column below the boundary. The
            # consumed part disappears while every retained source row is
            # identical to the complete-tail render, including both bands.
            assert tuple(reference[340, 40]) != (0, 0, 0)
            assert tuple(pixels[340, 40]) == (0, 0, 0)
            assert np.max(np.abs(pixels[350:].astype(int) - reference[350:].astype(int))) <= 1
