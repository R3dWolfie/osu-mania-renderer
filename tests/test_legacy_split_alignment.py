"""Independent stable geometry oracle, all-key fixtures and actual consumers."""

from __future__ import annotations

import hashlib
import os
from math import ceil
from types import SimpleNamespace

import moderngl
import numpy as np
import pytest
from PIL import Image, ImageDraw

from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.legacy_stage_geometry import (
    legacy_stage_layout,
)
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import SceneState, VisibleNote
from osu_mania_renderer_v2.wiki_elements import effects, notes, stage
from osu_mania_renderer_v2.wiki_elements.context import FrameContext
from scripts.generate_synthetic_keymatrix import beatmap_bytes, generate, objects, replay_bytes


def stable_oracle(
    keys,
    *,
    width=1920,
    height=1080,
    start=136,
    reserve=19,
    widths=None,
    spacings=None,
    separation=40,
):
    """Transcribe StageMania 194..220 and ColumnMania 127..131 independently.

    Accumulate each stage in 480-space, then convert to render pixels. Do
    not obtain expected values from the production layout being tested.
    """
    widths = widths or [30.0] * keys
    spacings = spacings or [0.0] * (keys - 1)
    screen = ceil(width / (height / 480))
    left, right = min(start, screen - reserve), min(reserve, screen - start)
    resize = sum(w + (spacings[c - 1] if c else 0) for c, w in enumerate(widths)) + separation
    fit = (resize - max(0, left + resize + right - screen)) / resize
    result, stages = [], []
    offset, origin = 0, left
    for count in ((keys + 1) // 2, keys // 2):
        stage_width = 0.0
        for c in range(offset, offset + count):
            spacing = spacings[c - 1] * fit if c else 0
            x = origin + stage_width + spacing
            w = widths[c] * fit
            result.append((x * height / 480, w * height / 480))
            stage_width += w + spacing
        stages.append((origin * height / 480, stage_width * height / 480))
        origin += stage_width + separation * fit
        offset += count
    return result, stages, fit


def renderer_geometry(keys, section=None):
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(width=1920, height=1080, key_count=keys)
    fr.mania_section = section
    fr._is_argon_default = lambda: False
    fr._compute_playfield_geometry()
    return fr


@pytest.mark.parametrize("keys", range(1, 19))
def test_all_stable_keycounts_mapping_and_renderer_geometry(keys):
    fr = renderer_geometry(keys)
    stages = fr.stage_layout.stages
    assert sum(s.column_count for s in stages) == keys
    assert len(stages) == (2 if keys >= 10 else 1)
    for c in range(keys):
        s, local = fr.stage_layout.locate(c)
        assert s.first_column + local == c
        assert (fr.stage_layout.column_kind(c) == "S") == (local == s.special_column)
        assert fr.col_w[c] > 0
        if c:
            assert fr.col_x[c] >= fr.col_x[c - 1] + fr.col_w[c - 1]
    if keys < 10:
        # Preserve the accepted historical 42-ref default and centring.
        assert fr.col_w == (67.5,) * keys
        assert fr.col_x == tuple(136 * 2.25 + c * 67.5 for c in range(keys))
    else:
        expected, expected_stages, fit = stable_oracle(keys)
        assert tuple(s.column_count for s in stages) == ((keys + 1) // 2, keys // 2)
        assert tuple(zip(fr.col_x, fr.col_w)) == pytest.approx(expected)
        assert tuple((s.x, s.width) for s in stages) == pytest.approx(expected_stages)
        assert fr.stage_layout.width_scale == pytest.approx(fit)
        assert fr.stage_layout.gap_geometry()["visible_column_gap"] == 90
        assert len(fr.stage_layout.geometry_rows()) == keys


@pytest.mark.parametrize(
    "keys,split",
    [(4, True), (7, True), (10, False), (10, True), (11, False), (12, False), (18, False)],
)
def test_authored_split_override_uses_expected_path(keys, split):
    fr = renderer_geometry(keys, ManiaSection(keys, split_stages=split))
    assert len(fr.stage_layout.stages) == (2 if split else 1)
    if split:
        expected, _, _ = stable_oracle(keys)
        assert tuple(zip(fr.col_x, fr.col_w)) == pytest.approx(expected)
    else:
        assert fr.col_w == (67.5,) * keys


@pytest.mark.parametrize("keys", [10, 11, 12, 13, 18])
def test_source_derived_fractional_fit_and_boundary_inset(keys):
    widths = [60.25 + c for c in range(keys)]
    spacing = [1.5 + c / 4 for c in range(keys - 1)]
    section = ManiaSection(
        keys,
        split_stages=True,  # 10K needs an authored override for split geometry.
        column_start=136.5,
        column_right=19.5,
        column_width=tuple(widths),
        column_spacing=tuple(spacing),
        stage_separation=40.5,
    )
    layout = legacy_stage_layout(keys, section, render_width=1920, render_height=1080)
    expected, stages, fit = stable_oracle(
        keys, start=136.5, reserve=19.5, widths=widths, spacings=spacing, separation=40.5
    )
    for c, (x, w) in enumerate(expected):
        assert layout.column_x[c] == pytest.approx(x)
        assert layout.column_width[c] == pytest.approx(w)
        assert layout.column_center(c) == pytest.approx(x + w / 2)
    np.testing.assert_allclose([(s.x, s.width) for s in layout.stages], stages, rtol=1e-12)
    gap = layout.gap_geometry()
    boundary = layout.stages[1].first_column
    assert gap["boundary_column_spacing"] == pytest.approx(spacing[boundary - 1] * fit * 2.25)
    assert gap["stage_separation"] == pytest.approx(40.5 * fit * 2.25)
    assert gap["visible_column_gap"] == pytest.approx((40.5 + spacing[boundary - 1]) * fit * 2.25)


@pytest.mark.parametrize("keys", [10, 11, 12, 18])
@pytest.mark.parametrize("slot", ["lighting_n", "lighting_l"])
def test_fractional_split_lights_receptors_and_notes_match_source_centres(keys, slot):
    fr = renderer_geometry(keys, ManiaSection(keys, split_stages=True))
    fr.skin_ini = SimpleNamespace(legacy_version=2.7)
    fr.atlas = SimpleNamespace(global_native_size=lambda _: (60, 100))
    expected, _, _ = stable_oracle(keys)
    boundary = (keys + 1) // 2
    for c in (0, boundary - 1, boundary, keys - 1):
        x, w = expected[c]
        assert (fr.col_x[c], fr.col_w[c]) == (x, w)
        rectangle = fr._legacy_lighting_rect(
            slot, c=c, x0=fr.col_x[c], cw=fr.col_w[c], centre_y=100
        )
        # Before this follow-up the floor division misses this centre by
        # .75px at 1080p default split widths, despite key/note agreement.
        assert rectangle[0] + rectangle[2] / 2 == pytest.approx(x + w / 2)
        assert rectangle[2:] == (84, 141)  # accepted native size is unchanged


def test_generator_is_deterministic_and_raw_replays_really_hit(tmp_path):
    manifest = generate(tmp_path)
    assert generate(tmp_path) == manifest
    assert [f["keys"] for f in manifest["fixtures"]] == list(range(1, 19))
    for fixture in manifest["fixtures"]:
        keys = fixture["keys"]
        assert fixture["raw_misses"] == 0
        assert fixture["raw_MAX_events"] == fixture["objects"] + keys
        assert len({c for c, p, r in objects(keys) if r - p > 40}) == keys
        data = beatmap_bytes(keys)
        assert data == beatmap_bytes(keys)
        md5 = hashlib.md5(data).hexdigest()
        assert replay_bytes(keys, md5) == replay_bytes(keys, md5)


@pytest.mark.parametrize("keys", range(1, 19))
@pytest.mark.parametrize("wiki", [False, True])
def test_every_column_consumer_uses_source_geometry_without_receptor_rewrites(keys, wiki):
    fr = renderer_geometry(keys, ManiaSection(keys))
    fr.options = SimpleNamespace(show_key_overlay=True)
    fr.skin_ini = SimpleNamespace(legacy_version=2.7)
    fr.atlas = SimpleNamespace(
        column_native_size=lambda *args: (60, 180),
        global_native_size=lambda *args: (60, 100),
        global_source=lambda *args: "user",
        global_aspect=lambda *args: 1.0,
        has_skin_notes=lambda: True,
        has_skin_note=lambda c: True,
        has_skin_hold=lambda c: True,
        column_aspect=lambda *args: 2.5,
        column_frame_count=lambda *args: 1,
        column_slot_index=lambda kind, c: c,
        frame_count=lambda *args: 1,
        index_of=lambda *args: 1000,
    )
    indexed, receptors, holds = [], [], []
    fr._draw_sprite_idx = lambda *args: indexed.append(args)
    fr._draw_sprite_idx_cropped_y = lambda *args, **kw: indexed.append(args)
    fr._draw_legacy_column_direct = lambda *args, **kw: receptors.append(args)
    fr._draw_legacy_hold_note = lambda scene, note, **kw: holds.append((note.column, kw))
    fr._stage_light_fps = lambda count: 60
    fr._stage_light_tint = lambda c: (1, 1, 1, 1)
    visible = tuple(
        VisibleNote(c, hold, 0.5, 0.5, 0.25) for c in range(keys) for hold in (False, True)
    )
    scene = SceneState(1000, visible, (True,) * keys, VisualMods())
    if wiki:
        ctx = FrameContext(fr, None, None, None, 1920, 1080, keys, scene=scene)
        notes._receptors(ctx)
        notes._draw_notes_body(ctx)
        # Native light primitive is shared by both stage-manager compositors.
        ctx.fr._draw_stage_lights(ctx.scene)
    else:
        fr._draw_receptors(scene)
        fr._draw_notes(scene)
        fr._draw_stage_lights(scene)
    expected = stable_oracle(keys)[0] if keys > 10 else list(zip(fr.col_x, fr.col_w))
    assert len(receptors) == len(holds) == keys
    for c, (x, width) in enumerate(expected):
        key = receptors[c]
        assert (key[1], key[3]) == pytest.approx((x, width))
        assert (key[2], key[4]) == (0, 253)  # native 180 * 1080/768, not aspect-height
        assert holds[c][0] == c
        assert (holds[c][1]["x0"], holds[c][1]["cw"]) == pytest.approx((x, width))
        taps = [call for call in indexed if call[0] == c]
        assert len(taps) == 1  # HC has one sprite and no fabricated ghosts
        assert all((call[1], call[3]) == pytest.approx((x, width)) for call in taps)
        for slot in ("lighting_n", "lighting_l"):
            light = fr._legacy_lighting_rect(slot, c=c, x0=x, cw=width, centre_y=100)
            assert light[0] + light[2] / 2 == pytest.approx(x + width / 2)
    lights = [call for call in indexed if call[0] == 1000]
    assert len(lights) == keys
    for light, (x, width) in zip(lights, expected):
        assert (light[1], light[3]) == pytest.approx((x, width))
        assert light[4] == 100 * 1080 / 768  # native Y unchanged


@pytest.mark.parametrize("keys", [10, 11, 12, 18])
@pytest.mark.parametrize("wiki", [False, True])
def test_split_chrome_edges_are_source_stage_local(keys, wiki):
    # 10K still exercises the split formulas through the preserved override.
    section = ManiaSection(keys, split_stages=True) if keys == 10 else None
    fr = renderer_geometry(keys, section)
    fr.atlas = SimpleNamespace(
        global_source=lambda _: "user", global_native_size=lambda _: (4, 768)
    )
    direct, sprites = [], []
    fr._draw_direct = lambda *args, **kw: direct.append(args)
    fr._draw_sprite = lambda *args: sprites.append(args)
    if wiki:
        ctx = FrameContext(fr, None, None, None, 1920, 1080, keys)
        stage.stage_decorations(element=None, skin=None, assets=None, variables=None, ctx=ctx)
    else:
        fr._draw_stage_decorations()
    _, expected_stages, _ = stable_oracle(keys)
    assert len(direct) == 4
    for index, (x, width) in enumerate(expected_stages):
        left, right = direct[index * 2 : index * 2 + 2]
        assert left[0] == "stage_left" and right[0] == "stage_right"
        assert left[1] + left[3] == pytest.approx(x + 0.05 * 1080 / 480)
        assert right[1] == pytest.approx(x + width + 0.05 * 1080 / 480)
    hints = [call for call in sprites if call[0] == "hit_light"]
    # StageHint belongs to the stage manager, after columns.
    assert hints == []


@pytest.mark.slow
@pytest.mark.parametrize("keys", [10, 11, 12, 18])
@pytest.mark.parametrize("wiki", [False, True])
def test_real_gl_split_centres_match_stable_not_just_each_other(tmp_path, keys, wiki):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 for GL")
    # Preserve the existing 10K split-centre probe via explicit skin override;
    # 11K/12K/18K still exercise the automatic split path.
    override = "SplitStages:1\n" if keys == 10 else ""
    (tmp_path / "skin.ini").write_text(
        f"[General]\nVersion:2.7\n[Mania]\nKeys:{keys}\nHitPosition:402\n{override}"
    )
    for suffix in ("1", "2", "S"):
        for name, size, colour in (
            (f"mania-key{suffix}", (60, 180), (0, 255, 0, 255)),
            (f"mania-note{suffix}", (60, 24), (0, 0, 255, 255)),
        ):
            image = Image.new("RGBA", size)
            ImageDraw.Draw(image).rectangle((28, 0, 31, size[1] - 1), fill=colour)
            image.save(tmp_path / f"{name}.png")
    image = Image.new("RGBA", (60, 100))
    ImageDraw.Draw(image).rectangle((28, 0, 31, 99), fill=(255, 0, 0, 255))
    image.save(tmp_path / "lightingN.png")
    boundary = (keys + 1) // 2
    selected = (0, boundary - 1, boundary, keys - 1)
    scene = SceneState(
        1000,
        tuple(VisibleNote(c, False, 0.65, 0.65, 0.65) for c in selected),
        (False,) * keys,
        VisualMods(),
    )
    expected, _, _ = stable_oracle(keys)
    with HeadlessGl(width=1920, height=1080) as gl:
        fr = FrameRenderer(RenderContext(gl.ctx, gl.fbo, 1920, 1080, keys), skin_dir=tmp_path)
        gl.ctx.enable(moderngl.BLEND)
        gl.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        gl.fbo.use()
        gl.fbo.clear()
        if wiki:
            ctx = FrameContext(fr, None, gl.ctx, gl.fbo, 1920, 1080, keys, scene=scene)
            notes._receptors(ctx)
            notes._draw_notes_body(ctx)
        else:
            fr._draw_receptors(scene)
            fr._draw_notes(scene)
        # Centre the light at a clear diagnostic row, using the actual rect.
        for c in selected:
            rect = fr._legacy_lighting_rect(
                "lighting_n", c=c, x0=fr.col_x[c], cw=fr.col_w[c], centre_y=450
            )
            fr._draw_additive_sprite_idx(fr.atlas.index_of("lighting_n"), *rect, (1, 1, 1, 1))
        fr._flush_sprite_batch()
        pixels = np.frombuffer(gl.fbo.read(components=3), np.uint8).reshape(1080, 1920, 3)
        for c in selected:
            x, w = expected[c]
            center = x + w / 2
            # Raster pixel centres are x+.5: verify against independent
            # stable positions, not against the renderer's own col_x.
            for row, channel in ((100, 1), (450, 0)):
                strip = pixels[row, int(x) : ceil(x + w), channel].astype(float)
                positions = np.arange(int(x), ceil(x + w)) + 0.5
                assert (positions * strip).sum() / strip.sum() == pytest.approx(center, abs=0.1)
            note_y = int(fr.receptor_centre_y_gl + 0.35 * (1080 - fr.receptor_centre_y_gl)) + 4
            strip = pixels[note_y, int(x) : ceil(x + w), 2].astype(float)
            positions = np.arange(int(x), ceil(x + w)) + 0.5
            assert (positions * strip).sum() / strip.sum() == pytest.approx(center, abs=0.1)
