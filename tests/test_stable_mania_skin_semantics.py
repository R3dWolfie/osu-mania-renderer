"""Synthetic contracts transcribed from SkinMania, HC/LN, CM and HpBarMania.

No skin identities, user directories, or video comparisons are test inputs.
"""
from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import moderngl
import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection, parse_skin_ini
from osu_mania_renderer_v2.gpu import atlas as atlas_module
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.legacy_mania import (
    legacy_key_flip, legacy_note_flip, legacy_note_body_style,
    legacy_hold_body_segments,
)
from osu_mania_renderer_v2.gpu.legacy_stage_geometry import legacy_stage_layout
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import SceneState, VisibleNote
from osu_mania_renderer_v2.wiki_elements.context import FrameContext
from osu_mania_renderer_v2.wiki_elements.notes import _draw_notes_body, _receptors


def asymmetric(path, *, width=8, height=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (width, height), (0, 0, 255, 255))
    image.paste((255, 0, 0, 255), (0, 0, width, height // 2))
    image.save(path)
    return image


def resolve(directory, kind, section, *, beatmap=None, column=0):
    return SpriteAtlas._resolve_column(
        kind=kind, col=column, key_count=section.keys,
        skin_dir=directory, beatmap_dir=beatmap, section=section,
    )


@pytest.mark.parametrize("version", [1, 2.4, 2.5, 2.7])
@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("part", ["", "H", "L", "T"])
def test_note_flip_dictionary_precedence_and_gates(version, up, part):
    section = ManiaSection(
        keys=2, note_flip_when_upside_down={part: False},
        note_flip_by_column={(0, part): True},
    )
    for column, configured in [(0, True), (1, False)]:
        expected = configured if up and version >= 2.5 else False
        if part == "T":
            expected = not expected  # LN rear inversion, including old/down skins
        assert legacy_note_flip(section, column, part, upside_down=up,
                                legacy_version=version) is expected


def test_head_inherits_resolved_tap_but_body_tail_have_independent_defaults():
    section = ManiaSection(keys=1, note_flip_when_upside_down={"": False})
    assert not legacy_note_flip(section, 0, "H", upside_down=True, legacy_version=2.7)
    assert legacy_note_flip(section, 0, "L", upside_down=True, legacy_version=2.7)
    assert not legacy_note_flip(section, 0, "T", upside_down=True, legacy_version=2.7)
    section = replace(section, note_flip_when_upside_down={"": False, "H": True})
    assert legacy_note_flip(section, 0, "H", upside_down=True, legacy_version=2.7)


@pytest.mark.parametrize("up", [False, True])
def test_key_flip_has_column_precedence_and_no_note_version_gate(up):
    section = ManiaSection(keys=2, key_flip_when_upside_down=False,
                           key_flip_by_column={0: True})
    assert legacy_key_flip(section, 0, upside_down=up) is up
    assert not legacy_key_flip(section, 1, upside_down=up)
    assert legacy_key_flip(None, 0, upside_down=up) is up


def test_parser_retains_all_flip_fields_and_reference_flags(tmp_path):
    (tmp_path / "skin.ini").write_text(
        "[General]\nVersion: latest\n[Mania]\nKeys:2\nSpecialStyle:Left\n"
        "KeyFlipWhenUpsideDown:0\nKeyFlipWhenUpsideDown0:1\n"
        "JudgementLine:0\nColourJudgementLine:1,2,3,4\nColourKeyWarning:5,6,7,8\n"
        + "".join(f"NoteFlipWhenUpsideDown{p}:0\nNoteFlipWhenUpsideDown1{p}:1\n"
                  for p in ("", "H", "L", "T")),
    )
    skin = parse_skin_ini(tmp_path)
    section = skin.mania_for_keycount(2)
    assert skin.legacy_version == 2.7
    assert section.special_style == 1
    assert section.key_flip_when_upside_down is False
    assert section.key_flip_by_column == {0: True}
    assert section.note_flip_when_upside_down == {p: False for p in ("", "H", "L", "T")}
    assert section.note_flip_by_column == {(1,p): True for p in ("", "H", "L", "T")}
    assert section.judgement_line is False
    assert section.colour_judgement_line == (1,2,3,4)
    assert section.colour_key_warning == (5,6,7,8)
    assert skin.mania_for_keycount(7) is None


@pytest.mark.parametrize("part", ["H", "L", "T"])
def test_missing_explicit_part_reaches_conventional_before_other_parts(tmp_path, part):
    for suffix, colour in [("", (100,0,0,255)), ("H", (0,100,0,255)),
                          ("L", (0,0,100,255)), ("T", (100,100,0,255))]:
        Image.new("RGBA", (3,3), colour).save(tmp_path / f"mania-note1{suffix}.png")
    section = ManiaSection(keys=4, note_image={0:"mania-note1"},
                           **{f"note_image_{part.lower()}": {0:"missing"}})
    frames, source = resolve(tmp_path, {"H":"note_hold_head", "L":"note_hold_body",
                                        "T":"note_hold_tail"}[part], section)
    expected = {"H":(0,100,0,255), "L":(0,0,100,255), "T":(100,100,0,255)}[part]
    assert source == "user"
    assert frames[0].getpixel((0,0)) == expected


def test_t_missing_falls_back_to_resolved_animated_explicit_head(tmp_path):
    first = asymmetric(tmp_path / "head-0.png")
    Image.new("RGBA", (8,8), (10,200,10,255)).save(tmp_path / "head-1.png")
    asymmetric(tmp_path / "tap.png", width=16)
    section = ManiaSection(keys=4, note_image={0:"tap"}, note_image_h={0:"head"},
                           note_image_t={0:"missing"})
    heads, hs = resolve(tmp_path, "note_hold_head", section)
    tails, ts = resolve(tmp_path, "note_hold_tail", section)
    assert hs == ts == "user"
    assert len(heads) == len(tails) == 2
    assert tails[0].tobytes() == first.tobytes() == heads[0].tobytes()
    assert tails[1].tobytes() == heads[1].tobytes()


def test_default_h_l_are_same_name_classic_assets_not_authored_tap(tmp_path):
    asymmetric(tmp_path / "tap.png", width=13)
    section = ManiaSection(keys=4, note_image={0:"tap"},
                           note_image_h={0:"missing"}, note_image_l={0:"missing"})
    for kind in ("note_hold_head", "note_hold_body", "note_hold_tail"):
        frames, source = resolve(tmp_path, kind, section)
        assert source == "classic"
        assert frames[0].width != 13
    # NormalKey2 never searches authored mania-note1 as a replacement.
    asymmetric(tmp_path / "mania-note1.png", width=13)
    frames, source = resolve(tmp_path, "note_tap", section, column=1)
    assert source == "classic"
    assert frames[0].width != 13


def test_explicit_name_resolves_all_tiers_before_conventional(tmp_path):
    skin, beatmap = tmp_path / "skin", tmp_path / "map"
    asymmetric(skin / "named.png", width=7)
    asymmetric(beatmap / "named.png", width=9)
    asymmetric(beatmap / "mania-note1H.png", width=11)
    section = ManiaSection(keys=4, note_image_h={0:"named"})
    frames, src = resolve(skin, "note_hold_head", section, beatmap=beatmap)
    assert src == "beatmap" and frames[0].width == 9
    (beatmap / "named.png").unlink()
    frames, src = resolve(skin, "note_hold_head", section, beatmap=beatmap)
    assert src == "user" and frames[0].width == 7
    # A more-specific static beats a less-specific animation of the same name.
    asymmetric(skin / "named-0.png", width=17)
    asymmetric(beatmap / "named.png", width=19)
    frames, src = resolve(skin, "note_hold_head", section, beatmap=beatmap)
    assert src == "beatmap" and len(frames) == 1 and frames[0].width == 19


def test_body_has_no_h_fallback_and_tail_has_no_tap_array_fallback(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(atlas_module, "CLASSIC_MANIA_DIR", empty)
    asymmetric(tmp_path / "head.png")
    section = ManiaSection(keys=4, note_image_h={0:"head"})
    frames, src = resolve(tmp_path, "note_hold_body", section)
    assert src == "missing" and frames[0].getchannel("A").getbbox() is None
    assert resolve(tmp_path, "note_hold_tail", section)[1] == "user"


@pytest.mark.parametrize("raw", [0, 2, 3, 4, 1, 99, -1])
def test_body_style_keeps_enum_and_old_gate(raw):
    section = ManiaSection(keys=2, note_body_style=4, note_body_style_by_column={0:raw})
    assert legacy_note_body_style(section, 0, 2.7) == raw
    assert legacy_note_body_style(section, 1, 2.7) == 4
    assert legacy_note_body_style(section, 0, 2.4) == 0


@pytest.mark.parametrize("ratio", [0.5, 1, 1.5])
@pytest.mark.parametrize("style", [2,3,4,99])
def test_body_repeat_phase_uses_integer_stable_drawtop_at_height_ratio(ratio, style):
    # HM.length=distance*1.6; native DrawHeight20, native window h/768=ratio.
    length = 37.25
    draw_top = int(20 - length) if style == 2 else (
        int((20 - length) / 2) if style == 4 else 0)
    pieces = legacy_hold_body_segments(0, length * ratio, 20 * ratio, style,
                                       height_ratio=ratio, flip_vertical=False)
    first = pieces[0]
    expected = (-(draw_top + length)) % 20 / 20
    assert first.source_bottom == pytest.approx(expected)
    assert sum(p.height for p in pieces) == pytest.approx(length * ratio)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["conventional_t", "explicit_t", "missing_t_h",
                                  "missing_t_named_h", "animated_t", "animated_h"])
@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("version", [2.4, 2.7])
@pytest.mark.parametrize("setting", ["default", "base_false", "global_t_false", "column_t_false"])
@pytest.mark.parametrize("cropped", [False, True])
def test_tail_rendered_pixels_and_both_paths(tmp_path, case, up, version, setting, cropped):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 for real asymmetric GL pixels")
    head, tail = "head", "tail"
    target = {"conventional_t":"mania-note1T", "explicit_t":"tail",
              "missing_t_h":"mania-note1H", "missing_t_named_h":"head",
              "animated_t":"tail", "animated_h":"head"}[case]
    animated = case.startswith("animated")
    asymmetric(tmp_path / f"{target}{'-0' if animated else ''}.png")
    if animated:
        asymmetric(tmp_path / f"{target}-1.png")
    if case in ("conventional_t", "explicit_t", "animated_t"):
        asymmetric(tmp_path / "head.png")  # this fixture specifies 16px cap geometry
    Image.new("RGBA", (8,8), (0,255,0,255)).save(tmp_path / "body.png")
    fields = {"default":"", "base_false":"NoteFlipWhenUpsideDown:0\n",
              "global_t_false":"NoteFlipWhenUpsideDownT:0\n",
              "column_t_false":"NoteFlipWhenUpsideDownT:1\nNoteFlipWhenUpsideDown0T:0\n"}[setting]
    h_ref = "mania-note1H" if case == "missing_t_h" else head
    t_ref = tail if case in ("explicit_t", "animated_t") else "absent"
    if case == "conventional_t":
        t_ref = "mania-note1T"
    (tmp_path / "skin.ini").write_text(
        f"[General]\nVersion:{version}\n[Mania]\nKeys:4\nUpsideDown:{int(up)}\n"
        f"NoteImage0H:{h_ref}\nNoteImage0T:{t_ref}\nNoteImage0L:body\n{fields}")
    expected_flip = not (up and version >= 2.5 and setting not in
                         ("global_t_false", "column_t_false"))
    with HeadlessGl(width=128, height=128) as gl:
        fr = FrameRenderer(RenderContext(gl.ctx, gl.fbo,128,128,4), skin_dir=tmp_path)
        fr.col_x, fr.col_w = (0,32,64,96), (16,16,16,16)
        fr.receptor_centre_y_gl = 96 if up else 32
        fr._legacy_note_height = lambda aspect: 16
        # Place rear away from head. Both paths render the complete composition.
        note = VisibleNote(0, True, 1, 1, 0.5, time_ms=0,
                           hold_clip_head_y_fraction=0.5 if cropped else None)
        scene = SceneState(0,(note,),(False,)*4,VisualMods())
        ctx = FrameContext(fr=fr,skin=None,gl=gl.ctx,fbo=gl.fbo,width=128,height=128,key_count=4)
        ctx.scene = scene
        results = []
        for wiki in (False, True):
            gl.fbo.use(); gl.fbo.clear(0,0,0,1)
            gl.ctx.enable(moderngl.BLEND)
            gl.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            (_draw_notes_body(ctx) if wiki else fr._draw_notes(scene))
            fr._flush_sprite_batch()
            # Down tail anchor80; up tail anchor48 and draw32..48. Height16.
            tail_bottom = 32 if up else 80
            top = tuple(gl.fbo.read(viewport=(8,tail_bottom+12,1,1),components=3))
            bottom = tuple(gl.fbo.read(viewport=(8,tail_bottom+3,1,1),components=3))
            if not cropped or not up:
                assert top[2 if expected_flip else 0] > 240
            else:
                assert top == (0,0,0)
            if not cropped or up:
                assert bottom[0 if expected_flip else 2] > 240
            else:
                assert bottom == (0,0,0)
            results.append((top,bottom))
        assert results[0] == results[1]


@pytest.mark.slow
@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("pressed", [False, True])
def test_key_source_pixels_respect_flip_even_for_old_skins(tmp_path, up, configured, pressed):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 for GL key pixels")
    for name in ("key", "pressed"):
        asymmetric(tmp_path / f"{name}.png", height=384)
    (tmp_path / "skin.ini").write_text(
        "[General]\nVersion:1\n[Mania]\nKeys:1\nKeyImage0:key\nKeyImage0D:pressed\n"
        f"UpsideDown:{int(up)}\nKeyFlipWhenUpsideDown0:{int(configured)}\n")
    with HeadlessGl(width=128,height=128) as gl:
        fr = FrameRenderer(RenderContext(gl.ctx,gl.fbo,128,128,1),skin_dir=tmp_path)
        fr.col_x,fr.col_w = (0,),(32,)
        scene = SceneState(0,(),(pressed,),VisualMods())
        ctx = FrameContext(fr=fr,skin=None,gl=gl.ctx,fbo=gl.fbo,width=128,height=128,key_count=1)
        ctx.scene=scene
        observed=[]
        for wiki in (False,True):
            gl.fbo.use();gl.fbo.clear(0,0,0,1)
            (_receptors(ctx) if wiki else fr._draw_receptors(scene))
            fr._flush_sprite_batch()
            y=64 if up else 0
            top=tuple(gl.fbo.read(viewport=(16,y+48,1,1),components=3))
            assert top[2 if up and configured else 0] > 240
            observed.append(top)
        assert observed[0] == observed[1]


def test_single_stage_default_and_clamps_are_not_broadcast():
    layout = legacy_stage_layout(4,ManiaSection(4,column_width=(200,2),column_spacing=(-200,)),
                                 render_width=640,render_height=480)
    assert layout.column_width == (100,5,30,30)
    assert layout.column_x == (136,231,236,266)


def test_nonmatching_mania_block_keeps_argon_without_current_mania_material():
    from osu_mania_renderer_v2.beatmap.skin_ini import SkinIni
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(key_count=4, width=640, height=480, replay_mods=0)
    fr.skin_ini = SkinIni(mania=(ManiaSection(7),), legacy_version=2.7)
    fr.mania_section = None
    fr.atlas = SimpleNamespace(column_source=lambda *a: "classic",
                               global_source=lambda *a: "classic")
    assert fr._is_argon_default()


def test_mania_colours_cast_int32_to_bytes_and_retain_break_alpha(tmp_path):
    (tmp_path / "skin.ini").write_text(
        "[Mania]\nKeys:1\nColour1:-1,256,257,128\nColourBreak:1,2,3,4\n"
        "ColourLight1:1,2,3,4,5\n")
    section = parse_skin_ini(tmp_path).mania_for_keycount(1)
    assert section.colour[1] == (255,0,1,128)
    assert section.colour_break == (1,2,3,4)
    assert not section.colour_light


@pytest.mark.parametrize("wiki", [False, True])
def test_shared_split_painter_finishes_each_stage_before_the_next(wiki):
    from osu_mania_renderer_v2.wiki_elements.notes import notes as wiki_notes
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(key_count=12)
    fr.mania_section = ManiaSection(12)
    fr.stage_layout = legacy_stage_layout(12,fr.mania_section,render_width=640,render_height=480)
    fr._is_argon_default = lambda: False
    calls = []
    fr._flush_sprite_batch = lambda: None
    fr.apply_note_cover = lambda *args: None
    for method, name in (("_draw_notes","notes"),("_draw_receptors","keys"),
                         ("_draw_legacy_stage_foreground","bottom"),
                         ("_draw_legacy_hit_lighting","lighting"),
                         ("_draw_combo_and_judgment","judgment"),
                         ("_draw_custom_legacy_combo","combo")):
        setattr(fr,method,lambda *a,_name=name,**kw: calls.append((_name,fr._legacy_active_stage.index)))
    scene = SceneState(0,(),(False,)*12,VisualMods())
    if wiki:
        ctx = FrameContext(fr,None,None,None,640,480,12,scene=scene)
        wiki_notes(element=None,skin=None,assets=None,variables=None,ctx=ctx)
    else:
        fr._draw_legacy_stage_gameplay(scene)
    assert calls == [(name,i) for i in (0,1)
                     for name in ("notes","combo","keys","bottom","lighting","judgment")]
    assert fr._legacy_active_stage is None


def test_body_pause_and_hold_lighting_use_replay_hold_facts_after_repress():
    from osu_mania_renderer_v2.render.scene import HoldVisualState, snapshot
    from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent
    states = {(0,1000):HoldVisualState(1000,drop_time_ms=1065)}
    scene = snapshot((HoldNote(0,1000,2000),),
                     (KeyEvent(1000,1),KeyEvent(1065,0),KeyEvent(1100,1)),
                     1125,1,600,VisualMods(),hold_visual_states=states)
    hold = scene.visible_notes[0]
    assert scene.keys_held == (True,)
    assert not hold.hold_active and hold.hold_animation_elapsed_ms == 65
    assert scene.hold_light_press_age_ms == (125,)
    assert scene.hold_light_release_age_ms == (60,)
    fr = object.__new__(FrameRenderer)
    assert fr._legacy_hold_body_frame_index(scene,hold,3) == 2


def test_hold_lighting_fades_on_release_and_empty_held_lane_has_none():
    # Hold facts carry their own fade; a keypress without a hold is insufficient.
    fr = object.__new__(FrameRenderer)
    fr._is_argon_default = lambda: False
    fr.rc = SimpleNamespace(key_count=1)
    fr.col_x, fr.col_w = (0,), (30,)
    fr.receptor_centre_y_gl = 80
    fr._legacy_lighting_rect = lambda *a, **kw: (0,0,30,30)
    fr.atlas = SimpleNamespace(global_source=lambda _: "user",index_of=lambda _: 0,
                               frame_count=lambda _: 1)
    fr.additive_draws = []
    fr._draw_additive_sprite_idx = lambda *a: fr.additive_draws.append(a)
    scene = SimpleNamespace(keys_held=(True,), hold_light_press_age_ms=(200,),
                            hold_light_release_age_ms=(-1,), hit_light_age_ms=(9999,),
                            hit_light_judgment=("",))
    scene.hold_light_release_age_ms = (60,)
    fr._draw_legacy_hit_lighting(scene)
    assert fr.additive_draws[-1][-1] == (1,1,1,0.5)
    fr.additive_draws.clear()
    scene.hold_light_press_age_ms = (-1,)
    fr._draw_legacy_hit_lighting(scene)
    assert fr.additive_draws == []


def test_original_mode_survives_conversion_for_skin_source_selection(tmp_path, monkeypatch):
    from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
    from osu_mania_renderer_v2.beatmap import converter
    path = tmp_path / "map.osu"
    path.write_text("[General]\nMode:3\n[Difficulty]\nCircleSize:4\n[HitObjects]\n")
    native = parse_beatmap(path)
    assert native.source_mode == 3
    monkeypatch.setattr(converter,"convert_standard_to_mania",lambda **kw: native)
    path.write_text("[General]\nMode:0\n[Difficulty]\nCircleSize:4\n[HitObjects]\n")
    converted = parse_beatmap(path,allow_converted=True)
    assert converted.source_mode == 0
    assert converted.key_count == 4 and converted.notes == native.notes


def test_wiki_frame_clock_reaches_shared_stage_bottom_animation():
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(height=480)
    fr.stage_layout = legacy_stage_layout(4,None,render_width=640,render_height=480)
    fr.atlas = SimpleNamespace(global_source=lambda _:"user",
                              global_native_size=lambda _:(20,4),frame_count=lambda _:2)
    fr.upside_down = False
    draws = []
    fr._draw_direct = lambda *a,**kw: draws.append(kw["frame_index"])
    ctx = FrameContext(fr,None,SimpleNamespace(enable=lambda _:None,blend_func=None),
                       SimpleNamespace(use=lambda:None,clear=lambda *a:None),640,480,4,
                       scene=SceneState(17,(),(False,)*4,VisualMods()))
    ctx.begin_frame()
    fr._draw_split_stage_foreground()
    assert draws == [1]
