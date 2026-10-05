"""Round 3.5 source contracts; all assets and timing inputs are synthetic."""
from dataclasses import replace
from types import SimpleNamespace
import os

import moderngl
import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.beatmap import _parse_timing_points
from osu_mania_renderer_v2.beatmap.models import HoldNote, VisualMods
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection, parse_skin_ini
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.gpu.renderer import legacy_mania_position_gl
from osu_mania_renderer_v2.render.legacy_mania_timing import LegacyManiaTiming, animation_frame
from osu_mania_renderer_v2.render.legacy_mania_events import LegacyHitLightFact, build_legacy_mania_presentation
from osu_mania_renderer_v2.beatmap.models import KeyEvent
from osu_mania_renderer_v2.render.scene import (
    SceneState, VisibleNote, LightingNEvent,
    lighting_n_events_at, sliding_colour_mix,
)
from osu_mania_renderer_v2.wiki_elements.context import FrameContext
from osu_mania_renderer_v2.wiki_elements.notes import _draw_notes_body, notes


def timing(rows="0,500,4,1,0,100,1,0", first=3000, last=6000):
    return LegacyManiaTiming.build(_parse_timing_points(rows), first_note_ms=first,
                                  last_note_ms=last, scroll_speed=20)


def test_constant_speed_exact_transformation_endpoints_and_phase():
    t = timing()
    assert t.offsets == (-4000,)
    assert t.velocities == pytest.approx((0.7,))
    assert t.movements(3000, 402)[0].start_ms == 2425
    assert t.first_movement_start(4000, 402) == 3425
    assert t.first_movement_start(3000, 402) != 3000 - 600
    assert t.position(3000, 3000, 402) == 402
    assert animation_frame(2450, 2425, 3) == 1
    assert animation_frame(50, 0, 20) == 3
    assert animation_frame(100, 0, 20) == 6
    for now in (2900, 2450, 3000, 2450):
        assert t.position(3000, now, 402) == pytest.approx((now-2425)/575*402)


def test_opening_inherited_sv_collapses_and_boundary_uses_previous_speed():
    t = timing("0,500,4,1,0,100,1,0\n1000,-50,4,1,0,100,0,0\n3000,-100,4,1,0,100,0,0")
    assert t.offsets == (-4000, 3000)
    assert t.velocities == pytest.approx((1.4, 0.7))
    assert t.first_movement_start(3000, 402) == 2712
    assert t.movements(3000, 402)[0].end_ms == 3000
    assert t.movements(3000, 402)[1].end_ms == 3111
    assert t.position(3000, 3050, 402) == pytest.approx(402 + 78*50/111)


def test_measure_spacing_red_changes_and_fractional_metadata():
    points = _parse_timing_points("0,500,3,1,0,100,1,0\n2000,-50,3,1,0,100,0,0\n4000.5,400,5,1,0,100,1,0")
    assert points[-1].raw_time_ms == 4000.5
    assert points[-1].time_signature == 5
    t = LegacyManiaTiming.build(points, first_note_ms=1000, last_note_ms=8000, scroll_speed=20)
    assert t.measures == (-3000, -1500, 0, 1500, 3000, 4000.5, 6000.5, 8000.5)
    assert timing().measures == (-4000, -2000, 0, 2000, 4000, 6000)
    assert LegacyManiaTiming.build((), first_note_ms=0, last_note_ms=1000) is None
    assert t.position(3000, 3000, 402) == 402
    assert t.distance_at(3000)-t.distance_at(2500) > t.distance_at(1500)-t.distance_at(1000)


@pytest.mark.parametrize('last', [0, 500])
def test_tempo_after_entire_chart_span_uses_untimed_fallback(last):
    assert timing('1000,500,4,1,0,100,1,0', first=0, last=last) is None


@pytest.mark.parametrize("first,expected", [(3000,(2000,1000,0)), (0,(-1000,-2000,-3000)), (-2500,(-3500,))])
def test_warning_count_stops_at_source_prestart(first, expected):
    assert timing(first=first).warning_times == expected


def test_warning_lookup_is_static_and_respects_named_active_source_tiers(tmp_path):
    skin, beatmap = tmp_path/'skin', tmp_path/'map'
    skin.mkdir(); beatmap.mkdir()
    section = ManiaSection(keys=4, warning_arrow='named')
    def resolve():
        return SpriteAtlas._resolve_global('warning_arrow', skin_dir=skin, beatmap_dir=beatmap, section=section)
    assert resolve()[1] == 'classic'
    Image.new('RGBA',(4,4),'red').save(skin/'named-0.png')
    assert resolve()[1] == 'classic'  # Skin.Load, not LoadAnimation
    Image.new('RGBA',(5,5),'red').save(skin/'mania-warningarrow.png')
    Image.new('RGBA',(6,6),'blue').save(beatmap/'mania-warningarrow.png')
    assert resolve()[0][0].width == 6
    Image.new('RGBA',(7,7),'green').save(skin/'named.png')
    assert resolve()[1] == 'user' and resolve()[0][0].width == 7
    Image.new('RGBA',(9,9),'white').save(beatmap/'named.png')
    assert resolve()[1] == 'beatmap' and resolve()[0][0].width == 9
    (skin/'skin.ini').write_text('[Mania]\nKeys:4\nWarningArrow:named\n')
    assert parse_skin_ini(skin).mania_for_keycount(4).warning_arrow == 'named'


def test_lighting_carrier_preserves_factual_final_hits_and_direct_seeks():
    events = tuple(LegacyHitLightFact(when,col,tier,'tap') for when,col,tier in
                   [(1000,0,'300'),(1050,1,'50'),(1100,0,'geki')])
    times = [e.time_ms for e in events]
    expected = (LightingNEvent(0,'300',150), LightingNEvent(1,'50',100), LightingNEvent(0,'geki',50))
    for now in (1150, 1400, 1000, 1150):
        result = lighting_n_events_at(events, times, now)
        if now == 1150:
            assert result == expected
        elif now == 1400:
            assert result == ()
    assert lighting_n_events_at(events,times,1250) == (LightingNEvent(0,'geki',150),)


def test_sliding_uses_ln_input_early_press_repress_and_column_union():
    held = (HoldNote(0,1000,2000), HoldNote(1,1500,2300))
    evidence = build_legacy_mania_presentation(held, tuple(KeyEvent(t,k) for t,k in
        [(900,4),(980,5),(1300,4),(1400,5),(1500,7),(2000,6),(2300,4)]), 3)
    intervals = evidence.sliding_intervals
    assert intervals == ((980,1300),(1400,2300))
    assert sliding_colour_mix(intervals, 1130) == 0.5
    assert sliding_colour_mix(intervals, 1601) == pytest.approx(201/300)
    assert sliding_colour_mix(intervals, 2450) == 0.5
    assert sliding_colour_mix(intervals, 2601) == 0


def test_short_sliding_reversals_use_oldest_active_transform_not_interrupted_colour():
    intervals = ((1000,1050),(1100,1600))
    for now, expected in [(1000,0),(1150,.5),(1300,1),(1301,1-251/300),(1351,251/300),(1450,1),(1750,.5),(2000,0)]:
        assert sliding_colour_mix(intervals,now) == pytest.approx(expected)


slow = pytest.mark.skipif(os.getenv('RUN_SLOW') != '1', reason='opt-in real GL')


@slow
@pytest.mark.parametrize('up',[False,True])
@pytest.mark.parametrize('keys',[4,10])
def test_scrolling_guides_stage_local_geometry_and_compositor_parity(tmp_path,up,keys):
    (tmp_path/'skin.ini').write_text(f'[General]\nVersion:2.7\n[Mania]\nKeys:{keys}\nUpsideDown:{int(up)}\nBarlineHeight:8\nColourBarline:40,80,120,255\n')
    t=timing()
    with HeadlessGl(640,480) as gl:
        fr=FrameRenderer(RenderContext(gl.ctx,gl.fbo,640,480,keys),skin_dir=tmp_path)
        calls=[]
        fr._draw_sprite=lambda *a,**kw:calls.append(('bar',a,kw))
        fr._draw_direct=lambda *a,**kw:calls.append(('arrow',a,kw))
        fr._draw_legacy_scroll_guides(SceneState(2000,(),(False,)*keys,VisualMods(),legacy_timing=t))
        stages=fr.stage_layout.stages
        bars=[c for c in calls if c[0]=='bar']; arrows=[c for c in calls if c[0]=='arrow']
        assert len(bars)==len(arrows)==len(stages)
        for stage,(_,bar,_),(_,arrow,uv) in zip(stages,bars,arrows):
            assert bar[1]==stage.x and bar[3]==stage.width
            assert bar[4]==5 and bar[5]==pytest.approx((40/255,80/255,120/255,1))
            assert arrow[1]+arrow[3]/2==pytest.approx(stage.center_x)
            assert uv['source_bottom']==int(up)
        # Both entry points consume the exact shared .62/.63 guide pass.
        fr._draw_notes=lambda *a:None
        fr._draw_custom_legacy_combo=lambda *a,**kw:None
        fr._draw_receptors=lambda *a:None
        fr._draw_legacy_stage_foreground=lambda:None
        fr._draw_legacy_hit_lighting=lambda *a:None
        fr._draw_combo_and_judgment=lambda *a,**kw:None
        scene=SceneState(2000,(),(False,)*keys,VisualMods(),legacy_timing=t)
        snapshots=[]
        for wiki in (False,True):
            calls.clear()
            ctx=FrameContext(fr,None,gl.ctx,gl.fbo,640,480,keys,scene=scene)
            if wiki: notes(element='notes',skin=None,assets=None,variables=None,ctx=ctx)
            else: fr._draw_legacy_stage_gameplay(scene)
            snapshots.append(list(calls))
        assert snapshots[0]==snapshots[1]


@slow
@pytest.mark.parametrize('up',[False,True])
@pytest.mark.parametrize('part',['tap','head','tail','fallback-tail'])
def test_animated_caps_keep_initial_scale_use_own_phase_and_update_origin(tmp_path,up,part):
    for kind in ('tap','head','tail'):
        if kind=='tail' and part=='fallback-tail': continue
        for frame,size in enumerate(((8,8),(16,4),(4,16))):
            Image.new('RGBA',size,('red','green','blue')[frame]).save(tmp_path/f'{kind}-{frame}.png')
    Image.new('RGBA',(8,8),'white').save(tmp_path/'body.png')
    (tmp_path/'skin.ini').write_text(f'[General]\nVersion:2.7\n[Mania]\nKeys:1\nUpsideDown:{int(up)}\nNoteImage0:tap\nNoteImage0H:head\nNoteImage0T:tail\nNoteImage0L:body\n')
    t=timing()
    with HeadlessGl(320,480) as gl:
        fr=FrameRenderer(RenderContext(gl.ctx,gl.fbo,320,480,1),skin_dir=tmp_path)
        fr.col_x,fr.col_w=(100,),(32,)
        note=VisibleNote(0,part!='tap',1,1,.5,time_ms=3000,end_time_ms=3080)
        scene=SceneState(2522,(note,),(False,),VisualMods(),legacy_timing=t)
        captured=[]
        fr._draw_legacy_hold_body=lambda *a,**kw:None
        fr._draw_sprite_idx=lambda *a:captured.append(a)
        fr._draw_sprite_idx_cropped_y=lambda *a,**kw:captured.append(a)
        kind={'tap':'note_tap','head':'note_hold_head','tail':'note_hold_tail','fallback-tail':'note_hold_tail'}[part]
        base=fr.atlas.column_slot_index(kind,0)
        expected_frame=1 if 'tail' in part else 2
        width,height=(64,16) if expected_frame==1 else (16,64)
        assert fr._legacy_cap_dimensions(kind,0,expected_frame)==(width,height)
        observed=[]
        for now,wiki in ((2522,False),(3000,False),(2522,True),(2522,False)):
            captured.clear(); current=replace(scene,t_ms=now)
            ctx=FrameContext(fr,None,gl.ctx,gl.fbo,320,480,1,scene=current)
            if wiki: _draw_notes_body(ctx)
            else: fr._draw_notes(current)
            if now!=2522: continue
            cap=next(c for c in captured if c[0]==base+expected_frame)
            assert cap[3:5]==(width,height)
            assert cap[1]+width/2==116
            when=3080 if 'tail' in part else 3000
            anchor=legacy_mania_position_gl(t.position(when,now,402,32),480,up)
            assert cap[2]==pytest.approx(anchor-height if up else anchor)
            observed.append(cap)
        assert observed[0]==observed[1]==observed[2]


@slow
def test_concurrent_normal_lights_have_independent_frames_and_fades_in_insertion_order(tmp_path):
    (tmp_path/'skin.ini').write_text('[Mania]\nKeys:2\nLightingN:flash\n')
    for i in range(3): Image.new('RGBA',(8,8),'white').save(tmp_path/f'flash-{i}.png')
    with HeadlessGl(320,240) as gl:
        fr=FrameRenderer(RenderContext(gl.ctx,gl.fbo,320,240,2),skin_dir=tmp_path)
        records=[]
        fr._draw_custom_legacy_lighting=lambda *a,**kw:records.append(('L',kw['c'],kw['draw_normal']))
        fr._draw_additive_sprite_idx=lambda *a:records.append(('N',a))
        scene=SceneState(1150,(),(False,False),VisualMods(),lighting_n_events=(LightingNEvent(0,'300',150),LightingNEvent(1,'50',100),LightingNEvent(0,'geki',50)))
        fr._draw_legacy_hit_lighting(scene)
        assert records[:2]==[('L',0,False),('L',1,False)]
        assert len(records)==5
        frames=[r[1][0]-fr.atlas.index_of('lighting_n') for r in records[2:]]
        assert frames==[2,1,0]
        assert records[2][1][-1][-1]==pytest.approx(50/120)
        assert records[-1][1][-1][-1]==pytest.approx(50/80)


@slow
@pytest.mark.parametrize('authored',[False,True])
def test_combo_colour_is_shared_across_split_stages_and_break_keeps_own_colour(tmp_path,authored):
    colour='ColourHold:20,80,140,128\n' if authored else ''
    (tmp_path/'skin.ini').write_text('[General]\nVersion:2.7\n[Mania]\nKeys:10\nColourBreak:200,30,10,255\n'+colour)
    with HeadlessGl(640,480) as gl:
        fr=FrameRenderer(RenderContext(gl.ctx,gl.fbo,640,480,10),skin_dir=tmp_path)
        calls=[]
        fr._legacy_font_available=lambda *a:True
        fr._draw_legacy_text=lambda *a,**kw:calls.append((a,kw))
        fr._draw_custom_legacy_judgment=lambda *a,**kw:None
        scene=SceneState(1500,(),(False,)*10,VisualMods(),combo=15,
                         combo_break_previous_value=12,combo_break_age_ms=100,
                         legacy_hold_colour_mix=.5)
        fr._draw_combo_and_judgment(scene)
        main=[kw for args,kw in calls if args[1]=='15']
        broken=[kw for args,kw in calls if args[1]=='12']
        assert len(main)==len(broken)==2
        assert main[0]['tint']==main[1]['tint']
        hold=(20,80,140,128) if authored else (255,199,51,255)
        assert main[0]['tint']==pytest.approx(tuple(int(255+(v-255)*.5)/255 for v in hold))
        assert broken[0]['tint'][:3]==pytest.approx((200/255,30/255,10/255))


@slow
@pytest.mark.parametrize('up',[False,True])
def test_changing_tap_frame_actual_pixels_and_both_paths(tmp_path,up):
    for i,size in enumerate(((8,8),(16,4),(4,16))):
        Image.new('RGBA',size,('red','green','blue')[i]).save(tmp_path/f'tap-{i}.png')
    (tmp_path/'skin.ini').write_text(f'[General]\nVersion:2.7\n[Mania]\nKeys:1\nUpsideDown:{int(up)}\nNoteImage0:tap\n')
    t=timing()
    with HeadlessGl(320,480) as gl:
        fr=FrameRenderer(RenderContext(gl.ctx,gl.fbo,320,480,1),skin_dir=tmp_path)
        fr.col_x,fr.col_w=(100,),(32,)
        note=VisibleNote(0,False,1,1,1,time_ms=3000)
        scene=SceneState(2522,(note,),(False,),VisualMods(),legacy_timing=t)
        anchor=legacy_mania_position_gl(t.position(3000,2522,402),480,up)
        y=int(anchor-32 if up else anchor+32)
        pixels=[]
        for wiki in (False,True):
            gl.fbo.use(); gl.fbo.clear(0,0,0,1)
            gl.ctx.enable(moderngl.BLEND)
            gl.ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
            ctx=FrameContext(fr,None,gl.ctx,gl.fbo,320,480,1,scene=scene)
            if wiki: _draw_notes_body(ctx)
            else: fr._draw_notes(scene)
            fr._flush_sprite_batch()
            inside=tuple(gl.fbo.read(viewport=(116,y,1,1),components=3))
            outside=tuple(gl.fbo.read(viewport=(105,y,1,1),components=3))
            assert inside==(0,0,255) and outside==(0,0,0)
            pixels.append((inside,outside))
        assert pixels[0]==pixels[1]


@pytest.mark.parametrize('version,ratio',[(2.3,1),(2.4,.5),(2.7,.5)])
def test_warning_arrow_version_and_minimum_width_ratio(version,ratio):
    fr=object.__new__(FrameRenderer)
    fr.rc=SimpleNamespace(height=480,key_count=4)
    fr.col_w=(15,15,15,15)
    fr.skin_ini=SimpleNamespace(legacy_version=version)
    fr.mania_section=ManiaSection(keys=4)
    fr.upside_down=False
    fr.stage_layout=SimpleNamespace(stages=(SimpleNamespace(x=100,width=60,center_x=130),))
    fr.atlas=SimpleNamespace(global_native_size=lambda slot:(64,32))
    fr._draw_sprite=lambda *a,**kw:None
    calls=[]
    fr._draw_direct=lambda *a,**kw:calls.append((a,kw))
    fr._draw_legacy_scroll_guides(SceneState(2000,(),(False,)*4,VisualMods(),legacy_timing=timing()))
    arrow=calls[0][0]
    assert arrow[3:5]==pytest.approx((40*ratio,20*ratio))
    assert arrow[1]+arrow[3]/2==130
    assert arrow[2]+arrow[4]==pytest.approx(78)  # flipped TopCentre at HitPosition402


def test_full_hold_movement_extends_beyond_head_sprite_and_uses_integer_endpoints():
    t=timing()
    assert t.position(3000,4000,402,16) is None  # head sprite expired
    body=t.movements(3000,402,8,1000)
    assert body[1].end_ms==4000 and body[1].end_y==1102
    assert t.position(3000,3500,402,8,1000)==752
    assert body[-1].end_y==1188


def test_rate_preserves_audio_clock_animation_and_source_integer_endpoints():
    t=LegacyManiaTiming.build(_parse_timing_points('0,500,4,1,0,100,1,0'),first_note_ms=2000,last_note_ms=4000,scroll_speed=20,rate=1.5)
    # Audio speed v=.7/1.5, source Time1=int(3000-402/v)=2138.
    assert t.first_movement_start(2000,402)==2138/1.5
    assert animation_frame(50,0,20,t.rate)==4
    assert animation_frame(100,0,20,t.rate)==9
    assert t.position(2000,2000,402)==402
    assert t.warning_times==pytest.approx((2000-1000/1.5,2000-2000/1.5,0))
    event=(LegacyHitLightFact(1000,0,"300","tap"),)
    assert lighting_n_events_at(event,[1000],1100,rate=1.5)==(LightingNEvent(0,"300",150),)
    assert lighting_n_events_at(event,[1000],1140,rate=1.5)==()
