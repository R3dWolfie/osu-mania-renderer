"""Stable Hit()/LN input authority, including actual plan -> scene wiring."""
import asyncio
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import os

from PIL import Image
import pytest

from osu_mania_renderer_v2.beatmap.judgments import compute_judgments
from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note, ReplayInfo, RenderOptions, VisualMods
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.legacy_mania_events import (
    LegacyHitLightFact, StableManiaWindows, build_legacy_mania_presentation,
)
from osu_mania_renderer_v2.render.scene import LightingNEvent, SceneState, lighting_n_events_at, sliding_colour_mix
from osu_mania_renderer_v2.wiki_elements.context import FrameContext
from osu_mania_renderer_v2.wiki_elements.notes import notes as wiki_notes

render_module = importlib.import_module('osu_mania_renderer_v2.render.render')


def presentation(notes, edges=(), **kwargs):
    return build_legacy_mania_presentation(tuple(notes), tuple(KeyEvent(t,k) for t,k in edges), 2, **kwargs)


def normal(evidence, now):
    return lighting_n_events_at(evidence.normal_hit_facts, evidence.normal_hit_times, now, evidence.rate)


def real_plan(tmp_path, monkeypatch, *, objects, edges, counts, mods=0):
    """Real .osu parsing, mods, judgment simulation and tally reconciliation."""
    path = tmp_path/'fixture.osu'
    path.write_text('osu file format v14\n[General]\nMode:3\n[Metadata]\nTitle:truth\nArtist:test\n'
                    'Creator:test\nVersion:review\n[Difficulty]\nCircleSize:2\nOverallDifficulty:5\n'
                    '[TimingPoints]\n0,500,4,1,0,100,1,0\n[HitObjects]\n'+objects+'\n')
    replay = ReplayInfo(3,hashlib.md5(path.read_bytes()).hexdigest(),'test','',mods,
                        tuple(KeyEvent(t,k) for t,k in edges),1000000,100,10,*counts,'X',300,300)
    monkeypatch.setattr(render_module, 'parse_replay', lambda _: replay)
    async def encoder(*_):
        return 'libx264'
    monkeypatch.setattr(render_module, 'probe_encoder', encoder)
    return asyncio.run(render_module.build_render_plan(
        osr_path=tmp_path/'fixture.osr',beatmap_dir=tmp_path,output_path=tmp_path/'unused.mp4',
        options=RenderOptions((320,240),60,encoder='libx264')))


@pytest.mark.parametrize('factual_hit', [False,True])
def test_actual_plan_reconciliation_cannot_fabricate_or_suppress_lighting_n(tmp_path,monkeypatch,factual_hit):
    edges = [(1000,1),(1040,0)] if factual_hit else []
    counts = (0,0,0,0,0,1) if factual_hit else (1,0,0,0,0,0)
    plan = real_plan(tmp_path,monkeypatch,objects='64,192,1000,1,0,0:0:0:0:',edges=edges,counts=counts)
    raw = compute_judgments(plan.modded.notes,plan.replay.key_events,2,overall_difficulty=5)
    assert raw.events[0].judgment == ('geki' if factual_hit else 'miss')
    assert plan.judgment_events[0].judgment == ('miss' if factual_hit else 'geki')
    expected = (LightingNEvent(0,'geki',50),) if factual_hit else ()
    evidence = plan.legacy_presentation
    for now in (1050,1300,999,1050):
        scene = render_module.build_frame_state(plan,now,0,100)[0]
        if now == 1050:
            assert scene.lighting_n_events == expected
    # Even a later replacement of the reconciled timeline has no authority.
    plan.judgment_timeline = []
    if hasattr(plan,'_tl_times'):
        del plan._tl_times
    assert render_module.build_frame_state(plan,1050,0,100)[0].lighting_n_events == expected
    assert plan.legacy_presentation is evidence
    with pytest.raises(FrozenInstanceError):
        evidence.rate = 2


def test_actual_score_v1_ln_final_is_visual_even_with_non_scoring_tail(tmp_path,monkeypatch):
    plan = real_plan(tmp_path,monkeypatch,objects='64,192,1000,128,0,2000:0:0:0:0:',
                     edges=[(1000,1),(2000,0)],counts=(1,0,0,0,0,0))
    assert any(event.is_tail and not event.scoring for event in plan.judgment_events)
    for now in (1050,2050,1950,2050):
        scene = render_module.build_frame_state(plan,now,0,100)[0]
        assert scene.lighting_n_events == ((LightingNEvent(0,'geki',50),) if now == 2050 else ())
    assert plan.legacy_presentation.normal_hit_facts == (LegacyHitLightFact(2000,0,'geki','ln-final'),)


def test_actual_plan_missed_head_late_press_drives_l_and_sliding_without_freezing_geometry(tmp_path,monkeypatch):
    plan = real_plan(tmp_path,monkeypatch,objects='64,192,1000,128,0,2000:0:0:0:0:',
                     edges=[(1500,1),(2000,0)],counts=(0,0,0,0,0,1))
    assert plan.hold_visual_states == {}  # preserve the existing frozen-head authority
    assert plan.legacy_presentation.sliding_intervals == ((1500,2000),)
    for now in (1550,2050,1400,1550):
        scene = render_module.build_frame_state(plan,now,0,100)[0]
        if now == 1550:
            assert scene.legacy_hold_colour_mix == pytest.approx(50/300)
            assert scene.legacy_long_lights[0].alpha == pytest.approx(50/80)
            assert scene.lighting_n_events == ()
        elif now == 2050:
            assert scene.lighting_n_events == (LightingNEvent(0,'50',50),)


def test_actual_plan_lighting_l_resumes_after_drop_without_refreezing_head(tmp_path,monkeypatch):
    plan = real_plan(tmp_path,monkeypatch,objects='64,192,1000,128,0,2000:0:0:0:0:',
                     edges=[(1000,1),(1300,0),(1340,1),(2000,0)],counts=(1,0,0,0,0,0))
    assert plan.hold_visual_states[(0,1000)].drop_time_ms == 1300
    for now in (1380,2050,1320,1380):
        scene = render_module.build_frame_state(plan,now,0,100)[0]
        if now == 1380:
            assert scene.legacy_long_lights[0].animation_age_ms == 40
            assert scene.legacy_long_lights[0].alpha == pytest.approx(5/6)
            assert scene.lighting_n_events == ()
        elif now == 2050:
            assert scene.lighting_n_events == (LightingNEvent(0,'50',50),)


@pytest.mark.parametrize('offset,tier', [(0,'geki'),(30,'300'),(70,'katu'),(100,'100'),(-130,'50')])
def test_tap_hit_source_windows_create_one_normal_light(offset,tier):
    evidence = presentation([Note(0,1000)],[(1000+offset,1),(1200,0)])
    assert evidence.normal_hit_facts == (LegacyHitLightFact(1000+offset,0,tier,'tap'),)
    assert evidence.sliding_intervals == ()
    assert all(light.alpha == 0 for light in evidence.long_lights_at(1100))


def test_tap_automatic_miss_gate_precedes_late_input_even_inside_50_window():
    evidence = presentation([Note(0,1000)],[(1130,1)])
    assert evidence.normal_hit_facts == ()  # W100=112, W50=136 at OD5


def test_early_consumed_tap_blocks_next_same_column_note_until_its_start():
    evidence = presentation([Note(0,1000),HoldNote(0,1050,2000)],
                            [(900,1),(920,0),(950,1),(970,0),(1010,1),(2000,0)])
    assert evidence.normal_hit_facts == (LegacyHitLightFact(900,0,'100','tap'),
                                        LegacyHitLightFact(2000,0,'300','ln-final'))
    assert evidence.sliding_intervals == ((1010,2000),)


def test_source_eligibility_and_owned_ln_press_are_required_for_lighting_l():
    too_early = presentation([HoldNote(0,1000,2000)],[(800,1),(1500,0)])
    assert too_early.sliding_intervals == too_early.normal_hit_facts == ()
    assert too_early.long_lights_at(1000)[0].alpha == 0
    early = presentation([HoldNote(0,1000,2000)],[(830,1),(2000,0)])
    assert early.sliding_intervals == ((830,2000),)
    assert early.long_lights_at(900)[0].alpha == pytest.approx(70/80)
    assert early.normal_hit_facts == (LegacyHitLightFact(2000,0,'50','ln-final'),)


def test_simultaneous_final_hits_follow_original_note_order_across_columns():
    evidence = presentation([HoldNote(1,900,2000),HoldNote(0,1000,2000)],
                            [(900,2),(1000,3),(2000,0)])
    assert evidence.normal_hit_facts == (LegacyHitLightFact(2000,1,'geki','ln-final'),
                                        LegacyHitLightFact(2000,0,'geki','ln-final'))


@pytest.mark.parametrize('release,final_time,tier', [(2000,2000,'geki'),(None,2136,'katu'),(2120,2136,'katu')])
def test_ln_hitstart_never_normal_light_final_hit_uses_source_autogate(release,final_time,tier):
    evidence = presentation([HoldNote(0,1000,2000)],[(1000,1)]+([] if release is None else [(release,0)]))
    assert normal(evidence,1050) == ()
    assert evidence.normal_hit_facts == (LegacyHitLightFact(final_time,0,tier,'ln-final'),)
    assert normal(evidence,final_time+50) == (LightingNEvent(0,tier,50),)
    # IsSliding stops at W100's automatic branch; L stays on through Ignore
    # until the successful final Hit at W50, even if replay releases meanwhile.
    assert evidence.sliding_intervals == ((1000,min(final_time,2112)),)
    if final_time == 2136:
        assert evidence.long_lights_at(2125)[0].alpha == 1
        assert evidence.long_lights_at(2196)[0].alpha == .5


@pytest.mark.parametrize('edges,intervals', [
    ([(1000,1),(2000,0)],((1000,2000),)),
    ([(980,1),(2000,0)],((980,2000),)),
    ([(1500,1),(2000,0)],((1500,2000),)),
    ([(1000,1),(1300,0)],((1000,1300),)),
    ([(1000,1),(1300,0),(1400,1),(2000,0)],((1000,1300),(1400,2000))),
    ([(1000,1),(1300,0),(1400,1),(1450,0),(1500,1),(2000,0)],((1000,1300),(1400,1450),(1500,2000))),
])
def test_ln_sliding_and_long_light_lifecycle_includes_missed_head_and_represses(edges,intervals):
    evidence = presentation([HoldNote(0,1000,2000)],edges)
    assert evidence.sliding_intervals == intervals
    expected = {}
    for now in (1020,1320,1420,1470,1550,2050,1020,1550):
        value = (normal(evidence,now),evidence.long_lights_at(now),sliding_colour_mix(intervals,now))
        if now in expected:
            assert value == expected[now]
        expected[now] = value
    for start,stop in intervals:
        assert evidence.long_lights_at(start+10)[0].alpha > 0
        assert evidence.long_lights_at(stop+121)[0].alpha == 0 or any(
            later <= stop+121 < end for later,end in intervals)
    if intervals[-1][1] == 1300:
        assert evidence.normal_hit_facts == ()  # drop and final miss never AddHitLight


def test_lighting_l_repress_resets_animation_and_fades_from_current_alpha():
    evidence = presentation([HoldNote(0,1000,2000)],[(1000,1),(1300,0),(1340,1),(2000,0)])
    assert evidence.long_lights_at(1320)[0].alpha == pytest.approx(5/6)
    resumed = evidence.long_lights_at(1340)[0]
    assert resumed.animation_age_ms == 0
    assert resumed.alpha == pytest.approx(2/3)
    assert evidence.long_lights_at(1380)[0].alpha == pytest.approx(5/6)
    assert normal(evidence,1380) == ()


def test_break_caps_final_ln_tier_and_never_creates_its_own_normal_light():
    evidence = presentation([HoldNote(0,1000,2000)],[(900,1),(930,0),(1000,1),(2000,0)])
    assert evidence.normal_hit_facts == (LegacyHitLightFact(2000,0,'katu','ln-final'),)
    assert normal(evidence,950) == ()


def test_simultaneous_lns_union_empty_held_lane_and_press_after_finalisation():
    evidence = presentation([HoldNote(0,1000,2000),HoldNote(1,1500,2300)],
                            [(980,1),(1300,0),(1400,1),(1500,3),(2000,2),(2300,0),(2500,3)])
    assert evidence.sliding_intervals == ((980,1300),(1400,2300))
    assert all(light.alpha == 0 for light in evidence.long_lights_at(2550))
    empty = presentation([HoldNote(0,1000,2000)],[(900,2),(1500,2),(2500,2)])
    assert empty.sliding_intervals == empty.normal_hit_facts == ()
    assert all(light.alpha == 0 for light in empty.long_lights_at(1600))


def test_factual_concurrent_final_hits_keep_column_insertion_order():
    evidence = concurrent_evidence()
    assert evidence.normal_hit_facts == (LegacyHitLightFact(2000,0,'geki','ln-final'),
        LegacyHitLightFact(2050,1,'300','tap'),LegacyHitLightFact(2100,0,'geki','ln-final'))
    expected = (LightingNEvent(0,'geki',150),LightingNEvent(1,'300',100),LightingNEvent(0,'geki',50))
    for now in (2150,3000,1999,2150):
        if now == 2150:
            assert normal(evidence,now) == expected


def concurrent_evidence():
    return presentation([HoldNote(0,1000,2000),HoldNote(0,2020,2100),Note(1,2020)],
                        [(1000,1),(2000,0),(2020,1),(2050,3),(2070,1),(2100,0)])


@pytest.mark.parametrize('mods,rate,ok,meh', [(0,1,112,136),(Mod.HR,1,80,97),
    (Mod.EZ,1,156,190),(Mod.DT,1.5,112,136),(Mod.HT,.75,112,136)])
def test_source_windows_and_automatic_gates_preserve_rate_and_difficulty_mods(mods,rate,ok,meh):
    windows = StableManiaWindows.build(5,mods=mods,rate=rate)
    assert (windows.ok,windows.meh) == (ok,meh)
    evidence = presentation([HoldNote(0,1000,2000)],[(1000,1)],mods=mods,rate=rate)
    assert evidence.sliding_intervals == ((1000,2000+ok),)
    assert evidence.normal_hit_facts[0].time_ms == 2000+meh
    assert normal(evidence,2000+meh+50/rate)[0].age_ms == pytest.approx(50)


def test_converted_windows_are_fixed_and_perfect_mod_can_suppress_add_hit_light():
    assert StableManiaWindows.build(9,source_mode=0).great == 34
    assert StableManiaWindows.build(4,source_mode=0).great == 47
    assert presentation([Note(0,1000)],[(1070,1)],mods=Mod.PF).normal_hit_facts == ()


@pytest.mark.skipif(os.getenv('RUN_SLOW') != '1',reason='opt-in real GL')
def test_factual_lights_sliding_and_repress_share_wiki_monolithic_painter(tmp_path):
    (tmp_path/'skin.ini').write_text('[Mania]\nKeys:2\nLightingN:flash\nLightingL:long\n')
    for kind in ('flash','long'):
        for frame in range(3):
            Image.new('RGBA',(8,8),'white').save(tmp_path/f'{kind}-{frame}.png')
    evidence = concurrent_evidence()
    resumed = presentation([HoldNote(0,1000,2000)],[(1000,1),(1300,0),(1340,1),(2000,0)])
    with HeadlessGl(320,240) as gl:
        renderer = FrameRenderer(RenderContext(gl.ctx,gl.fbo,320,240,2),skin_dir=tmp_path)
        records = []
        renderer._draw_additive_sprite_idx = lambda *args: records.append(args)
        renderer._draw_notes = lambda *_: None
        renderer._draw_custom_legacy_combo = lambda scene,**__: records.append(('colour',scene.legacy_hold_colour_mix))
        renderer._draw_legacy_scroll_guides = lambda *_: None
        for carrier,now in ((evidence,2150),(resumed,1380),(evidence,2150)):
            scene = SceneState(now,(),(False,False),VisualMods(),lighting_n_events=normal(carrier,now),
                               legacy_long_lights=carrier.long_lights_at(now),
                               legacy_hold_colour_mix=sliding_colour_mix(carrier.sliding_intervals,now))
            outputs = []
            for wiki in (False,True):
                records.clear()
                if wiki:
                    ctx = FrameContext(renderer,None,gl.ctx,gl.fbo,320,240,2,scene=scene)
                    wiki_notes(element='notes',skin=None,assets=None,variables=None,ctx=ctx)
                else:
                    renderer._draw_legacy_stage_gameplay(scene)
                outputs.append(tuple(records))
            assert outputs[0] == outputs[1]
            normal_base = renderer.atlas.index_of('lighting_n')
            long_base = renderer.atlas.index_of('lighting_l')
            assert ('colour',scene.legacy_hold_colour_mix) in outputs[0]
            draws = [draw for draw in outputs[0] if draw[0] != 'colour']
            if now == 2150:
                assert [draw[0]-normal_base for draw in draws if normal_base <= draw[0] < normal_base+3] == [2,1,0]
                assert [draw[1]+draw[3]/2 for draw in draws if normal_base <= draw[0] < normal_base+3] == [
                    renderer.col_x[c]+renderer.col_w[c]/2 for c in (0,1,0)]
            else:
                long_draw = next(draw for draw in draws if long_base <= draw[0] < long_base+3)
                assert long_draw[0]-long_base == 0  # 40ms since re-press; source 170/3 ms per frame
                assert long_draw[-1][-1] == pytest.approx(5/6)
                assert not any(normal_base <= draw[0] < normal_base+3 for draw in draws)
