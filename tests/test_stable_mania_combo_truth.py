"""Stable client combo from the shared raw-input authority, separate from score."""
import asyncio
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import importlib
import lzma
import struct
from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note, RenderOptions
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.beatmap.replay import parse_replay
from osu_mania_renderer_v2.render.legacy_mania_events import build_legacy_mania_presentation
from osu_mania_renderer_v2.render.stable_mania_combo import StableComboFact, StableComboTimeline

render_module = importlib.import_module('osu_mania_renderer_v2.render.render')
replay_module = importlib.import_module('osu_mania_renderer_v2.beatmap.replay')


def events(notes, edges=(), **kwargs):
    return build_legacy_mania_presentation(tuple(notes),tuple(KeyEvent(*e) for e in edges),2,
                                          include_combo=True,**kwargs)


def facts(notes, edges=(), **kwargs):
    return events(notes,edges,**kwargs).stable_combo_facts


def hold_times(carrier):
    return [fact.time_ms for fact in carrier.stable_combo_facts if fact.source == 'ln-hold']


@pytest.mark.parametrize('version,mods,lazer,mw,aw', [
    (20261001,0,False,320,300), (20261001,Mod.V2,False,305,305),
    (30000000,0,True,305,305), (30000001,Mod.V2,True,305,305), (29999999,0,False,320,300),
])
def test_parser_client_provenance_is_independent_of_score_v2_weights(tmp_path,monkeypatch,version,mods,lazer,mw,aw):
    path=tmp_path/'stub.osr'; path.write_bytes(b'stub')
    raw=SimpleNamespace(mode=3,game_version=version,mods=mods,replay_data=[],count_geki=1,count_300=1,
                        count_katu=1,count_100=1,count_50=1,count_miss=1,beatmap_hash='map',username='player',
                        replay_hash='replay',score=123,max_combo=456)
    monkeypatch.setattr(replay_module.Replay,'from_path',lambda _:raw)
    parsed=parse_replay(path)
    assert parsed.is_lazer_replay is lazer
    assert (parsed.mania_max_weight,parsed.mania_acc_weight)==(mw,aw)
    assert parsed.accuracy==round((aw+300+200+100+50)/(aw*6)*100,4)
    with pytest.raises(FrozenInstanceError):
        parsed.is_lazer_replay=not lazer


def test_taps_increment_miss_resets_and_success_after_break_resumes_at_one():
    result=facts([Note(0,1000),Note(0,1500),Note(0,2000)],[(1000,1),(1040,0),(2000,1),(2040,0)])
    assert result==(StableComboFact(1000,'increment','tap',0),StableComboFact(1612,'reset','miss',0),
                    StableComboFact(2000,'increment','tap',0))
    timeline=StableComboTimeline.build(result)
    assert [timeline.at(t).combo for t in (1050,1612,2050)]==[1,0,1]
    assert timeline.at(1650).break_previous_value==1
    assert timeline.at(1650).last_break_ms==1612


def test_perfect_converted_tap_resets_instead_of_incrementing():
    result=facts([Note(0,1000),Note(0,1300)],[(1000,1),(1040,0),(1370,1)],mods=Mod.PF)
    assert result[-1]==StableComboFact(1370,'reset','miss',0)
    assert StableComboTimeline.build(result).at(1400).combo==0


def test_clean_ln_head_is_ignore_hold_clock_ticks_and_final_hit_adds_one():
    carrier=events([HoldNote(0,1000,2000)],[(1000,1),(2000,0)])
    assert hold_times(carrier)==list(range(1100,2000,100))
    assert carrier.stable_combo_facts[-1]==StableComboFact(2000,'increment','ln-final',0)
    timeline=StableComboTimeline.build(carrier.stable_combo_facts)
    assert timeline.at(1000).combo==0
    assert timeline.at(1150).combo==1
    assert timeline.at(2000).combo==10
    assert timeline.at(2100).last_increment_ms==2000


def test_held_through_tail_gets_endtime_tick_then_source_final_hit():
    carrier=events([HoldNote(0,1000,2000)],[(1000,1)])
    assert hold_times(carrier)==list(range(1100,2001,100))
    assert carrier.stable_combo_facts[-1]==StableComboFact(2136,'increment','ln-final',0)
    assert StableComboTimeline.build(carrier.stable_combo_facts).at(2200).combo==11


@pytest.mark.parametrize('release,holds,source', [(1100,[],'ln-break'),(1200,[1100],'ln-break'),
                                                (2000,list(range(1100,2000,100)),'ln-final')])
def test_release_on_due_tick_runs_hit_branch_without_holding(release,holds,source):
    carrier=events([HoldNote(0,1000,2000)],[(1000,1),(release,0)])
    assert hold_times(carrier)==holds
    at_release=[fact for fact in carrier.stable_combo_facts if fact.time_ms==release]
    assert len(at_release)==1 and at_release[0].source==source


@pytest.mark.parametrize('press,expected', [(980,list(range(1080,2000,100))),
                                          (830,[1000]+list(range(1030,2000,100))),
                                          (1500,list(range(1600,2000,100)))])
def test_early_and_missed_head_late_presses_use_press_clock_not_ln_length(press,expected):
    carrier=events([HoldNote(0,1000,2000)],[(press,1),(2000,0)])
    assert hold_times(carrier)==expected
    assert not any(f.kind=='increment' and f.time_ms==press for f in carrier.stable_combo_facts)


def test_overdue_early_press_clock_adds_once_per_logical_audio_update():
    carrier=events([HoldNote(0,1000,2000)],[(770,1),(2000,0)],mods=Mod.EZ)
    assert hold_times(carrier)[:4]==[1000,1001,1070,1170]
    # LastScoreTime advances by 100 rather than restarting at StartTime.
    assert hold_times(carrier)[-1]==1970


def test_rising_edge_at_starttime_does_not_also_run_holding():
    carrier=events([HoldNote(0,1000,2000)],[(830,1),(1000,0),(1000,1),(2000,0)])
    assert not any(f.time_ms==1000 and f.source=='ln-hold' for f in carrier.stable_combo_facts)
    assert hold_times(carrier)[0]==1100


def test_short_ln_has_no_hold_tick_but_successful_final_hit_counts():
    carrier=events([HoldNote(0,1000,1050)],[(1000,1),(1050,0)])
    assert hold_times(carrier)==[]
    assert carrier.stable_combo_facts==(StableComboFact(1050,'increment','ln-final',0),)


def test_drop_resets_immediately_repress_restarts_clock_and_final_hit_resumes():
    carrier=events([HoldNote(0,1000,2000)],[(1000,1),(1250,0),(1320,1),(2000,0)])
    assert hold_times(carrier)==[1100,1200,1420,1520,1620,1720,1820,1920]
    timeline=StableComboTimeline.build(carrier.stable_combo_facts)
    broken=timeline.at(1260)
    assert (broken.combo,broken.break_previous_value,broken.last_break_ms)==(0,2,1250)
    assert timeline.at(1320).combo==0
    assert timeline.at(1420).combo==1
    assert timeline.at(2000).combo==7


def test_multiple_drop_repress_cycles_have_independent_reset_facts():
    carrier=events([HoldNote(0,1000,2500)],[(1000,1),(1250,0),(1320,1),(1570,0),(1610,1),(2500,0)])
    breaks=[f for f in carrier.stable_combo_facts if f.source=='ln-break']
    assert breaks==[StableComboFact(1250,'reset','ln-break',0),StableComboFact(1570,'reset','ln-break',0)]
    assert hold_times(carrier)==[1100,1200,1420,1520,*range(1710,2500,100)]
    assert StableComboTimeline.build(carrier.stable_combo_facts).at(2500).combo==9


def test_final_ln_miss_resets_instead_of_adding_a_tail_combo():
    carrier=events([HoldNote(0,1000,2000),Note(1,1800)],[(1000,1),(1250,0),(1800,2),(1840,0)])
    assert carrier.stable_combo_facts[-1]==StableComboFact(2112,'reset','ln-final',0)
    assert StableComboTimeline.build(carrier.stable_combo_facts).at(2112).combo==0


def test_perfect_mod_converts_final_ln_hit_but_not_holding_additions():
    carrier=events([HoldNote(0,1000,2000)],[(1000,1)],mods=Mod.PF)
    assert len(hold_times(carrier))==10
    assert carrier.stable_combo_facts[-1]==StableComboFact(2136,'reset','ln-final',0)


def test_simultaneous_lns_generate_independent_additions_in_source_order():
    carrier=events([HoldNote(1,1000,1500),HoldNote(0,1000,1500)],[(1000,3),(1500,0)])
    at_tick=[f for f in carrier.stable_combo_facts if f.time_ms==1100]
    assert at_tick==[StableComboFact(1100,'increment','ln-hold',1),StableComboFact(1100,'increment','ln-hold',0)]
    assert StableComboTimeline.build(carrier.stable_combo_facts).at(1500).combo==10


def test_same_column_owned_ln_blocks_tap_and_second_ln():
    carrier=events([HoldNote(0,1000,1800),Note(0,1200),HoldNote(0,1300,2000)],[(1000,1),(1800,0)])
    assert [f.time_ms for f in carrier.stable_combo_facts if f.source=='ln-hold']==list(range(1100,1800,100))
    assert not any(f.source=='tap' for f in carrier.stable_combo_facts)
    # The blocked tap is auto-missed only after the first LN relinquishes ownership.
    assert StableComboFact(1800,'reset','miss',0) in carrier.stable_combo_facts


@pytest.mark.parametrize('tap_first', [False,True])
def test_tap_auto_miss_and_ln_tick_at_same_time_use_object_source_order(tap_first):
    objects=[Note(1,988),HoldNote(0,1000,1500)] if tap_first else [HoldNote(0,900,1500),Note(1,988)]
    press=1000 if tap_first else 900
    carrier=events(objects,[(press,1),(1500,0)])
    tied=[f for f in carrier.stable_combo_facts if f.time_ms==1100]
    assert [f.source for f in tied]==(['miss','ln-hold'] if tap_first else ['ln-hold','miss'])
    expected=1 if tap_first else 0
    assert StableComboTimeline.build(carrier.stable_combo_facts).at(1100).combo==expected


def test_empty_held_lane_cannot_generate_any_combo():
    assert facts([],[(1000,3),(2000,0)])==()
    carrier=events([HoldNote(0,1000,2000)],[(1000,2),(2000,0)])
    assert not any(f.kind=='increment' for f in carrier.stable_combo_facts)


def test_logical_holding_update_also_observes_other_lanes_hold_breaks_in_source_order():
    carrier=events([HoldNote(0,900,1500),HoldNote(1,1000,2000)],[(900,3),(950,1),(1500,0)])
    at_tick=[f for f in carrier.stable_combo_facts if f.time_ms==1100]
    assert at_tick==[StableComboFact(1100,'increment','ln-hold',0),StableComboFact(1100,'reset','ln-break',1)]
    assert StableComboTimeline.build(carrier.stable_combo_facts).at(1100).combo==0


@pytest.mark.parametrize('tap_first',[False,True])
def test_positive_tap_and_hold_tick_same_input_frame_keep_source_order(tap_first):
    objects=[Note(1,1000),HoldNote(0,1100,1500)] if tap_first else [HoldNote(0,1000,1500),Note(1,1100)]
    carrier=events(objects,[(1000,1),(1100,3),(1140,1),(1500,0)])
    tied=[f for f in carrier.stable_combo_facts if f.time_ms==1100]
    assert [f.source for f in tied]==(['tap','ln-hold'] if tap_first else ['ln-hold','tap'])


@pytest.mark.parametrize('rate',[.75,1,1.5])
def test_hold_clock_is_100_audio_ms_at_each_rate(rate):
    carrier=events([HoldNote(0,1000,2000)],[(1000,1)],rate=rate)
    times=hold_times(carrier)
    assert times[0]==pytest.approx(1000+100/rate)
    assert all((b-a)*rate==pytest.approx(100) for a,b in zip(times,times[1:]))
    assert max(times)<=2000


def string_bytes(text):
    raw=text.encode(); n=len(raw); encoded=bytearray([11])
    while True:
        encoded.append((n & 127) | (128 if n>127 else 0)); n >>= 7
        if not n: return bytes(encoded)+raw


def real_plan(tmp_path,monkeypatch,*,edges=((1000,1),(2000,0)),objects=None,
              counts=(1,0,0,0,0,0),header_max=99,mods=0,lazer=False,fps=60,skin_dir=None):
    objects=objects or '64,192,1000,128,0,2000:0:0:0:0:'
    map_path=tmp_path/'fixture.osu'
    map_path.write_text('osu file format v14\n[General]\nMode:3\n[Metadata]\nTitle:combo\nArtist:test\n'
                        'Creator:test\nVersion:truth\n[Difficulty]\nCircleSize:2\nOverallDifficulty:5\n'
                        '[TimingPoints]\n0,500,4,1,0,100,1,0\n[HitObjects]\n'+objects+'\n')
    clock=0;frames=['0|0|18|0']
    for when,mask in edges:
        frames.append(f'{when-clock}|{mask}|18|0');clock=when
    frames.append('-12345|0|0|42')
    data=lzma.compress((','.join(frames)+',').encode(),format=lzma.FORMAT_ALONE)
    g,c,k,h,l,m=counts
    raw=(struct.pack('<Bi',3,20260101)+string_bytes(hashlib.md5(map_path.read_bytes()).hexdigest())
         +string_bytes('player')+string_bytes('replay')+struct.pack('<6H',c,h,l,g,k,m)
         +struct.pack('<iHBi',123456,header_max,0,mods)+string_bytes('')+struct.pack('<qi',0,len(data))
         +data+struct.pack('<q',0))
    replay_path=tmp_path/'fixture.osr';replay_path.write_bytes(raw)
    if lazer:
        replay=parse_replay(replay_path)
        monkeypatch.setattr(render_module,'parse_replay',lambda _:replace(replay,is_lazer_replay=True))
    async def encoder(*_): return 'libx264'
    monkeypatch.setattr(render_module,'probe_encoder',encoder)
    return asyncio.run(render_module.build_render_plan(osr_path=replay_path,beatmap_dir=tmp_path,
        output_path=tmp_path/'unused.mp4',options=RenderOptions((320,240),fps,encoder='libx264'),skin_dir=skin_dir))


def scene(plan,now):
    return render_module.build_frame_state(plan,now,0,100)[0]


@pytest.mark.parametrize('factual_hit',[False,True])
def test_actual_plan_reconciled_promotion_demotion_cannot_alter_raw_combo(tmp_path,monkeypatch,factual_hit):
    counts=(0,0,0,0,0,1) if factual_hit else (1,0,0,0,0,0)
    plan=real_plan(tmp_path,monkeypatch,objects='64,192,1000,1,0,0:0:0:0:',counts=counts,
                   edges=[(1000,1),(1040,0)] if factual_hit else [])
    assert plan.judgment_events[0].judgment==('miss' if factual_hit else 'geki')
    assert scene(plan,1050).combo==int(factual_hit)
    raw=plan.stable_combo_timeline.facts
    plan.judgment_timeline=[]
    del plan._tl_times
    assert scene(plan,1050).combo==int(factual_hit)
    assert plan.stable_combo_timeline.facts is raw


def test_real_ln_heavy_plan_preserves_header_and_score_while_live_combo_counts_holding(tmp_path,monkeypatch,caplog):
    plan=real_plan(tmp_path,monkeypatch)
    assert plan.replay.is_lazer_replay is False
    assert len([j for j in plan.judgment_events if j.judgment!='miss'])==2
    assert plan.stable_combo_timeline.reconstructed_max_combo==10
    assert plan.replay.max_combo==99
    diagnostic=next(record for record in caplog.records if record.msg.startswith('stable_combo_diagnostic'))
    assert (diagnostic.header_max_combo,diagnostic.reconstructed_max_combo)==(99,10)
    before=scene(plan,2050)
    assert before.combo==before.reconstructed_max_combo==10
    assert before.max_combo==99
    assert before.combo_age_ms==50
    assert plan._fs_cache['sd_combo']==1  # scoring combo did not absorb nine hold additions
    assert scene(plan,plan.results_start_ms+1000).max_combo==99
    assert scene(plan,plan.results_start_ms+1000).score==plan.score_final
    raw=plan.stable_combo_timeline.facts
    plan.judgment_timeline=[];del plan._tl_times
    assert scene(plan,2050).combo==10
    assert plan.stable_combo_timeline.facts is raw


def test_real_plan_direct_backward_forward_seeks_and_fps_sampling_are_identical(tmp_path,monkeypatch):
    results=[]
    for fps in (30,60,120):
        plan=real_plan(tmp_path,monkeypatch,fps=fps,edges=[(1000,1),(1250,0),(1320,1),(2000,0)])
        wanted={t:(scene(plan,t).combo,scene(plan,t).combo_age_ms,scene(plan,t).combo_break_previous_value,
                   scene(plan,t).combo_break_age_ms,scene(plan,t).reconstructed_max_combo) for t in (1050,1260,1450,2050)}
        for frame in range(int(2.1*fps)):
            scene(plan,frame*1000/fps)
        for now in (2050,1050,1260,2050,1450):
            actual=scene(plan,now)
            assert (actual.combo,actual.combo_age_ms,actual.combo_break_previous_value,
                    actual.combo_break_age_ms,actual.reconstructed_max_combo)==wanted[now]
        results.append(wanted)
    assert results[0]==results[1]==results[2]


def test_real_plan_stable_score_v2_still_uses_stable_combo(tmp_path,monkeypatch):
    plan=real_plan(tmp_path,monkeypatch,mods=Mod.V2)
    assert not plan.replay.is_lazer_replay
    assert plan.replay.mania_max_weight==plan.replay.mania_acc_weight==305
    assert scene(plan,1950).combo==9


@pytest.mark.parametrize('mods,mask',[(Mod.DT,1),(Mod.HT,1),(Mod.DT|Mod.MR,2)])
def test_real_rate_mod_plan_release_on_due_audio_tick_does_not_round_into_a_hold_addition(tmp_path,monkeypatch,mods,mask):
    plan=real_plan(tmp_path,monkeypatch,mods=mods,edges=[(1001,mask),(1101,0)])
    assert not any(f.source=='ln-hold' for f in plan.stable_combo_timeline.facts)
    release=next(f for f in plan.stable_combo_timeline.facts if f.source=='ln-break')
    assert release.time_ms==pytest.approx(1101/plan.audio_rate)


def test_lazer_plan_uses_head_tail_combo_and_preserves_score_path(tmp_path,monkeypatch):
    plan=real_plan(tmp_path,monkeypatch,lazer=True)
    assert plan.stable_combo_timeline is None
    assert plan.lazer_combo_timeline.reconstructed_max_combo==2
    assert plan.legacy_presentation.stable_combo_facts==()
    assert scene(plan,1050).combo==1
    assert scene(plan,1950).combo==1
    after=scene(plan,2050)
    assert after.combo==2 and after.reconstructed_max_combo==2


@pytest.mark.parametrize('release,maximum,large',[(3050,20,False),(3150,21,True)])
def test_large_break_uses_raw_hold_combo_threshold_and_break_timestamp(tmp_path,monkeypatch,release,maximum,large):
    plan=real_plan(tmp_path,monkeypatch,objects='64,192,1000,128,0,4000:0:0:0:0:',edges=[(1000,1),(release,0)])
    after=scene(plan,release+50)
    assert after.combo==0 and after.combo_break_previous_value==maximum
    assert after.combo_break_age_ms==50
    assert after.miss_break_age_ms==50 if large else after.miss_break_age_ms>300
    assert after.reconstructed_max_combo==maximum


def test_stable_combo_is_gameplay_truth_for_argon_and_authored_legacy_skins(tmp_path,monkeypatch):
    skin=tmp_path/'skin';skin.mkdir();(skin/'skin.ini').write_text('[Mania]\nKeys:2\n')
    argon=real_plan(tmp_path,monkeypatch)
    legacy=real_plan(tmp_path,monkeypatch,skin_dir=skin)
    assert argon.stable_combo_timeline.facts==legacy.stable_combo_timeline.facts
    assert scene(argon,1950).combo==scene(legacy,1950).combo==9


def test_client_combo_selection_changes_no_score_accuracy_or_presentation_fields(tmp_path,monkeypatch):
    stable=real_plan(tmp_path,monkeypatch)
    lazer=real_plan(tmp_path,monkeypatch,lazer=True)
    assert stable.judgment_events==lazer.judgment_events
    for now in (1050,1250,1950,2050,stable.results_start_ms+1000):
        actual=scene(stable,now);original=scene(lazer,now)
        adjusted=replace(actual,combo=original.combo,combo_age_ms=original.combo_age_ms,
                         combo_break_previous_value=original.combo_break_previous_value,
                         combo_break_age_ms=original.combo_break_age_ms,
                         miss_break_age_ms=original.miss_break_age_ms,
                         reconstructed_max_combo=original.reconstructed_max_combo)
        assert asdict(adjusted)==asdict(original)


@pytest.mark.parametrize('edges', [[(1000,1),(2000,0)],[(830,1),(2000,0)],[(1000,1)],
    [(1000,1),(1250,0),(1320,1),(1450,0),(1500,1),(2000,0)],[(1500,1),(2000,0)]])
def test_enabling_combo_does_not_change_round35_presentation_outputs(edges):
    objects=(HoldNote(0,1000,2000),HoldNote(1,1500,2300))
    keys=tuple(KeyEvent(*e) for e in edges)
    before=build_legacy_mania_presentation(objects,keys,2)
    after=build_legacy_mania_presentation(objects,keys,2,include_combo=True)
    assert replace(after,stable_combo_facts=())==before
    for now in (900,1050,1300,1450,1700,2050,2200,1050):
        assert before.long_lights_at(now)==after.long_lights_at(now)


def test_combo_facts_and_prefix_fold_are_immutable():
    timeline=StableComboTimeline.build(facts([HoldNote(0,1000,2000)],[(1000,1),(2000,0)]))
    with pytest.raises(FrozenInstanceError): timeline.facts=()
    with pytest.raises(FrozenInstanceError): timeline.facts[0].time_ms=0
    with pytest.raises(FrozenInstanceError): timeline.at(1500).combo=99
