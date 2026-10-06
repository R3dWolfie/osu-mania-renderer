"""HitObjectManagerMania.UpdateHitObjects post-object cached PlaySound calls."""
from dataclasses import replace

import numpy as np
import soundfile as sf

from osu_mania_renderer_v2.beatmap.models import HoldNote, Note, KeyEvent, TimingPoint, HitSample
from osu_mania_renderer_v2.render.legacy_mania_events import build_legacy_mania_presentation
from osu_mania_renderer_v2.render import hitsounds as hs
from tests.test_mania_hitsound_pipeline import oracle, wave, plan_setup, plan, SONG
from osu_mania_renderer_v2.beatmap.models import RenderOptions
from osu_mania_renderer_v2.render import render as rendering


def simulate(notes, edges, **kwargs):
    return build_legacy_mania_presentation(tuple(notes), tuple(KeyEvent(t,m) for t,m in edges),
        4, od=8, include_combo=True, include_audio=True, **kwargs)


def sounds(result):
    return [(f.time_ms, f.object_time_ms, f.source) for f in result.sound_facts]


def test_empty_future_tap_is_audio_only_and_timeout_does_not_clear_cache():
    notes=(Note(0,1000),Note(0,3000))
    edges=((100,1),(150,0),(300,1),(350,0),(1600,1),(1650,0))
    result=simulate(notes,edges)
    assert sounds(result)==[(100,1000,'empty-tap'),(300,1000,'empty-tap'),(1600,1000,'empty-tap')]
    without=build_legacy_mania_presentation(notes,tuple(KeyEvent(t,m) for t,m in edges),4,od=8,include_combo=True)
    assert replace(result,sound_facts=())==without
    assert all(f.kind=='reset' for f in result.stable_combo_facts)
    assert not result.normal_hit_facts and not result.sliding_intervals


def test_actual_tap_clears_cached_future_sound():
    result=simulate((Note(0,1000),Note(0,3000)),((100,1),(150,0),(1000,1),(1050,0),(1500,1)))
    assert sounds(result)==[(100,1000,'empty-tap'),(1000,1000,'tap'),(1500,3000,'empty-tap')]


def test_native_hold_empty_dispatch_is_end_until_soundstart_then_cache_clears():
    result=simulate((HoldNote(0,1000,1600),Note(0,3000)),
        ((100,1),(150,0),(1000,1),(1600,0),(1800,1)))
    assert sounds(result)==[(100,1000,'empty-final'),(1000,1000,'head'),(1800,3000,'empty-tap')]
    assert not any(f.source=='final' for f in result.sound_facts)


def test_unstarted_native_hold_cached_after_timeout_keeps_virtual_end_sound():
    result=simulate((HoldNote(0,1000,1600),Note(0,3000)),((100,1),(150,0),(1900,1)))
    assert sounds(result)==[(100,1000,'empty-final'),(1900,1000,'empty-final')]


def test_empty_lookup_is_physical_column_and_strictly_future():
    result=simulate((Note(1,500),Note(0,1000)),((100,1),(150,0),(200,2)))
    assert [(f.column,f.object_time_ms) for f in result.sound_facts]==[(0,1000),(1,500)]


def test_empty_sound_uses_future_object_metadata_and_end_control_point(tmp_path,oracle):
    _,beatmap,_=oracle
    wave(tmp_path/'head.wav',.5);wave(tmp_path/'tail.wav',.25)
    hold=HoldNote(0,1000,1800,hit_sample=HitSample(filename='head.wav'),tail_hit_sample=HitSample(filename='tail.wav'))
    result=simulate((hold,),((100,1),(150,0)))
    beatmap=replace(beatmap,notes=(hold,),timing_points=(TimingPoint(0,1,1,10),TimingPoint(1802,1,1,100)))
    path=hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,output_wav=tmp_path/'out.wav',
        duration_ms=300,stable_sound_facts=result.sound_facts,miss_hitsound=False)
    data,sr=sf.read(path)
    assert not np.any(data[:int(.1*sr)])
    assert np.max(data)==.25  # end sample/CP, emitted at the early empty press


def test_stable_plan_header_reconciliation_cannot_remove_empty_sound(tmp_path,monkeypatch,oracle):
    replay=plan_setup(monkeypatch,oracle)
    _,beatmap,_=oracle
    replay=replace(replay,is_lazer_replay=False,key_events=(KeyEvent(100,1),KeyEvent(200,0)),
        count_geki=0,count_300=0,count_100=0,count_50=0,count_katu=0,count_miss=1,max_combo=0)
    monkeypatch.setattr(rendering,'parse_replay',lambda _:replay)
    monkeypatch.setattr(rendering,'parse_beatmap',lambda *a,**k:replace(beatmap,notes=(Note(0,1000),),audio_filename=str(SONG)))
    received=[]
    def capture(**kw): received.append(kw['stable_sound_facts']);return kw['output_wav']
    monkeypatch.setattr(rendering,'build_hitsound_track',capture)
    opts=RenderOptions((16,16),10,encoder='libx264',normalize_loudness=False)
    plan(tmp_path,opts)
    replay=replace(replay,count_geki=1,count_miss=0,max_combo=999)
    plan(tmp_path,opts)
    assert received[0]==received[1]
    assert [(f.time_ms,f.source) for f in received[0]]==[(100,'empty-tap')]
