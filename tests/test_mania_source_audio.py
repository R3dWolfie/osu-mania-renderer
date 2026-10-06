"""Stable audio calls, source combo breaks, and production source selection."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from osu_mania_renderer_v2.beatmap.models import HitSample, HoldNote, KeyEvent, Note, RenderOptions, TimingPoint
from osu_mania_renderer_v2.render import hitsounds as hs
from osu_mania_renderer_v2.render import render as rendering
from osu_mania_renderer_v2.render.legacy_mania_events import build_legacy_mania_presentation
from tests.test_mania_hitsound_pipeline import MAP, REPLAY, SONG, oracle, wave


def combo_events(streaks):
    result=[];now=0
    for streak in streaks:
        for _ in range(streak):
            now+=10;result.append(SimpleNamespace(time_ms=now,kind='increment'))
        now+=10;result.append(SimpleNamespace(time_ms=now,kind='reset'))
    return result


@pytest.mark.parametrize('lazer,streaks,expected',[
    (False,[20,21],[430]),(True,[1,20,21],[20,450]),
    (True,[0,2,0,0,21],[40,280]),(False,[0,20,0],[]),
])
def test_source_combo_break_boundaries_first_break_and_duplicate_zero_resets(lazer,streaks,expected):
    actual=list(hs._combo_break_times(combo_events(streaks),is_lazer_replay=lazer))
    assert actual==expected


@pytest.mark.parametrize('mods',[1<<7,1<<13,(1<<7)|(1<<13)])
def test_stable_relax_and_autopilot_suppress_break_sound(mods):
    assert list(hs._combo_break_times(combo_events([21]),is_lazer_replay=False,mods=mods))==[]


def test_combo_threshold_is_strict_user_override():
    assert list(hs._combo_break_times(combo_events([3,4]),is_lazer_replay=False,threshold=3))==[90]
    assert list(hs._combo_break_times(combo_events([1,3,4]),is_lazer_replay=True,threshold=3))==[20,110]


@pytest.mark.parametrize('sound,miss,expected',[(True,True,True),(False,True,False),(True,False,False),(False,False,False)])
def test_both_combo_controls_must_be_enabled(tmp_path,oracle,sound,miss,expected):
    _,beatmap,_=oracle
    wave(tmp_path/'combobreak.wav',.25)
    output=hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,output_wav=tmp_path/'out.wav',
        duration_ms=1000,is_lazer_replay=True,lazer_facts=(),combo_facts=combo_events([1,0,0]),
        combo_break_sound=sound,miss_hitsound=miss)
    data,rate=sf.read(output)
    assert bool(np.any(data))==expected
    if expected:
        assert np.max(data)==.25  # source SFX volume, no arbitrary .26 attenuation
        assert not np.any(data[:int(.02*rate)])
        assert not np.any(data[int(.04*rate):])


def test_stable_audio_uses_hitstart_repress_and_pressed_tap_even_consuming_miss():
    notes=(HoldNote(0,1000,1800),Note(1,2000))
    keys=(KeyEvent(1000,1),KeyEvent(1200,0),KeyEvent(1500,1),KeyEvent(1800,0),
          KeyEvent(1840,2),KeyEvent(1900,0))
    result=build_legacy_mania_presentation(notes,keys,4,od=8,include_combo=True,include_audio=True)
    assert [(f.time_ms,f.source,f.object_time_ms) for f in result.sound_facts]==[
        (1000,'head',1000),(1500,'head',1000),(1840,'tap',2000)]
    assert any(f.kind=='reset' and f.source=='miss' and f.time_ms==1840 for f in result.stable_combo_facts)
    assert not any(f.source=='final' for f in result.sound_facts)  # native Hold.SoundAtEnd=false


def test_converted_stable_long_note_has_final_sound_and_end_control_point(tmp_path,oracle):
    _,beatmap,_=oracle
    wave(tmp_path/'normal-hitnormal.wav')
    hold=HoldNote(0,1000,1800)
    presentation=build_legacy_mania_presentation((hold,), (KeyEvent(1000,1),KeyEvent(1800,0)),
        4,od=8,source_mode=0,include_combo=True,include_audio=True)
    assert [(f.time_ms,f.source) for f in presentation.sound_facts]==[(1000,'head'),(1800,'final')]
    beatmap=replace(beatmap,notes=(hold,),source_mode=0,timing_points=(
        TimingPoint(0,1,1,10),TimingPoint(1802,1,1,100)))
    output=hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,output_wav=tmp_path/'out.wav',
        duration_ms=2200,stable_sound_facts=presentation.sound_facts,combo_facts=presentation.stable_combo_facts)
    data,rate=sf.read(output)
    assert np.max(data[int(1*rate):int(1.1*rate)])==pytest.approx(.04,abs=1/32768)
    assert np.max(data[int(1.8*rate):int(1.9*rate)])==pytest.approx(.4,abs=1/32768)


@pytest.mark.parametrize('beatmap_on,skin_on',[(True,False),(True,True),(False,True),(False,False)])
def test_actual_plan_sample_path_source_combinations(tmp_path,oracle,monkeypatch,beatmap_on,skin_on):
    _,beatmap,_=oracle
    map_dir=tmp_path/'map';map_dir.mkdir();(map_dir/MAP.name).write_bytes(MAP.read_bytes())
    skin=tmp_path/'skin';skin.mkdir()
    wave(map_dir/'normal-hitnormal.wav',.5);wave(skin/'normal-hitnormal.wav',.25)
    async def encoder(*args): return 'libx264'
    monkeypatch.setattr(rendering,'probe_encoder',encoder)
    monkeypatch.setattr(rendering,'parse_beatmap',lambda *args,**kwargs:replace(beatmap,audio_filename=str(SONG),timing_points=tuple(replace(tp,custom_index=1) for tp in beatmap.timing_points)))
    received={}
    def capture(**kwargs):
        cache=hs._SampleCache(44100,beatmap_dir=kwargs['beatmap_dir'],skin_dirs=kwargs['skin_dirs'],
                              beatmap_hitsounds=kwargs['beatmap_hitsounds'])
        layers=hs._resolve_samples_for_note(kwargs['sample_notes'][0],kwargs['beatmap'],cache,is_lazer_replay=True)
        received['path']=layers[0].path
        received['options']=kwargs
        return kwargs['output_wav']
    monkeypatch.setattr(rendering,'build_hitsound_track',capture)
    asyncio.run(rendering.build_render_plan(osr_path=REPLAY,beatmap_dir=map_dir,skin_dir=skin,
        output_path=tmp_path/'out.mp4',options=RenderOptions((16,16),10,encoder='libx264',normalize_loudness=False,
            beatmap_hitsounds=beatmap_on,use_skin_hitsounds=skin_on,combo_break_sound=False,combo_break_threshold=7)))
    expected=(map_dir if beatmap_on else skin if skin_on else hs._DEFAULT_HITSOUND_DIR)/'normal-hitnormal.wav'
    assert received['path']==expected
    assert received['options']['combo_break_sound'] is False
    assert received['options']['combo_break_threshold']==7


def test_bundled_combo_break_is_available_without_host_skin():
    path=hs._find_combobreak_sample(None,()).path
    assert path==hs._DEFAULT_HITSOUND_DIR/'combobreak.mp3'
    assert np.max(np.abs(sf.read(path)[0]))>0
