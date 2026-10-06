"""Final loader parity audit against stable and the pinned lazer decoder."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import HitSample, Note, TimingPoint
from osu_mania_renderer_v2.render import hitsounds as hs
from tests.test_mania_hitsound_pipeline import MAP, wave


@pytest.fixture
def sources(tmp_path, monkeypatch):
    roots={name:tmp_path/name for name in ('map','skin','default')}
    for root in roots.values():root.mkdir()
    for bank in ('normal','soft','drum'):
        for layer in ('normal','clap'):
            wave(roots['default']/f'{bank}-hit{layer}.wav',.25)
    monkeypatch.setattr(hs,'_DEFAULT_HITSOUND_DIR',roots['default'])
    return roots


def chart(*,sample_set=0,index=1,general='Drum',sample=None):
    note=Note(0,100,hit_sound=8,hit_sample=sample or HitSample())
    return replace(parse_beatmap(MAP),notes=(note,),default_sample_set=general,
                   timing_points=(TimingPoint(0,sample_set,index,100),))


def resolve(sources,beatmap,*,lazer=False,cache=None):
    cache=cache or hs._SampleCache(44100,beatmap_dir=sources['map'],skin_dirs=(sources['skin'],))
    return hs._resolve_samples_for_note(beatmap.notes[0],beatmap,cache,is_lazer_replay=lazer)


def track(sources,beatmap,*,lazer=False,**kwargs):
    return hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=sources['map'],skin_dirs=(sources['skin'],),
        output_wav=sources['map']/'out.wav',duration_ms=500,is_lazer_replay=lazer,
        stable_sound_facts=(SimpleNamespace(column=0,object_time_ms=100,time_ms=100,source='tap'),),
        lazer_facts=(SimpleNamespace(column=0,object_time_ms=100,time_ms=100,source='tap',kind='increment'),),
        combo_break_sound=False,**kwargs)


@pytest.mark.parametrize('lazer,expected',[(False,'soft'),(True,'normal')])
@pytest.mark.parametrize('index',[0,1,2])
def test_timing_zero_is_client_specific_and_addition_inherits(sources,lazer,expected,index):
    for bank in ('normal','soft','drum'):
        for suffix in ('','2'):
            for layer in ('normal','clap'):
                wave(sources['map']/f'{bank}-hit{layer}{suffix}.wav')
    beatmap=chart(index=index)
    layers=resolve(sources,beatmap,lazer=lazer)
    suffix='2' if index==2 else ''
    root=sources['default'] if index==0 else sources['map']
    assert [x.path for x in layers]==[root/f'{expected}-hit{kind}{suffix}.wav' for kind in ('normal','clap')]
    assert beatmap.timing_points[0].sample_set==0  # raw metadata remains intact


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('object_set,expected',[(1,'normal'),(2,'soft'),(3,'drum')])
def test_object_bank_overrides_zero_timing_bank(sources,lazer,object_set,expected):
    layers=resolve(sources,chart(sample=HitSample(normal_set=object_set)),lazer=lazer)
    assert [x.path.name for x in layers]==[f'{expected}-hitnormal.wav',f'{expected}-hitclap.wav']


def test_no_timing_point_has_no_stable_playsound_and_lazer_uses_default_normal(sources):
    beatmap=replace(chart(general='Drum'),timing_points=())
    assert resolve(sources,beatmap)==[]  # stable PlaySound only calls through when CP != null
    assert resolve(sources,beatmap,lazer=True)[0].path.name=='normal-hitnormal.wav'
