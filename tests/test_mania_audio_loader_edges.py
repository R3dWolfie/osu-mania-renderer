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
    note=Note(0,100,hit_sound=9,hit_sample=sample or HitSample())
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


def encoded_sample(path):
    """Real OGG/MP3 payloads, large enough to avoid the stable silence sentinel."""
    data=np.full((4410,2),.5,dtype=np.float32)
    if path.suffix=='.ogg':
        sf.write(path,data,44100,format='OGG',subtype='VORBIS')
    else:
        import subprocess
        subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar','44100','-ac','2',
                        '-i','pipe:0','-b:a','128k',str(path)],input=data.tobytes(),check=True,capture_output=True)
    assert path.stat().st_size>=1024


@pytest.mark.parametrize('filename',['custom.missing','custom'])
@pytest.mark.parametrize('bad_ext,good_ext',[('wav','ogg'),('ogg','mp3')])
def test_stable_explicit_stem_selects_existing_bytes_before_decode(sources,filename,bad_ext,good_ext,monkeypatch):
    (sources['map']/f'custom.{bad_ext}').write_bytes(b'invalid sample'*200)
    encoded_sample(sources['map']/f'custom.{good_ext}')
    beatmap=chart(sample_set=1,sample=HitSample(filename=filename))
    cache=hs._SampleCache(44100,beatmap_dir=sources['map'],skin_dirs=(sources['skin'],))
    attempted=[];original=cache.soundfile.info
    def info(path,*a,**k):attempted.append(Path(path));return original(path,*a,**k)
    monkeypatch.setattr(cache.soundfile,'info',info)
    for _ in range(2):
        layers=resolve(sources,beatmap,cache=cache)
        assert [x.path.name for x in layers]==['normal-hitnormal.wav','normal-hitclap.wav']
    assert sources['map']/f'custom.{good_ext}' not in attempted
    assert attempted.count(sources['map']/f'custom.{bad_ext}')==1


def test_conventional_corrupt_wav_can_continue_to_mp3(sources):
    (sources['map']/'normal-hitnormal.wav').write_bytes(b'corrupt'*300)
    encoded_sample(sources['map']/'normal-hitnormal.mp3')
    assert resolve(sources,chart(sample_set=1))[0].path.suffix=='.mp3'


def test_lazer_explicit_resource_store_can_continue_after_corrupt_wav(sources):
    (sources['map']/'custom.wav').write_bytes(b'corrupt'*300)
    encoded_sample(sources['map']/'custom.mp3')
    assert resolve(sources,chart(sample=HitSample(filename='custom.missing')),lazer=True)[0].path.suffix=='.mp3'


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('tier',['map','skin'])
def test_authored_zero_pcm_is_valid_and_suppresses_fallback(sources,lazer,tier):
    name=sources[tier]/'normal-hitnormal.wav';wave(name,0)
    beatmap=chart(sample_set=1,index=1 if tier=='map' else 0)
    beatmap=replace(beatmap,notes=(replace(beatmap.notes[0],hit_sound=0),))
    layer=resolve(sources,beatmap,lazer=lazer)[0]
    assert layer.path==name and not np.any(layer.samples)
    path=track(sources,beatmap,lazer=lazer)
    data,_=sf.read(path,always_2d=True)
    assert not np.any(data)


def test_no_valid_gameplay_sources_still_raise(sources):
    from osu_mania_renderer_v2.errors import RendererError
    for p in sources['default'].iterdir():p.unlink()
    with pytest.raises(RendererError):track(sources,chart(sample_set=1))
    assert not (sources['map']/'out.wav').exists()


@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('byte_count',[0,1023])
def test_stable_tiny_beatmap_file_resolves_as_valid_silence(sources,explicit,byte_count):
    name='tiny.wav' if explicit else 'normal-hitnormal.wav'
    (sources['map']/name).write_bytes(b'x'*byte_count)
    wave(sources['skin']/'normal-hitnormal.wav',.75)
    beatmap=chart(sample_set=1,sample=HitSample(filename=name) if explicit else None)
    beatmap=replace(beatmap,notes=(replace(beatmap.notes[0],hit_sound=0),))
    layer=resolve(sources,beatmap)[0]
    assert layer.path==sources['map']/name and not np.any(layer.samples)
    data,_=sf.read(track(sources,beatmap))
    assert not np.any(data)


@pytest.mark.parametrize('lazer',[False,True])
def test_tiny_beatmap_combobreak_suppresses_fallback_only_in_stable(sources,lazer):
    (sources['map']/'combobreak.wav').write_bytes(b'x'*32)
    wave(sources['skin']/'combobreak.wav',.5)
    facts=tuple(SimpleNamespace(time_ms=i,kind='increment') for i in range(21))+(SimpleNamespace(time_ms=100,kind='reset'),)
    output=hs.build_hitsound_track(beatmap=chart(),beatmap_dir=sources['map'],skin_dirs=(sources['skin'],),
        output_wav=sources['map']/'combo.wav',duration_ms=500,stable_sound_facts=(),lazer_facts=(),
        combo_facts=facts,is_lazer_replay=lazer)
    data,_=sf.read(output)
    assert np.max(data)==(.5 if lazer else 0)


@pytest.mark.parametrize('size',[1024,1025])
def test_large_corrupt_beatmap_file_is_invalid_not_silence(sources,size):
    (sources['map']/'normal-hitnormal.wav').write_bytes(b'x'*size)
    assert resolve(sources,chart(sample_set=1))[0].path.parent==sources['default']


@pytest.mark.parametrize('lazer,tier,expected_silent',[(False,'map',True),(True,'map',False),(False,'skin',False)])
def test_tiny_valid_wave_obeys_beatmap_only_stable_rule(sources,lazer,tier,expected_silent):
    wave(sources[tier]/'normal-hitnormal.wav',.75,frames=8)
    assert (sources[tier]/'normal-hitnormal.wav').stat().st_size<1024
    layers=resolve(sources,chart(sample_set=1,index=1 if tier=='map' else 0),lazer=lazer)
    assert layers[0].path.parent==sources[tier]
    assert bool(np.any(layers[0].samples)) is not expected_silent


def test_corrupt_candidate_is_decoded_once_for_ten_thousand_resolutions(sources,monkeypatch,caplog):
    bad=sources['map']/'normal-hitnormal.wav';bad.write_bytes(b'corrupt'*300)
    beatmap=chart(sample_set=1)
    cache=hs._SampleCache(44100,beatmap_dir=sources['map'],skin_dirs=(sources['skin'],))
    attempts=0;original=cache.soundfile.info
    def info(path,*a,**k):
        nonlocal attempts
        if Path(path)==bad:attempts+=1
        return original(path,*a,**k)
    monkeypatch.setattr(cache.soundfile,'info',info)
    for _ in range(10000):
        layers=resolve(sources,beatmap,cache=cache)
        assert layers[0].path==sources['default']/'normal-hitnormal.wav'
    assert attempts==1
    assert sum(r.msg=='hitsound_load_failed' for r in caplog.records)==1


def test_stable_dot_in_directory_counts_as_extension_for_exact_file(sources):
    directory=sources['map']/'folder.withdot';directory.mkdir()
    exact=directory/'custom';sf.write(exact,np.ones((441,2))*.5,44100,format='WAV')
    layer=resolve(sources,chart(sample=HitSample(filename='folder.withdot/custom')))[0]
    assert layer.path==exact


def test_stable_leading_dot_is_not_an_extension(sources):
    wave(sources['map']/'.custom.wav',.25)
    wave(sources['map']/'.custom.wav.wav',.75)
    layer=resolve(sources,chart(sample=HitSample(filename='.custom.wav')))[0]
    assert layer.path.name=='.custom.wav.wav'
