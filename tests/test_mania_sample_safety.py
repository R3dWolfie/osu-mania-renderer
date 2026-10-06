"""Confined legacy audio paths and bounded sample decoding."""
import pytest
from osu_mania_renderer_v2.beatmap.models import HitSample
from tests.test_mania_sample_resolution import banks, resolve, wave


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('filename,stored',[
    ('normal-hitnormal.wav','Normal-HitNormal.WAV'),
    ('normal-hitnormal.mp3','normal-hitnormal.MP3'),
    (r'fx\Click.WAV','Fx/cLiCk.wav'),
])
def test_case_insensitive_and_backslash_paths(banks,lazer,filename,stored):
    target=banks['map']/stored;target.parent.mkdir(exist_ok=True)
    import numpy as np
    import soundfile as sf
    sf.write(target,np.ones((441,2))*.2,44100,format='WAV')
    layer=resolve(banks,1,lazer=lazer,sample=HitSample(filename=filename))[0]
    assert layer.path==target


@pytest.mark.parametrize('lazer',[False,True])
def test_case_insensitive_conventional_skin(banks,lazer):
    target=banks['skin']/'Normal-HitNormal.WAV';wave(target)
    assert resolve(banks,0,lazer=lazer)[0].path==target


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('form',['dotdot','absolute','backslash','drive','unc','symlink'])
def test_external_sample_paths_are_confined(banks,lazer,form):
    outside=banks['map'].parent/'outside.wav';wave(outside)
    filename={'dotdot':'../outside.wav','absolute':str(outside),'backslash':r'..\outside.wav',
              'drive':r'C:\outside.wav','unc':r'\\server\share\outside.wav','symlink':'escape.wav'}[form]
    if form=='symlink':(banks['map']/filename).symlink_to(outside)
    layers=resolve(banks,1,lazer=lazer,sample=HitSample(filename=filename))
    assert layers and all(layer.path.parent==banks['default'] for layer in layers)


def test_conventional_symlink_escape_is_rejected(banks):
    outside=banks['map'].parent/'outside.wav';wave(outside)
    (banks['map']/'normal-hitnormal.wav').symlink_to(outside)
    assert resolve(banks,1)[0].path.parent==banks['default']


def test_case_insensitive_overlay_paths_and_in_root_symlink(banks):
    from osu_mania_renderer_v2.render import hitsounds as hs
    target=banks['skin']/'ComboBreak.WAV';wave(target)
    assert hs._find_combobreak_sample(None,(banks['skin'],)).path==target
    wave(banks['skin']/'Nightcore-Clap.WAV')
    cache=hs._SampleCache(44100,skin_dirs=(banks['skin'],))
    assert hs._find_skin_sample(('nightcore-clap.wav',),(banks['skin'],),cache) is not None
    target=banks['map']/'local.wav';wave(target)
    (banks['map']/'safe.wav').symlink_to(target)
    assert resolve(banks,1,sample=HitSample(filename='safe.wav'))[0].path==target


@pytest.mark.parametrize('frames,rate,channels',[
    (10**10,44100,2),(10**1000,44100,2),(44100,1,2),(44100,44100,100000),(float('inf'),44100,2),
    (-1,44100,2),(100,0,2),(100,44100,0),
])
def test_oversized_or_invalid_metadata_rejected_before_decode(banks,monkeypatch,caplog,frames,rate,channels):
    from types import SimpleNamespace
    from osu_mania_renderer_v2.render import hitsounds as hs
    path=banks['map']/'huge.wav';wave(path)
    cache=hs._SampleCache(44100,beatmap_dir=banks['map'])
    monkeypatch.setattr(cache.soundfile,'info',lambda *_:SimpleNamespace(frames=frames,samplerate=rate,channels=channels))
    def forbidden(*a,**k):raise AssertionError('full decode attempted before metadata bound')
    monkeypatch.setattr(cache.soundfile,'read',forbidden)
    monkeypatch.setattr(cache.soundfile,'SoundFile',forbidden)
    caplog.set_level('WARNING')
    assert cache.get(path) is None
    assert any(r.msg=='hitsound_sample_rejected' for r in caplog.records)


def test_oversized_candidate_falls_back_to_skin_without_read(banks,monkeypatch):
    from types import SimpleNamespace
    from osu_mania_renderer_v2.render import hitsounds as hs
    path=banks['map']/'normal-hitnormal2.wav';wave(path)
    wave(banks['skin']/'normal-hitnormal.wav')
    original_info=hs.require_hitsound_runtime().info
    original_read=hs.require_hitsound_runtime().read
    def info(file,*a,**k):
        if str(file)==str(path):return SimpleNamespace(frames=10**10,samplerate=44100,channels=2)
        return original_info(file,*a,**k)
    def read(file,*a,**k):
        assert str(file)!=str(path),'oversized candidate decoded'
        return original_read(file,*a,**k)
    monkeypatch.setattr(hs.require_hitsound_runtime(),'info',info)
    monkeypatch.setattr(hs.require_hitsound_runtime(),'read',read)
    assert resolve(banks,2)[0].path==banks['skin']/'normal-hitnormal.wav'


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('root_name',['map','skin'])
def test_explicit_absolute_path_is_rejected_even_inside_an_allowed_root(banks,lazer,root_name):
    path=banks[root_name]/'private.wav';wave(path)
    layers=resolve(banks,1,lazer=lazer,sample=HitSample(filename=str(path)))
    assert layers and all(layer.path.parent==banks['default'] for layer in layers)


def test_opened_metadata_is_rechecked_before_read(banks,monkeypatch):
    from osu_mania_renderer_v2.render import hitsounds as hs
    path=banks['map']/'changing.wav';wave(path)
    cache=hs._SampleCache(44100,beatmap_dir=banks['map'])
    original_info=cache.soundfile.info(path)
    monkeypatch.setattr(cache.soundfile,'info',lambda *_:original_info)
    class ChangedReader:
        frames=10**10
        samplerate=44100
        channels=2
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self,**_):raise AssertionError('changed metadata reached decode')
    monkeypatch.setattr(cache.soundfile,'SoundFile',lambda *_:ChangedReader())
    assert cache.get(path) is None


def test_general_sample_set_cannot_supply_a_path_into_another_tier(banks):
    from dataclasses import replace
    from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
    from osu_mania_renderer_v2.beatmap.models import Note, TimingPoint
    from osu_mania_renderer_v2.render import hitsounds as hs
    from tests.test_mania_hitsound_pipeline import MAP
    private=banks['skin']/'private-hitnormal.wav';wave(private)
    fallback=banks['default']/'soft-hitnormal.wav';wave(fallback)
    beatmap=replace(parse_beatmap(MAP),default_sample_set=str(banks['skin']/'private'),
                    timing_points=(TimingPoint(0,0,0,100),))
    cache=hs._SampleCache(44100,beatmap_dir=banks['map'],skin_dirs=(banks['skin'],))
    layers=hs._resolve_samples_for_note(Note(0,100),beatmap,cache)
    assert layers and layers[0].path==fallback
