"""Source-tier legacy sample resolution; no user media required."""
from dataclasses import replace
from pathlib import Path

import pytest

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import HitSample, Note, TimingPoint
from osu_mania_renderer_v2.render import hitsounds as hs
from tests.test_mania_hitsound_pipeline import MAP, wave


@pytest.fixture
def banks(tmp_path,monkeypatch):
    roots={name:tmp_path/name for name in ('map','skin','default')}
    for p in roots.values():p.mkdir()
    for layer in ('normal','whistle','finish','clap'):
        wave(roots['default']/f'normal-hit{layer}.wav')
    monkeypatch.setattr(hs,'_DEFAULT_HITSOUND_DIR',roots['default'])
    return roots


def resolve(banks,index,*,lazer=False,beatmap_on=True,skin_on=True,sample=None,bits=0):
    beatmap=replace(parse_beatmap(MAP),timing_points=(TimingPoint(0,1,index,100),))
    cache=hs._SampleCache(44100,beatmap_dir=banks['map'],skin_dirs=(banks['skin'],) if skin_on else (),
                          beatmap_hitsounds=beatmap_on)
    return hs._resolve_samples_for_note(Note(0,100,hit_sound=bits,hit_sample=sample or HitSample()),
                                        beatmap,cache,is_lazer_replay=lazer)


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('index,numbered,base,skin_on,beatmap_on,expected',[
    (0,False,True,True,True,'skin/normal-hitnormal.wav'),
    (0,False,True,False,True,'default/normal-hitnormal.wav'),
    (1,True,True,True,True,'map/normal-hitnormal.wav'),
    (2,True,True,True,True,'map/normal-hitnormal2.wav'),
    (2,False,True,True,True,'skin/normal-hitnormal.wav'),
    (2,False,False,True,True,'skin/normal-hitnormal.wav'),
    (2,True,True,True,False,'skin/normal-hitnormal.wav'),
    (2,False,True,False,True,'default/normal-hitnormal.wav'),
])
def test_source_bank_tiers(banks,lazer,index,numbered,base,skin_on,beatmap_on,expected):
    if numbered: wave(banks['map']/f'normal-hitnormal{index}.wav')
    if base: wave(banks['map']/'normal-hitnormal.wav')
    wave(banks['skin']/'normal-hitnormal.wav');wave(banks['skin']/f'normal-hitnormal{index}.wav')
    layers=resolve(banks,index,lazer=lazer,skin_on=skin_on,beatmap_on=beatmap_on)
    assert layers[0].path==banks['map'].parent/expected


@pytest.mark.parametrize('lazer',[False,True])
def test_addition_set_has_same_source_tier_rules(banks,lazer):
    for root in ('map','skin'):
        for suffix in ('','2'):
            wave(banks[root]/f'soft-hitclap{suffix}.wav')
    sample=HitSample(addition_set=2)
    layer,=resolve(banks,2,lazer=lazer,sample=sample,bits=8)
    assert layer.path==banks['map']/'soft-hitclap2.wav'
    (banks['map']/'soft-hitclap2.wav').unlink()
    layer,=resolve(banks,2,lazer=lazer,sample=sample,bits=8)
    assert layer.path==banks['skin']/'soft-hitclap.wav'


@pytest.mark.parametrize('tier,index',[('map',1),('skin',0)])
@pytest.mark.parametrize('extensions,expected',[(['mp3'],'mp3'),(['wav','mp3','ogg'],'wav'),(['mp3','ogg'],'mp3')])
def test_stable_conventional_extensions(banks,tier,index,extensions,expected):
    for ext in extensions:
        # WAV payload under alternate extension exercises identity ordering;
        # actual MP3 decoding has a separate encoded-media smoke below.
        import soundfile as sf
        import numpy as np
        sf.write(banks[tier]/f'normal-hitnormal.{ext}',np.ones((441,2))*.2,44100,format='WAV')
    assert resolve(banks,index)[0].path==banks[tier]/f'normal-hitnormal.{expected}'


@pytest.mark.parametrize('lazer',[False,True])
def test_real_mp3_only_conventional_sample(banks,lazer):
    import subprocess,shutil
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg unavailable')
    source=banks['map']/'input.wav';wave(source)
    target=banks['map']/'normal-hitnormal.mp3'
    subprocess.run(['ffmpeg','-v','error','-i',str(source),str(target)],check=True,capture_output=True)
    assert resolve(banks,1,lazer=lazer)[0].path==target


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('filename,available,expected',[
    ('custom.wav',['wav','mp3','ogg'],'wav'),
    ('custom.missing',['wav','mp3','ogg'],'wav'),
    ('custom.missing',['ogg'],'ogg'),('custom.missing',['mp3'],'mp3'),
    ('custom',['wav'],'wav'),
])
def test_explicit_filename_exact_or_stem_fallback(banks,lazer,filename,available,expected):
    import numpy as np
    import soundfile as sf
    for ext in available:sf.write(banks['map']/f'custom.{ext}',np.ones((441,2))*.2,44100,format='WAV')
    layer=resolve(banks,1,lazer=lazer,sample=HitSample(filename=filename))[0]
    assert layer.path==banks['map']/f'custom.{expected}'


def test_stable_explicit_stem_prefers_ogg_to_mp3(banks):
    import numpy as np
    import soundfile as sf
    for ext in ('ogg','mp3'):sf.write(banks['map']/f'custom.{ext}',np.ones((441,2))*.2,44100,format='WAV')
    assert resolve(banks,1,sample=HitSample(filename='custom.missing'))[0].path==banks['map']/'custom.ogg'


def test_lazer_explicit_stem_uses_resource_store_mp3_before_ogg(banks):
    import numpy as np
    import soundfile as sf
    for ext in ('ogg','mp3'):sf.write(banks['map']/f'custom.{ext}',np.ones((441,2))*.2,44100,format='WAV')
    assert resolve(banks,1,lazer=True,sample=HitSample(filename='custom.missing'))[0].path==banks['map']/'custom.mp3'


def test_stable_complete_direct_miss_keeps_normal_and_additions(banks):
    layers=resolve(banks,2,sample=HitSample(filename='missing.wav'),bits=15)
    assert [s.type_name for s in layers]==['normal','whistle','finish','clap']
    assert all(s.path.parent==banks['default'] for s in layers)


@pytest.mark.parametrize('lazer',[False,True])
def test_beatmap_off_disables_explicit_file(banks,lazer):
    wave(banks['map']/'custom.wav')
    assert resolve(banks,1,beatmap_on=False,lazer=lazer,sample=HitSample(filename='custom.wav'))[0].path.parent==banks['default']
