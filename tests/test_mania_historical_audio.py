"""Narrow stable old-format audio metadata and source-specific fallback."""
from dataclasses import replace

import pytest

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from tests.test_mania_audio_loader_edges import sources, resolve
from tests.test_mania_hitsound_pipeline import wave


def historical(sources,version,*,row='0,500',general='Drum',extra='',sample_index=0,filename='',mode=3):
    path=sources['map']/'old.osu'
    sample='' if general is None else f'SampleSet:{general}\n'
    path.write_text(f'osu file format v{version}\n[General]\nMode:{mode}\n'+sample+extra+
        '\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[TimingPoints]\n'+row+
        f'\n[HitObjects]\n64,192,100,1,0,0:0:{sample_index}:0:{filename}\n')
    return parse_beatmap(path,allow_converted=mode!=3)


@pytest.mark.parametrize('version',[3,4,5,14])
@pytest.mark.parametrize('lazer',[False,True])
def test_two_field_row_carries_version_and_client_custom_bank(sources,version,lazer):
    beatmap=historical(sources,version)
    wave(sources['map']/'drum-hitnormal.wav')
    assert beatmap.format_version==version
    assert len(beatmap.timing_points)==1
    tp=beatmap.timing_points[0]
    assert tp.field_count==2 and tp.time_ms==0 and tp.uninherited and tp.beat_length_ms==500
    expected=sources['map'] if version<4 and not lazer else sources['default']
    assert resolve(sources,beatmap,lazer=lazer)[0].path==expected/'drum-hitnormal.wav'


@pytest.mark.parametrize('version',[3,4,5,14])
@pytest.mark.parametrize('lazer',[False,True])
def test_pre_v5_unprefixed_name_only_for_stable_eligible_beatmap_bank(sources,version,lazer):
    wave(sources['map']/'hitnormal.wav')
    beatmap=historical(sources,version,sample_index=2)
    expected=sources['map']/'hitnormal.wav' if version<5 and not lazer else sources['default']/'drum-hitnormal.wav'
    assert resolve(sources,beatmap,lazer=lazer)[0].path==expected


@pytest.mark.parametrize('version',[3,4,5,14])
def test_old_explicit_search_tries_original_name_plus_extension(sources,version):
    wave(sources['map']/'custom.missing.wav')
    beatmap=historical(sources,version,filename='custom.missing')
    expected=sources['map']/'custom.missing.wav' if version<5 else sources['default']/'drum-hitnormal.wav'
    assert resolve(sources,beatmap)[0].path==expected


@pytest.mark.parametrize('version,override,expected',[(3,'0','default'),(4,'1','map'),(14,'1','map')])
def test_stable_general_custom_samples_overrides_header_default(sources,version,override,expected):
    wave(sources['map']/'drum-hitnormal.wav')
    beatmap=historical(sources,version,extra=f'CustomSamples:{override}\n')
    assert resolve(sources,beatmap)[0].path.parent==sources[expected]
    assert resolve(sources,beatmap,lazer=True)[0].path.parent==sources['default']


@pytest.mark.parametrize('row,stable_gain,lazer_gain',[('0,500',.8,.25),('0,500,4,3',.2,.25)])
def test_missing_volume_is_source_specific(sources,row,stable_gain,lazer_gain):
    beatmap=historical(sources,3,row=row,extra='SampleVolume:25\n')
    assert resolve(sources,beatmap)[0].gain==stable_gain
    assert resolve(sources,beatmap,lazer=True)[0].gain==lazer_gain
    if row.count(',')==3:
        assert resolve(sources,beatmap)[0].path.parent==sources['default']  # no v3 default bank for 4-field row


@pytest.mark.parametrize('lazer',[False,True])
@pytest.mark.parametrize('general,expected',[(None,'normal'),('Normal','normal'),('Soft','soft'),('Drum','drum'),
    ('1','normal'),('2','soft'),('3','drum'),('+1','normal'),('02','soft'),('None',None),('0',None)])
def test_general_defined_enum_names_and_numbers_in_short_rows(sources,lazer,general,expected):
    beatmap=historical(sources,14,general=general)
    expected=expected or ('normal' if lazer else 'soft')
    assert resolve(sources,beatmap,lazer=lazer)[0].path.name==f'{expected}-hitnormal.wav'


def test_short_row_general_sample_set_cannot_be_a_path(sources):
    wave(sources['skin']/'secret-hitnormal.wav')
    beatmap=historical(sources,3,general=str(sources['skin']/'secret'))
    assert resolve(sources,beatmap)[0].path==sources['default']/'soft-hitnormal.wav'


def test_old_modern_row_with_explicit_zero_keeps_client_zero_rule(sources):
    beatmap=historical(sources,3,row='0,500,4,0,0,100,1,0')
    assert resolve(sources,beatmap)[0].path==sources['default']/'soft-hitnormal.wav'
    assert resolve(sources,beatmap,lazer=True)[0].path==sources['default']/'normal-hitnormal.wav'


def test_old_conventional_corrupt_banked_name_does_not_try_unprefixed_same_extension(sources):
    (sources['map']/'drum-hitnormal2.wav').write_bytes(b'corrupt'*300)
    wave(sources['map']/'hitnormal.wav')
    beatmap=historical(sources,4,sample_index=2)
    assert resolve(sources,beatmap)[0].path==sources['default']/'drum-hitnormal.wav'


def test_converted_parser_retains_old_audio_metadata(sources):
    beatmap=historical(sources,3,extra='SampleVolume:25\nCustomSamples:1\n',mode=0)
    assert beatmap.format_version==3 and beatmap.sample_volume==25 and beatmap.custom_samples==1
    assert beatmap.timing_points[0].field_count==2


@pytest.mark.parametrize('general',['All','-1'])
def test_stable_all_enum_uses_normal_cache_but_does_not_invent_extra_bank(sources,general):
    wave(sources['map']/'normal-hitnormal3.wav')
    beatmap=historical(sources,14,general=general,sample_index=3)
    assert resolve(sources,beatmap)[0].path==sources['default']/'normal-hitnormal.wav'


def test_old_explicit_original_name_wav_precedes_stem_ogg(sources):
    from tests.test_mania_audio_loader_edges import encoded_sample
    wave(sources['map']/'custom.missing.wav')
    encoded_sample(sources['map']/'custom.ogg')
    assert resolve(sources,historical(sources,4,filename='custom.missing'))[0].path.name=='custom.missing.wav'


@pytest.mark.parametrize('general',['4','-2'])
@pytest.mark.parametrize('lazer',[False,True])
def test_general_undefined_numeric_enum_is_safe_and_client_specific(sources,general,lazer):
    wave(sources['default']/f'{general}-hitnormal.wav')
    beatmap=historical(sources,14,general=general)
    expected=general if lazer else 'normal'
    assert resolve(sources,beatmap,lazer=lazer)[0].path.name==f'{expected}-hitnormal.wav'
