"""Real ffmpeg preview matrix: master identity and factual hit timing/gain."""
import json
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import KeyEvent, Note, TimingPoint
from osu_mania_renderer_v2.render.encode import LOUDNORM, build_ffmpeg_cmd
from osu_mania_renderer_v2.render.hitsounds import build_hitsound_track
from osu_mania_renderer_v2.render.lazer_mania_combo import build_lazer_combo_timeline
from tests.test_mania_hitsound_pipeline import MAP


def decode(path):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vn',
                             '-ac', '2', '-ar', '44100', '-f', 'f32le', 'pipe:1'],
                            capture_output=True, check=True, timeout=30)
    return np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2)


def streams(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of',
                             'json', str(path)], capture_output=True, check=True, timeout=30)
    return {s['codec_type']: s for s in json.loads(result.stdout)['streams']}


@pytest.mark.slow
@pytest.mark.parametrize('case,rate,pitch,with_hits,cached', [
    ('song', 1, False, False, False),
    ('hits', 1, False, True, False),
    ('dt', 1.5, False, True, False),
    ('nc', 1.5, True, True, False),
    ('cached', 1, False, True, True),
])
def test_inline_preview_preserves_master_and_single_factual_hit(tmp_path, case, rate, pitch, with_hits, cached):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    sr = 44100
    song = tmp_path / 'song.wav'
    wave = .05 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)
    sf.write(song, np.column_stack((wave, wave)), sr, subtype='FLOAT')
    # One source-accepted press, 10ms late. A decaying chirp is easy
    # to distinguish from the music without requiring subjective listening.
    sample_t = np.arange(int(sr * .12)) / sr
    sample = .2 * np.sin(2 * np.pi * (1500 * sample_t + 5000 * sample_t ** 2)) * np.exp(-sample_t * 20)
    sf.write(tmp_path / 'normal-hitnormal.wav', np.column_stack((sample, sample)), sr, subtype='FLOAT')
    beatmap = replace(parse_beatmap(MAP), notes=(Note(0, 750),),
                      timing_points=(TimingPoint(0, 1, 1, 100),))
    timeline = build_lazer_combo_timeline(beatmap.notes, (KeyEvent(760, 1), KeyEvent(800, 0)), 4, od=8, rate=rate)
    hits = build_hitsound_track(beatmap=beatmap, beatmap_dir=tmp_path,
        output_wav=tmp_path / 'hits.wav', duration_ms=2000, audio_rate=rate,
        is_lazer_replay=True, lazer_facts=timeline.facts, miss_hitsound=False)
    silence = tmp_path / 'silence.wav'
    sf.write(silence, np.zeros((2 * sr, 2)), sr, subtype='PCM_16')
    pcm = tmp_path / 'cached.f32le'
    if cached:
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(song), '-af', LOUDNORM,
                        '-ar', '48000', '-ac', '2', '-f', 'f32le', str(pcm)],
                       capture_output=True, check=True, timeout=30)

    frames = np.zeros((60, 90, 160, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(160)
    frames[:, :, :, 1] = np.arange(90)[:, None]
    outputs = {}
    for mode in ('off', 'on', 'silent') if with_hits else ('off', 'on'):
        output = tmp_path / f'{case}-{mode}.mp4'
        preview = tmp_path / f'{case}-{mode}-preview.mp4' if mode != 'off' else None
        cmd = build_ffmpeg_cmd(encoder='libx264', encoder_device=None, resolution=(160, 90), fps=30,
            audio_path=song, audio_rate=rate, audio_pitch=pitch, audio_lead_in_ms=0,
            prenormalized_audio_path=pcm if cached else None, video_bitrate='300k', audio_bitrate='160k',
            output_path=output, total_duration_ms=2000,
            hitsound_path=(silence if mode == 'silent' else hits) if with_hits else None,
            preview_path=preview)
        graph = cmd[cmd.index('-filter_complex') + 1]
        assert graph.count('amix=') == int(with_hits)
        assert graph.count('loudnorm=') == int(not cached) + int(preview is not None)
        if with_hits:
            assert '[2:a]anull[hits]' in graph
            if not cached:
                assert graph.index('loudnorm=') < graph.index('[song][hits]amix=')
        if preview:
            assert '[aout]asplit=2[am][ap0];[ap0]aresample,loudnorm=' in graph
            if with_hits:
                assert graph.index('asplit=') > graph.index('amix=')
        result = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr.decode()
        outputs[mode] = output, preview

    off, on = outputs['off'][0], outputs['on'][0]
    assert off.read_bytes() == on.read_bytes()  # #28's strongest master guarantee
    assert streams(off)['audio']['sample_rate'] == streams(on)['audio']['sample_rate']
    if cached:
        assert streams(on)['audio']['sample_rate'] == '48000'
    preview_streams = streams(outputs['on'][1])
    assert preview_streams['video']['codec_name'] == 'h264'
    assert (preview_streams['video']['width'], preview_streams['video']['height']) == (1280, 720)
    assert preview_streams['video']['r_frame_rate'] == '30/1'
    assert preview_streams['audio']['sample_rate'] == '48000'
    assert np.sqrt(np.mean(decode(on) ** 2)) > .001
    assert np.sqrt(np.mean(decode(outputs['on'][1]) ** 2)) > .001
    if not with_hits:
        return

    expected, _ = sf.read(hits, always_2d=True)
    start = int(760 / rate / 1000 * sr)
    end = start + len(sample)
    for index in (0, 1):  # master and independently normalised preview
        mixed, quiet = decode(outputs['on'][index]), decode(outputs['silent'][index])
        length = min(len(mixed), len(quiet))
        diff = mixed[:length] - quiet[:length]
        before = diff[max(0, start - sr // 4):start - sr // 10]
        hit = diff[start:end + 90]
        assert np.sqrt(np.mean(hit ** 2)) > .01
        assert np.sqrt(np.mean(hit ** 2)) > 10 * np.sqrt(np.mean(before ** 2))
        if index == 0:
            # AAC is lossy; the limiter adds 1ms. Fit the source waveform
            # over that small latency range and reject doubled/attenuated hits.
            reference = expected[start:end]
            fits = []
            for shift in range(90):
                actual = diff[start + shift:end + shift]
                gain = np.sum(actual * reference) / np.sum(reference ** 2)
                fits.append((np.mean((actual - gain * reference) ** 2), gain))
            error, gain = min(fits)
            assert gain == pytest.approx(1, abs=.08)
            assert error < .0001
