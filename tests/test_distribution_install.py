"""Build real distributions; isolated installs must not import the checkout."""
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def run(args,cwd):
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env['PYTHONNOUSERSITE']='1'
    proc=subprocess.run([str(a) for a in args],cwd=cwd,env=env,capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    return proc.stdout


@pytest.fixture(scope='module')
def distributions(tmp_path_factory):
    if os.environ.get('RUN_SLOW')!='1':
        pytest.skip('RUN_SLOW=1 enables isolated build/install checks')
    workspace=tmp_path_factory.mktemp('distribution-build')
    # Build from a copy: do not leave build/egg-info artifacts in the checkout.
    import shutil
    source=workspace/'source';source.mkdir()
    for name in ('pyproject.toml','MANIFEST.in','README.md','LICENSE','COPYRIGHT'):
        shutil.copy2(ROOT/name,source/name)
    shutil.copytree(ROOT/'osu_mania_renderer_v2',source/'osu_mania_renderer_v2',
                    ignore=shutil.ignore_patterns('__pycache__','*.pre_recovery_bak'))
    run([sys.executable,'-m','build','--outdir',workspace/'dist'],source)
    return workspace/'dist'


@pytest.mark.slow
@pytest.mark.parametrize('kind',['wheel','sdist'])
def test_isolated_distribution_import_assets_dependency_cli_and_pcm(distributions,tmp_path,kind):
    archive=next(distributions.glob('*.whl' if kind=='wheel' else '*.tar.gz'))
    if kind=='wheel':
        with zipfile.ZipFile(archive) as z: names=z.namelist()
    else:
        with tarfile.open(archive) as t: names=t.getnames()
    assert not any('pre_recovery_bak' in p or 'prepush-audit.md' in p for p in names)
    venv=tmp_path/'venv'
    run([sys.executable,'-m','venv',venv],tmp_path)
    scripts=venv/('Scripts' if os.name=='nt' else 'bin')
    python=scripts/('python.exe' if os.name=='nt' else 'python')
    run([python,'-m','pip','install',archive],tmp_path)
    help_text=run([scripts/('osu-renderer.exe' if os.name=='nt' else 'osu-renderer'),'--help'],tmp_path)
    assert '--skin-hitsounds' in help_text and '--no-combo-break' in help_text
    smoke=r'''
import importlib.metadata as metadata
import json,sys
from pathlib import Path
import numpy as np,soundfile
import osu_mania_renderer_v2 as package
import osu_mania_renderer_v2.cli
import osu_mania_renderer_v2.beatmap
import osu_mania_renderer_v2.render
import osu_mania_renderer_v2.wiki_elements
from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import KeyEvent
from osu_mania_renderer_v2.render.lazer_mania_combo import build_lazer_combo_timeline
from osu_mania_renderer_v2.render.hitsounds import build_hitsound_track
root=Path(package.__file__).parent
assert root.is_relative_to(Path(sys.prefix))
for asset in ('classic_mania/mania-note1.png','classic_mania/provenance.json',
              'default_hitsounds/normal-hitnormal.wav','default_hitsounds/combobreak.mp3',
              'default_nightcore/nightcore-kick.wav','shaders/sprite.vert','shaders/sprite.frag'):
 assert (root/'assets'/asset).is_file(),asset
meta=metadata.metadata('osu-mania-renderer-v2')
assert meta['License-Expression']=='AGPL-3.0-or-later'
assert set(meta.get_all('License-File'))=={'LICENSE','COPYRIGHT'}
assert any(d.startswith('soundfile>=0.12') for d in metadata.requires('osu-mania-renderer-v2'))
path=Path('smoke.osu')
path.write_text('osu file format v14\n[General]\nMode:3\nSampleSet:Normal\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[HitObjects]\n64,192,100,1,0,0:0:0:10:\n')
b=parse_beatmap(path)
t=build_lazer_combo_timeline(b.notes,(KeyEvent(100,1),KeyEvent(101,0)),4,od=8)
wav=build_hitsound_track(beatmap=b,beatmap_dir=Path('.'),is_lazer_replay=True,lazer_facts=t.facts,
                       duration_ms=500,output_wav=Path('smoke.wav'))
samples,rate=soundfile.read(wav,always_2d=True)
assert samples.shape[1]==2 and rate==44100 and np.max(np.abs(samples))>0
print(json.dumps({'root':str(root),'soundfile':soundfile.__version__,'peak':float(np.max(np.abs(samples)))}))
'''
    result=run([python,'-I','-c',smoke],tmp_path)
    (tmp_path/'install-result.json').write_text(result)
