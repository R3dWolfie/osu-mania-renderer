# Running slow tests

The slow tests open a real OpenGL context and invoke ffmpeg. They're gated
behind `RUN_SLOW=1` so unit-test runs stay fast.

## Prerequisites

- `ffmpeg` on `$PATH` (already required for any render).
- A working `glcontext` install. On Bazzite/atomic Fedora, the easiest path is
  Toolbox:
  ```bash
  toolbox create -c osumania
  toolbox enter osumania
  sudo dnf install -y libX11-devel python3.13 ffmpeg mesa-libEGL-devel
  cd /var/home/red/Projects/Reddie/OsuManiaRenderer
  python3.13 -m venv .venv-toolbox
  . .venv-toolbox/bin/activate
  pip install -e ".[dev]"
  RUN_SLOW=1 pytest -q -m slow
  ```
- On a normal mutable distro: just `sudo dnf install libX11-devel` (or your
  distro's equivalent) and reinstall in the existing venv:
  ```bash
  pip install --force-reinstall glcontext
  RUN_SLOW=1 pytest -q -m slow
  ```

## What the slow tests cover

- GL context bootstrap (`tests/test_gpu_context.py`)
- Shader compilation (`tests/test_gpu_shaders.py`)
- Sprite atlas upload (`tests/test_gpu_atlas.py`)
- Frame readback (`tests/test_gpu_readback.py`)
- Each GPU draw pass: playfield, receptors, judgments, HUD/banner, background
  (`tests/test_gpu_renderer_*.py`)
- End-to-end render of the AO-INFINITY fixture replay
  (`tests/test_render_orchestrator.py::test_orchestrator_end_to_end`)
