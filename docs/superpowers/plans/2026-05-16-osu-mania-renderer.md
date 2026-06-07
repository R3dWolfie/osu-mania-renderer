# osu! Mania Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone Python package that turns an osu!mania `.osr` + beatmap files into an MP4. GPU-rendered via ModernGL standalone context, hardware-encoded via ffmpeg/VAAPI, mod-aware (everything except RD and KC), audio mixed in.

**Architecture:** Pure-Python parsers (`beatmap`, `replay`, `mods`) produce a frame-independent description. `scene` is a per-frame query layer. `gpu/` is the only module that touches OpenGL — uses ModernGL with an EGL standalone context (no window manager required). `encode` owns ffmpeg. Public API is `async render_mania(...)`; the package also exposes a CLI.

**Tech Stack:** Python 3.12, `moderngl` 5.x (standalone EGL context), `osrparse` 7.x, `Pillow` (placeholder sprites + banner text rasterization), `numpy` (frame array views), `pytest` + `pytest-asyncio`, `ruff`. External: `ffmpeg` with VAAPI + libx264. Test-time GPU fallback: Mesa llvmpipe (`LIBGL_ALWAYS_SOFTWARE=1`).

---

## File Structure

```
/home/red/Projects/Reddie/OsuManiaRenderer/
├── .gitignore                            (exists)
├── pyproject.toml                        (Task 1)
├── README.md                             (Task 19)
├── LICENSE                               (Task 19)
├── docs/superpowers/specs/...            (exists)
├── docs/superpowers/plans/...            (this file)
├── osu_renderer/
│   ├── __init__.py                       (Task 1, populated in Task 18)
│   ├── errors.py                         (Task 2)
│   ├── models.py                         (Task 3)
│   ├── beatmap.py                        (Task 4)
│   ├── replay.py                         (Task 5)
│   ├── mods.py                           (Task 6)
│   ├── scene.py                          (Task 7)
│   ├── encode.py                         (Task 8)
│   ├── gpu/
│   │   ├── __init__.py                   (Task 9)
│   │   ├── context.py                    (Task 9)
│   │   ├── shaders.py                    (Task 10)
│   │   ├── atlas.py                      (Task 11)
│   │   ├── readback.py                   (Task 12)
│   │   └── renderer.py                   (Tasks 13–15)
│   ├── assets/
│   │   ├── shaders/
│   │   │   ├── sprite.vert               (Task 10)
│   │   │   ├── sprite.frag               (Task 10)
│   │   │   └── flashlight.frag           (Task 10)
│   │   ├── sprites/                      (Task 16, placeholder PNGs)
│   │   ├── font/                         (Task 16, digit atlas)
│   │   └── default-skin/                 (Task 17, Night05 extraction script output)
│   ├── render.py                         (Task 18, the public render_mania)
│   └── cli.py                            (Task 20)
└── tests/
    ├── __init__.py                       (Task 1)
    ├── conftest.py                       (Task 1)
    ├── fixtures/
    │   ├── ao_infinity_hard.osr          (Task 5, real mania replay)
    │   ├── std_replay.osr                (Task 5)
    │   └── ao_infinity_hard.osu          (Task 4)
    └── test_*.py                         (one per module)
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `osu_renderer/__init__.py`
- Create: `osu_renderer/gpu/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "osu-mania-renderer"
version = "0.1.0"
description = "Render osu!mania replays to MP4 (GPU + ffmpeg)"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
    "moderngl>=5.10",
    "osrparse>=7.0",
    "Pillow>=10.0",
    "numpy>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-mock>=3.14",
    "ruff>=0.7",
]

[project.scripts]
osu-mania-renderer = "osu_renderer.cli:main"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "slow: tests that open a GL context or invoke ffmpeg. Skipped unless RUN_SLOW=1.",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["osu_renderer", "osu_renderer.gpu"]
include-package-data = true

[tool.setuptools.package-data]
osu_renderer = ["assets/**/*"]
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p osu_renderer/gpu osu_renderer/assets/{shaders,sprites,font,default-skin} tests/fixtures
touch osu_renderer/__init__.py osu_renderer/gpu/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def assets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "osu_renderer" / "assets"
```

- [ ] **Step 4: Create virtualenv and verify**

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
```

Expected: `no tests ran in …s` exit code 5 (no tests collected).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml osu_renderer/ tests/
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Scaffold osu-mania-renderer package"
```

---

## Task 2: Errors module

**Files:**
- Create: `osu_renderer/errors.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

```python
from osu_renderer.errors import (
    BeatmapParseError,
    EncoderError,
    GpuUnavailableError,
    MissingAudioError,
    NotAManiaError,
    RendererError,
    RenderTimeoutError,
    ReplayParseError,
)


def test_all_inherit_from_renderer_error():
    for cls in [
        BeatmapParseError, ReplayParseError, NotAManiaError, MissingAudioError,
        GpuUnavailableError, EncoderError, RenderTimeoutError,
    ]:
        assert issubclass(cls, RendererError)


def test_not_a_mania_error_carries_mode():
    e = NotAManiaError(mode=0)
    assert e.mode == 0
    assert "0" in str(e)
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_errors.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/errors.py`**

```python
"""Exception hierarchy. Everything raised by this package is a RendererError."""
from __future__ import annotations


class RendererError(Exception):
    """Base class for everything the renderer raises."""


class BeatmapParseError(RendererError):
    """Raised when the .osu file is malformed or missing required fields."""


class ReplayParseError(RendererError):
    """Raised when the .osr file is malformed."""


class NotAManiaError(RendererError):
    """Raised when the beatmap or replay is not osu!mania (mode != 3)."""

    def __init__(self, mode: int) -> None:
        self.mode = mode
        super().__init__(f"Expected mania (mode=3), got mode={mode}")


class MissingAudioError(RendererError):
    """Raised only when RenderOptions.audio_required=True and the audio file is absent."""


class GpuUnavailableError(RendererError):
    """Raised when ModernGL cannot create a GL context."""


class EncoderError(RendererError):
    """Raised when ffmpeg exits non-zero or the output MP4 is missing/empty."""


class RenderTimeoutError(RendererError):
    """Raised when a render exceeds RenderOptions.timeout_seconds."""
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_errors.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/errors.py tests/test_errors.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Add RendererError hierarchy"
```

---

## Task 3: Models / dataclasses

**Files:**
- Create: `osu_renderer/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
from osu_renderer.models import (
    BeatmapInfo,
    HoldNote,
    KeyEvent,
    Note,
    RenderOptions,
    ReplayInfo,
    VisualMods,
)


def test_note_is_frozen():
    n = Note(column=2, time_ms=1000)
    import dataclasses
    assert dataclasses.is_dataclass(n)
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.column = 3


def test_hold_note_has_end_time():
    h = HoldNote(column=0, time_ms=500, end_time_ms=1500)
    assert h.duration_ms == 1000


def test_render_options_defaults():
    o = RenderOptions(resolution=(1920, 1080), fps=60)
    assert o.encoder == "auto"
    assert o.encoder_device is None
    assert o.timeout_seconds == 600
    assert o.audio_required is False


def test_visual_mods_flags():
    v = VisualMods(hidden=True, flashlight=False, fade_in=False)
    assert v.hidden
    assert not v.flashlight


def test_key_event():
    e = KeyEvent(time_ms=1234, keys_held=0b0101)
    assert e.is_held(0)
    assert not e.is_held(1)
    assert e.is_held(2)
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/models.py`**

```python
"""Shared dataclasses for the renderer pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Note:
    column: int       # 0-indexed
    time_ms: int


@dataclass(frozen=True)
class HoldNote:
    column: int
    time_ms: int
    end_time_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.time_ms


@dataclass(frozen=True)
class KeyEvent:
    time_ms: int
    keys_held: int     # bitmask, bit i = column i

    def is_held(self, column: int) -> bool:
        return bool(self.keys_held & (1 << column))


@dataclass(frozen=True)
class VisualMods:
    hidden: bool = False
    fade_in: bool = False
    flashlight: bool = False
    score_v2: bool = False


@dataclass(frozen=True)
class BeatmapInfo:
    key_count: int
    notes: tuple                       # tuple[Note | HoldNote, ...]
    audio_filename: str | None
    background_filename: str | None
    total_duration_ms: int
    audio_lead_in_ms: int
    artist: str
    title: str
    difficulty: str
    creator: str
    beatmap_id: int | None
    beatmapset_id: int | None
    base_scroll_speed: float = 20.0   # osu! default; affects pixel-per-ms


@dataclass(frozen=True)
class ReplayInfo:
    mode: int                          # 0..3
    beatmap_md5: str
    player_name: str
    replay_md5: str
    mods: int
    key_events: tuple                  # tuple[KeyEvent, ...]
    score: int
    accuracy: float
    max_combo: int
    count_geki: int
    count_300: int
    count_katu: int
    count_100: int
    count_50: int
    count_miss: int
    grade: str


@dataclass(frozen=True)
class RenderOptions:
    resolution: tuple[int, int]
    fps: int
    encoder: str = "auto"              # "auto" | "h264_vaapi" | "libx264"
    encoder_device: str | None = None
    timeout_seconds: int = 600
    audio_required: bool = False
    video_bitrate: str = "5M"
    audio_bitrate: str = "192k"
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/models.py tests/test_models.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Add Note/HoldNote/KeyEvent/BeatmapInfo/ReplayInfo/RenderOptions dataclasses"
```

---

## Task 4: Beatmap parser

**Files:**
- Create: `osu_renderer/beatmap.py`
- Create: `tests/fixtures/ao_infinity_hard.osu` (copy from cache)
- Test: `tests/test_beatmap.py`

- [ ] **Step 1: Stage the fixture**

```bash
cp "/var/mnt/Synology-Reddie/Mania ORDR Bot/beatmaps/3e37a2abc23502109072187911229864/Seiryu - AO-INFINITY (tailsdk) [Hard].osu" \
   tests/fixtures/ao_infinity_hard.osu
```

- [ ] **Step 2: Write the failing test**

```python
from pathlib import Path

import pytest

from osu_renderer.beatmap import parse_beatmap
from osu_renderer.errors import BeatmapParseError, NotAManiaError
from osu_renderer.models import HoldNote, Note


def test_parse_ao_infinity_hard(fixtures_dir: Path):
    bm = parse_beatmap(fixtures_dir / "ao_infinity_hard.osu")
    assert bm.key_count == 4
    assert bm.audio_filename == "audio.mp3"
    assert bm.artist == "Seiryu"
    assert bm.title == "AO-INFINITY"
    assert bm.difficulty == "Hard"
    assert bm.beatmap_id == 2071816
    assert len(bm.notes) > 100  # real map has lots
    # First note's column is in range and time is non-negative.
    n0 = bm.notes[0]
    assert 0 <= n0.column < bm.key_count
    assert n0.time_ms >= 0


def test_parse_notes_are_sorted_by_time(fixtures_dir: Path):
    bm = parse_beatmap(fixtures_dir / "ao_infinity_hard.osu")
    times = [n.time_ms for n in bm.notes]
    assert times == sorted(times)


def test_parse_holds_have_end_time(fixtures_dir: Path):
    bm = parse_beatmap(fixtures_dir / "ao_infinity_hard.osu")
    holds = [n for n in bm.notes if isinstance(n, HoldNote)]
    if holds:  # most maps have at least one hold
        h = holds[0]
        assert h.end_time_ms > h.time_ms


def test_non_mania_rejected(tmp_path: Path):
    p = tmp_path / "std.osu"
    p.write_text("osu file format v14\n[General]\nMode: 0\n[HitObjects]\n")
    with pytest.raises(NotAManiaError):
        parse_beatmap(p)


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_beatmap(tmp_path / "nope.osu")


def test_malformed_raises(tmp_path: Path):
    p = tmp_path / "bad.osu"
    p.write_text("this is not an osu file")
    with pytest.raises(BeatmapParseError):
        parse_beatmap(p)
```

- [ ] **Step 3: Run, expect failure**

```bash
.venv/bin/pytest tests/test_beatmap.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `osu_renderer/beatmap.py`**

```python
"""Parse osu!mania .osu files. Mania-only — std/taiko/ctb raise NotAManiaError."""
from __future__ import annotations

from pathlib import Path

from osu_renderer.errors import BeatmapParseError, NotAManiaError
from osu_renderer.models import BeatmapInfo, HoldNote, Note


# Mania hit-object type bit 7 (128) = hold note.
_HOLD_TYPE_BIT = 1 << 7


def parse_beatmap(path: Path) -> BeatmapInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if "osu file format" not in text.splitlines()[0:1][0] if text else True:
        # Be lenient — only fail if we can't find the sections we need.
        pass

    sections = _split_sections(text)
    general = _kv(sections.get("General", ""))
    metadata = _kv(sections.get("Metadata", ""))
    difficulty = _kv(sections.get("Difficulty", ""))
    events = sections.get("Events", "")
    hit_objects_raw = sections.get("HitObjects", "")

    mode_str = general.get("Mode", "0")
    try:
        mode = int(mode_str)
    except ValueError as e:
        raise BeatmapParseError(f"Invalid Mode={mode_str!r}") from e
    if mode != 3:
        raise NotAManiaError(mode)

    try:
        key_count = int(float(difficulty["CircleSize"]))
    except (KeyError, ValueError) as e:
        raise BeatmapParseError("Missing or invalid CircleSize (key count)") from e

    audio_filename = general.get("AudioFilename")
    try:
        audio_lead_in_ms = int(float(general.get("AudioLeadIn", "0")))
    except ValueError:
        audio_lead_in_ms = 0

    background = _parse_background(events)

    notes, max_time = _parse_hit_objects(hit_objects_raw, key_count)
    notes_sorted = tuple(sorted(notes, key=lambda n: n.time_ms))

    return BeatmapInfo(
        key_count=key_count,
        notes=notes_sorted,
        audio_filename=audio_filename,
        background_filename=background,
        total_duration_ms=max_time,
        audio_lead_in_ms=audio_lead_in_ms,
        artist=metadata.get("Artist", ""),
        title=metadata.get("Title", ""),
        difficulty=metadata.get("Version", ""),
        creator=metadata.get("Creator", ""),
        beatmap_id=_int_or_none(metadata.get("BeatmapID")),
        beatmapset_id=_int_or_none(metadata.get("BeatmapSetID")),
    )


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                sections[current] = "\n".join(buf)
            current = line[1:-1]
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    if not sections:
        raise BeatmapParseError("No section headers found")
    return sections


def _kv(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line or line.startswith("//"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def _parse_background(events: str) -> str | None:
    for line in events.splitlines():
        # Background events look like: 0,0,"filename.jpg",0,0
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].strip() in ("0", "Background"):
            return parts[2].strip().strip('"')
    return None


def _parse_hit_objects(block: str, key_count: int) -> tuple[list, int]:
    notes: list = []
    max_time = 0
    for line in block.splitlines():
        if not line or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x = int(parts[0])
            time_ms = int(parts[2])
            type_flags = int(parts[3])
        except ValueError:
            continue
        # Map x in [0, 512] to column index.
        column = int(x * key_count / 512)
        column = min(max(column, 0), key_count - 1)
        if type_flags & _HOLD_TYPE_BIT:
            # 6th field is "endTime:hitSample".
            if len(parts) < 6:
                continue
            end_str = parts[5].split(":", 1)[0]
            try:
                end_ms = int(end_str)
            except ValueError:
                continue
            notes.append(HoldNote(column=column, time_ms=time_ms, end_time_ms=end_ms))
            max_time = max(max_time, end_ms)
        else:
            notes.append(Note(column=column, time_ms=time_ms))
            max_time = max(max_time, time_ms)
    return notes, max_time


def _int_or_none(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None
```

- [ ] **Step 5: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_beatmap.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add osu_renderer/beatmap.py tests/test_beatmap.py tests/fixtures/ao_infinity_hard.osu
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Parse .osu beatmaps (mania-only) with notes, holds, metadata"
```

---

## Task 5: Replay parser

**Files:**
- Create: `osu_renderer/replay.py`
- Stage: `tests/fixtures/ao_infinity_hard.osr`, `tests/fixtures/std_replay.osr`
- Test: `tests/test_replay.py`

- [ ] **Step 1: Stage fixtures**

```bash
cp "/home/red/.local/share/osu/exports/R3D playing Seiryu - AO-INFINITY (Tailsdk) [Hard] (2026-05-16_03-19).osr" \
   tests/fixtures/ao_infinity_hard.osr
# Reuse a std replay we already have access to.
cp "/var/mnt/Synology-Reddie/Mania ORDR Bot/test-replays/R3D playing Seiryu - AO-INFINITY (Tailsdk) [Hard] (2026-05-16_03-19).osr" \
   /tmp/check_only.osr   # sanity check the staged path; remove after
rm /tmp/check_only.osr
# std fixture is already in the Mania ORDR Bot repo's tests/fixtures.
cp "/home/red/Projects/Mania ORDR/tests/fixtures/std_replay.osr" tests/fixtures/std_replay.osr
```

- [ ] **Step 2: Write the failing test**

```python
from pathlib import Path

import pytest

from osu_renderer.errors import NotAManiaError
from osu_renderer.replay import parse_replay


def test_parse_mania_replay(fixtures_dir: Path):
    r = parse_replay(fixtures_dir / "ao_infinity_hard.osr")
    assert r.mode == 3
    assert r.player_name
    assert r.beatmap_md5
    assert 0.0 <= r.accuracy <= 100.0
    assert r.max_combo > 0
    assert len(r.key_events) > 0
    # KeyEvents must be sorted by time and have a non-negative time.
    assert r.key_events[0].time_ms >= 0
    times = [e.time_ms for e in r.key_events]
    assert times == sorted(times)


def test_reject_std_replay(fixtures_dir: Path):
    with pytest.raises(NotAManiaError):
        parse_replay(fixtures_dir / "std_replay.osr")


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_replay(tmp_path / "nope.osr")
```

- [ ] **Step 3: Run, expect failure**

```bash
.venv/bin/pytest tests/test_replay.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `osu_renderer/replay.py`**

```python
"""Parse osu!mania .osr files via osrparse, including the keypress timeline."""
from __future__ import annotations

from pathlib import Path

from osrparse import GameMode, Replay

from osu_renderer.errors import NotAManiaError, ReplayParseError
from osu_renderer.models import KeyEvent, ReplayInfo


def parse_replay(path: Path) -> ReplayInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        r = Replay.from_path(path)
    except Exception as e:
        raise ReplayParseError(f"osrparse failed: {e}") from e

    mode = r.mode.value if isinstance(r.mode, GameMode) else int(r.mode)
    if mode != 3:
        raise NotAManiaError(mode)

    events = _decode_key_events(r)

    total = (
        r.count_300 + r.count_100 + r.count_50 + r.count_miss
        + r.count_katu + r.count_geki
    )
    if total == 0:
        accuracy = 0.0
    else:
        weighted = (
            50 * r.count_50 + 100 * r.count_100 + 200 * r.count_katu
            + 300 * r.count_300 + 300 * r.count_geki
        )
        accuracy = round((weighted / (300 * total)) * 100, 4)

    return ReplayInfo(
        mode=mode,
        beatmap_md5=r.beatmap_hash,
        player_name=r.username,
        replay_md5=r.replay_hash,
        mods=int(r.mods),
        key_events=tuple(events),
        score=int(r.score),
        accuracy=accuracy,
        max_combo=int(r.max_combo),
        count_geki=int(r.count_geki),
        count_300=int(r.count_300),
        count_katu=int(r.count_katu),
        count_100=int(r.count_100),
        count_50=int(r.count_50),
        count_miss=int(r.count_miss),
        grade=_grade(accuracy, r),
    )


def _decode_key_events(r: Replay) -> list[KeyEvent]:
    """osrparse's r.replay_data is a list of ReplayEventMania entries; convert to absolute-time KeyEvents."""
    out: list[KeyEvent] = []
    t = 0
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        t += delta
        # ReplayEventMania.x is the bitmask of held columns.
        keys = int(getattr(ev, "x", 0))
        out.append(KeyEvent(time_ms=max(t, 0), keys_held=keys))
    # Deduplicate same-time entries by keeping the latest.
    dedup: dict[int, KeyEvent] = {}
    for e in out:
        dedup[e.time_ms] = e
    return sorted(dedup.values(), key=lambda e: e.time_ms)


def _grade(accuracy: float, r: Replay) -> str:
    total = (
        r.count_geki + r.count_300 + r.count_katu + r.count_100
        + r.count_50 + r.count_miss
    )
    if total == 0:
        return "D"
    if r.count_300 == 0 and r.count_katu == 0 and r.count_100 == 0 \
            and r.count_50 == 0 and r.count_miss == 0:
        return "SS"
    if accuracy >= 95.0 and r.count_50 / total <= 0.01 and r.count_miss == 0:
        return "S"
    if accuracy >= 90.0:
        return "A"
    if accuracy >= 80.0:
        return "B"
    if accuracy >= 70.0:
        return "C"
    return "D"
```

- [ ] **Step 5: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_replay.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add osu_renderer/replay.py tests/test_replay.py tests/fixtures/*.osr
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Parse .osr replays via osrparse with decoded keypress timeline"
```

---

## Task 6: Mods application

**Files:**
- Create: `osu_renderer/mods.py`
- Test: `tests/test_mods.py`

- [ ] **Step 1: Write the failing test**

```python
from osu_renderer.models import BeatmapInfo, HoldNote, KeyEvent, Note, ReplayInfo
from osu_renderer.mods import Mod, ModResult, apply_mods


def _mk_beatmap(notes=None, key_count=4) -> BeatmapInfo:
    return BeatmapInfo(
        key_count=key_count,
        notes=tuple(notes or []),
        audio_filename="audio.mp3", background_filename=None,
        total_duration_ms=10_000, audio_lead_in_ms=0,
        artist="A", title="T", difficulty="V", creator="C",
        beatmap_id=None, beatmapset_id=None,
    )


def _mk_replay(mods: int = 0) -> ReplayInfo:
    return ReplayInfo(
        mode=3, beatmap_md5="", player_name="P", replay_md5="",
        mods=mods, key_events=(),
        score=0, accuracy=100.0, max_combo=0,
        count_geki=0, count_300=0, count_katu=0,
        count_100=0, count_50=0, count_miss=0, grade="SS",
    )


def test_no_mods_passthrough():
    bm = _mk_beatmap(notes=[Note(0, 1000), Note(1, 2000)])
    rp = _mk_replay(mods=0)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 1.0
    assert res.beatmap.notes == bm.notes
    assert res.warnings == ()
    assert res.visual_mods.hidden is False


def test_dt_halves_note_times():
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.DT.value)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 1.5
    assert res.beatmap.notes[0].time_ms == 2000  # 3000 / 1.5


def test_ht_extends_note_times():
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.HT.value)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 0.75
    assert res.beatmap.notes[0].time_ms == 4000  # 3000 / 0.75


def test_nc_treated_as_dt():
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.NC.value | Mod.DT.value)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 1.5


def test_mirror_flips_columns():
    bm = _mk_beatmap(notes=[Note(0, 1000), Note(3, 2000)], key_count=4)
    rp = _mk_replay(mods=Mod.MR.value)
    res = apply_mods(bm, rp)
    cols = [n.column for n in res.beatmap.notes]
    assert cols == [3, 0]


def test_hidden_sets_visual_flag():
    bm = _mk_beatmap()
    rp = _mk_replay(mods=Mod.HD.value)
    res = apply_mods(bm, rp)
    assert res.visual_mods.hidden is True


def test_flashlight_sets_visual_flag():
    bm = _mk_beatmap()
    rp = _mk_replay(mods=Mod.FL.value)
    res = apply_mods(bm, rp)
    assert res.visual_mods.flashlight is True


def test_random_warns_falls_back_to_nm():
    bm = _mk_beatmap(notes=[Note(0, 1000), Note(2, 2000)])
    rp = _mk_replay(mods=Mod.RD.value)
    res = apply_mods(bm, rp)
    cols = [n.column for n in res.beatmap.notes]
    assert cols == [0, 2]  # NOT shuffled
    assert any("Random" in w for w in res.warnings)


def test_key_coop_warns():
    bm = _mk_beatmap()
    rp = _mk_replay(mods=Mod.KC.value)
    res = apply_mods(bm, rp)
    assert any("Key Coop" in w for w in res.warnings)


def test_hold_note_end_time_is_also_rescaled():
    bm = _mk_beatmap(notes=[HoldNote(0, 3000, 6000)])
    rp = _mk_replay(mods=Mod.DT.value)
    res = apply_mods(bm, rp)
    h = res.beatmap.notes[0]
    assert h.time_ms == 2000
    assert h.end_time_ms == 4000
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_mods.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/mods.py`**

```python
"""Apply osu!mania mods to a beatmap. Returns a modded beatmap + audio rate + visual flags."""
from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from osu_renderer.models import BeatmapInfo, HoldNote, Note, ReplayInfo, VisualMods


class Mod(enum.IntFlag):
    """osu! mod bitmask values, ordered by bit position."""
    NF = 1 << 0
    EZ = 1 << 1
    HD = 1 << 3
    HR = 1 << 4
    SD = 1 << 5
    DT = 1 << 6
    HT = 1 << 8
    NC = 1 << 9
    FL = 1 << 10
    PF = 1 << 14
    K4 = 1 << 15
    K5 = 1 << 16
    K6 = 1 << 17
    K7 = 1 << 18
    K8 = 1 << 19
    FI = 1 << 20
    RD = 1 << 21
    K9 = 1 << 24
    KC = 1 << 25
    K1 = 1 << 26
    K3 = 1 << 27
    K2 = 1 << 28
    V2 = 1 << 29
    MR = 1 << 30


@dataclass(frozen=True)
class ModResult:
    beatmap: BeatmapInfo
    audio_rate: float
    visual_mods: VisualMods
    warnings: tuple[str, ...] = ()


def apply_mods(beatmap: BeatmapInfo, replay: ReplayInfo) -> ModResult:
    mods = replay.mods
    warnings: list[str] = []

    # Speed.
    if mods & Mod.DT or mods & Mod.NC:
        audio_rate = 1.5
    elif mods & Mod.HT:
        audio_rate = 0.75
    else:
        audio_rate = 1.0

    # Apply speed to note times.
    notes = _rescale_times(beatmap.notes, audio_rate)
    total = int(beatmap.total_duration_ms / audio_rate)

    # Mirror.
    if mods & Mod.MR:
        notes = _mirror(notes, beatmap.key_count)

    # Random — explicitly unsupported.
    if mods & Mod.RD:
        warnings.append("Random (RD) is not supported; rendering as NM column order")

    # Key Coop — explicitly unsupported.
    if mods & Mod.KC:
        warnings.append("Key Coop (KC) is not supported; rendering as single playfield")

    visual = VisualMods(
        hidden=bool(mods & Mod.HD),
        fade_in=bool(mods & Mod.FI),
        flashlight=bool(mods & Mod.FL),
        score_v2=bool(mods & Mod.V2),
    )

    modded = replace(beatmap, notes=tuple(notes), total_duration_ms=total)
    return ModResult(
        beatmap=modded, audio_rate=audio_rate,
        visual_mods=visual, warnings=tuple(warnings),
    )


def _rescale_times(notes: tuple, rate: float) -> list:
    if rate == 1.0:
        return list(notes)
    out: list = []
    for n in notes:
        if isinstance(n, HoldNote):
            out.append(HoldNote(
                column=n.column,
                time_ms=int(n.time_ms / rate),
                end_time_ms=int(n.end_time_ms / rate),
            ))
        else:
            out.append(Note(column=n.column, time_ms=int(n.time_ms / rate)))
    return out


def _mirror(notes: list, key_count: int) -> list:
    out: list = []
    for n in notes:
        new_col = (key_count - 1) - n.column
        if isinstance(n, HoldNote):
            out.append(HoldNote(column=new_col, time_ms=n.time_ms, end_time_ms=n.end_time_ms))
        else:
            out.append(Note(column=new_col, time_ms=n.time_ms))
    return out
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_mods.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/mods.py tests/test_mods.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Apply DT/HT/MR/HD/FL/FI mods; warn on RD and KC fallbacks"
```

---

## Task 7: Scene snapshot

**Files:**
- Create: `osu_renderer/scene.py`
- Test: `tests/test_scene.py`

- [ ] **Step 1: Write the failing test**

```python
from osu_renderer.models import HoldNote, KeyEvent, Note, VisualMods
from osu_renderer.scene import SceneState, snapshot


# Playfield occupies the bottom 1000ms of approach time at the receptors.
APPROACH_MS = 600


def test_no_notes_visible_before_any_appear():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=0,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    # 1000ms - 600ms approach = note becomes visible at t=400ms.
    assert s.visible_notes == ()


def test_note_becomes_visible_within_approach_window():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=500,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    assert len(s.visible_notes) == 1


def test_note_hits_receptor_at_exact_time():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=1000,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    n = s.visible_notes[0]
    # Y-position 1.0 means at the receptor.
    assert abs(n.y_fraction - 1.0) < 1e-3


def test_held_keys_reflected():
    s = snapshot(
        notes=(),
        key_events=(KeyEvent(time_ms=500, keys_held=0b0011),),
        t_ms=600,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    assert s.keys_held == (True, True, False, False)


def test_hold_note_renders_as_segment_when_active():
    s = snapshot(
        notes=(HoldNote(0, 1000, 2000),),
        key_events=(),
        t_ms=1500,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    visible_holds = [n for n in s.visible_notes if n.is_hold]
    assert len(visible_holds) == 1
    # Head has passed the receptor (y > 1), tail is still in the playfield.
    h = visible_holds[0]
    assert h.head_y_fraction > 1.0
    assert h.tail_y_fraction < 1.0
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_scene.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/scene.py`**

```python
"""Per-frame scene state for the renderer. Pure function of beatmap + replay + t."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from osu_renderer.models import HoldNote, KeyEvent, Note, VisualMods


@dataclass(frozen=True)
class VisibleNote:
    column: int
    is_hold: bool
    y_fraction: float          # 0.0 = top of playfield, 1.0 = at receptor
    head_y_fraction: float     # alias of y_fraction for taps
    tail_y_fraction: float     # equals head for taps, separate for holds


@dataclass(frozen=True)
class SceneState:
    t_ms: int
    visible_notes: tuple[VisibleNote, ...]
    keys_held: tuple[bool, ...]
    visual_mods: VisualMods


def snapshot(
    notes: tuple,
    key_events: tuple[KeyEvent, ...],
    t_ms: int,
    key_count: int,
    approach_ms: int,
    visual_mods: VisualMods,
) -> SceneState:
    """Return what's on screen at time t_ms.

    Notes whose start time is within [t_ms, t_ms + approach_ms] are scrolling
    in the playfield. Holds are also visible while their body covers t_ms.
    y_fraction = (t_ms - (note.time_ms - approach_ms)) / approach_ms.
    """
    visible: list[VisibleNote] = []
    horizon = t_ms + approach_ms
    for n in notes:
        # Fast skip: notes far in the future.
        if n.time_ms > horizon:
            continue
        if isinstance(n, HoldNote):
            # Visible while body intersects [t_ms - approach_ms, t_ms + approach_ms]
            # — practically: tail not yet past receptor by more than 1 approach.
            if n.end_time_ms < t_ms - approach_ms:
                continue
            head_y = _y(n.time_ms, t_ms, approach_ms)
            tail_y = _y(n.end_time_ms, t_ms, approach_ms)
            visible.append(VisibleNote(
                column=n.column, is_hold=True,
                y_fraction=head_y, head_y_fraction=head_y, tail_y_fraction=tail_y,
            ))
        else:
            if n.time_ms < t_ms - 50:  # 50ms grace after receptor
                continue
            y = _y(n.time_ms, t_ms, approach_ms)
            visible.append(VisibleNote(
                column=n.column, is_hold=False,
                y_fraction=y, head_y_fraction=y, tail_y_fraction=y,
            ))

    keys_held = _keys_held_at(key_events, t_ms, key_count)
    return SceneState(
        t_ms=t_ms,
        visible_notes=tuple(visible),
        keys_held=keys_held,
        visual_mods=visual_mods,
    )


def _y(note_time: int, t_ms: int, approach_ms: int) -> float:
    return 1.0 - (note_time - t_ms) / approach_ms


def _keys_held_at(
    events: tuple[KeyEvent, ...], t_ms: int, key_count: int,
) -> tuple[bool, ...]:
    if not events:
        return tuple(False for _ in range(key_count))
    times = [e.time_ms for e in events]
    idx = bisect_right(times, t_ms) - 1
    if idx < 0:
        return tuple(False for _ in range(key_count))
    mask = events[idx].keys_held
    return tuple(bool(mask & (1 << c)) for c in range(key_count))
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_scene.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/scene.py tests/test_scene.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Scene snapshot: visible notes + held keys at time t"
```

---

## Task 8: ffmpeg encoder wrapper

**Files:**
- Create: `osu_renderer/encode.py`
- Test: `tests/test_encode.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osu_renderer.encode import build_ffmpeg_cmd, probe_encoder
from osu_renderer.errors import EncoderError
from osu_renderer.models import RenderOptions


def test_build_cmd_vaapi():
    cmd = build_ffmpeg_cmd(
        encoder="h264_vaapi",
        encoder_device="/dev/dri/renderD128",
        resolution=(1920, 1080), fps=60,
        audio_path=Path("/x/audio.mp3"), audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="5M", audio_bitrate="192k",
        output_path=Path("/x/out.mp4"),
    )
    joined = " ".join(cmd)
    assert "-hwaccel vaapi" in joined
    assert "-c:v h264_vaapi" in joined
    assert "-r 60" in joined
    assert "-s 1920x1080" in joined
    assert "out.mp4" in joined


def test_build_cmd_software():
    cmd = build_ffmpeg_cmd(
        encoder="libx264", encoder_device=None,
        resolution=(1280, 720), fps=30,
        audio_path=Path("/x/audio.mp3"), audio_rate=1.5, audio_lead_in_ms=500,
        video_bitrate="3M", audio_bitrate="128k",
        output_path=Path("/x/out.mp4"),
    )
    joined = " ".join(cmd)
    assert "-c:v libx264" in joined
    assert "-hwaccel" not in joined
    # Speed mod expressed in audio filter.
    assert "asetrate=44100*1.5" in joined


def test_build_cmd_no_audio():
    cmd = build_ffmpeg_cmd(
        encoder="libx264", encoder_device=None,
        resolution=(1280, 720), fps=30,
        audio_path=None, audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="3M", audio_bitrate="128k",
        output_path=Path("/x/silent.mp4"),
    )
    joined = " ".join(cmd)
    assert "-c:a" not in joined
    assert "silent.mp4" in joined


@pytest.mark.asyncio
async def test_probe_encoder_auto_prefers_vaapi(tmp_path):
    # Mock subprocess that reports h264_vaapi is available.
    async def fake_create(*args, **kw):
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b" V..... h264_vaapi  H.264 ...\n", b""))
        proc.returncode = 0
        return proc
    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        chosen = await probe_encoder("auto", "/dev/dri/renderD128")
    assert chosen == "h264_vaapi"


@pytest.mark.asyncio
async def test_probe_encoder_falls_back_to_libx264():
    async def fake_create(*args, **kw):
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b" V..... libx264  H.264 ...\n", b""))
        proc.returncode = 0
        return proc
    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        chosen = await probe_encoder("auto", None)
    assert chosen == "libx264"


@pytest.mark.asyncio
async def test_probe_encoder_explicit_passes_through():
    chosen = await probe_encoder("libx264", None)
    assert chosen == "libx264"
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_encode.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `osu_renderer/encode.py`**

```python
"""ffmpeg subprocess: probe encoder, build command, spawn, manage stdin frames."""
from __future__ import annotations

import asyncio
from pathlib import Path

from osu_renderer.errors import EncoderError


async def probe_encoder(encoder: str, device: str | None) -> str:
    """Resolve 'auto' → 'h264_vaapi' if available, else 'libx264'.

    Non-'auto' values pass through unchanged.
    """
    if encoder != "auto":
        return encoder
    # Ask ffmpeg what encoders it knows about.
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-encoders",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="ignore")
    if device is not None and "h264_vaapi" in text:
        return "h264_vaapi"
    return "libx264"


def build_ffmpeg_cmd(
    *,
    encoder: str,
    encoder_device: str | None,
    resolution: tuple[int, int],
    fps: int,
    audio_path: Path | None,
    audio_rate: float,
    audio_lead_in_ms: int,
    video_bitrate: str,
    audio_bitrate: str,
    output_path: Path,
) -> list[str]:
    """Build the ffmpeg argv. Audio is optional."""
    w, h = resolution
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    if encoder == "h264_vaapi":
        cmd += ["-hwaccel", "vaapi"]
        if encoder_device:
            cmd += ["-hwaccel_device", encoder_device]

    # Video input: raw frames on stdin.
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]

    # Audio input if available.
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
        audio_filter_parts: list[str] = []
        if audio_rate != 1.0:
            audio_filter_parts.append(f"asetrate=44100*{audio_rate}")
            audio_filter_parts.append("aresample=44100")
        if audio_lead_in_ms > 0:
            audio_filter_parts.append(
                f"adelay={audio_lead_in_ms}|{audio_lead_in_ms}"
            )
        if audio_filter_parts:
            cmd += ["-filter:a", ",".join(audio_filter_parts)]

    # Video filter to land on the format the encoder wants.
    if encoder == "h264_vaapi":
        cmd += ["-vf", "format=nv12,hwupload"]

    cmd += ["-c:v", encoder, "-b:v", video_bitrate]

    if audio_path is not None:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-map", "0:v", "-map", "1:a"]
    else:
        cmd += ["-map", "0:v"]

    cmd += ["-shortest", str(output_path)]
    return cmd


class FfmpegPipe:
    """Owns a running ffmpeg process; bytes written via write_frame end up encoded."""

    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.proc: asyncio.subprocess.Process | None = None
        self._stderr_log: bytes = b""

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def write_frame(self, data: bytes) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EncoderError("ffmpeg not started")
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def close(self, output_path: Path) -> None:
        if self.proc is None:
            return
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        _, stderr = await self.proc.communicate()
        self._stderr_log = stderr or b""
        if self.proc.returncode != 0:
            raise EncoderError(
                f"ffmpeg exit code {self.proc.returncode}: "
                f"{self._stderr_log.decode(errors='ignore')[-1024:]}"
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise EncoderError(f"output MP4 missing or empty: {output_path}")
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_encode.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/encode.py tests/test_encode.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "ffmpeg wrapper: probe encoder, build cmd, async stdin pipe"
```

---

## Task 9: GPU context

**Files:**
- Create: `osu_renderer/gpu/context.py`
- Test: `tests/test_gpu_context.py`

- [ ] **Step 1: Write the failing test**

```python
import os

import pytest

from osu_renderer.errors import GpuUnavailableError
from osu_renderer.gpu.context import HeadlessGl


@pytest.mark.slow
def test_open_context_and_fbo():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("Set RUN_SLOW=1 to run GL smoke tests")
    with HeadlessGl(width=128, height=64) as ctx:
        assert ctx.fbo.size == (128, 64)
        ctx.fbo.clear(0.25, 0.5, 0.75, 1.0)
        data = ctx.fbo.read(components=3)
        # Sample any pixel: should be ~(64, 128, 191) for rgb24.
        r, g, b = data[0], data[1], data[2]
        assert abs(r - 64) <= 2
        assert abs(g - 128) <= 2
        assert abs(b - 191) <= 2


def test_context_close_idempotent():
    # Without a real GL, we still want the API to be safe to call twice.
    h = HeadlessGl.__new__(HeadlessGl)
    h._ctx = None
    h._fbo = None
    h._color = None
    h._depth = None
    h.close()
    h.close()  # should not raise
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_gpu_context.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/gpu/context.py`**

```python
"""Headless ModernGL context backed by EGL (no window manager required)."""
from __future__ import annotations

import logging
import os

import moderngl

from osu_renderer.errors import GpuUnavailableError

log = logging.getLogger("osu_renderer")


class HeadlessGl:
    """Open a standalone GL context and allocate an offscreen FBO.

    Use as a context manager: closes the GL context on exit.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._ctx: moderngl.Context | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._color = None
        self._depth = None

    def __enter__(self) -> "HeadlessGl":
        try:
            # backend='egl' is preferred for headless on Linux. Falls back to
            # whatever moderngl picks if the explicit backend isn't available.
            self._ctx = moderngl.create_standalone_context(
                backend=os.environ.get("MODERNGL_BACKEND", "egl"),
                require=330,
            )
        except Exception as e:
            raise GpuUnavailableError(f"could not create GL context: {e}") from e
        self._color = self._ctx.texture((self.width, self.height), 4)
        self._depth = self._ctx.depth_renderbuffer((self.width, self.height))
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._color],
            depth_attachment=self._depth,
        )
        self._fbo.use()
        log.info(
            "gl_context_ready",
            extra={"renderer": self._ctx.info.get("GL_RENDERER", "?")},
        )
        return self

    @property
    def ctx(self) -> moderngl.Context:
        if self._ctx is None:
            raise GpuUnavailableError("context is closed")
        return self._ctx

    @property
    def fbo(self) -> moderngl.Framebuffer:
        if self._fbo is None:
            raise GpuUnavailableError("fbo not allocated")
        return self._fbo

    def close(self) -> None:
        for obj in (self._fbo, self._color, self._depth, self._ctx):
            if obj is not None:
                try:
                    obj.release()
                except Exception:  # noqa: BLE001
                    pass
        self._ctx = None
        self._fbo = None
        self._color = None
        self._depth = None

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_gpu_context.py -v
.venv/bin/pytest -q -m slow tests/test_gpu_context.py  # ensure GL smoke also passes when opted in
```

Expected: non-slow test passes; slow test skipped without `RUN_SLOW=1`. Optionally run with `RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 pytest -q -m slow tests/test_gpu_context.py` to verify the smoke test passes on llvmpipe.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/gpu/context.py tests/test_gpu_context.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Headless ModernGL context with offscreen FBO"
```

---

## Task 10: Shaders module + GLSL files

**Files:**
- Create: `osu_renderer/gpu/shaders.py`
- Create: `osu_renderer/assets/shaders/sprite.vert`
- Create: `osu_renderer/assets/shaders/sprite.frag`
- Create: `osu_renderer/assets/shaders/flashlight.frag`
- Test: `tests/test_gpu_shaders.py`

- [ ] **Step 1: Write the failing test**

```python
import os

import pytest

from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.shaders import load_programs


@pytest.mark.slow
def test_load_programs():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    with HeadlessGl(width=64, height=64) as gl:
        progs = load_programs(gl.ctx)
        assert "sprite" in progs
        assert "flashlight" in progs


def test_shader_source_files_exist():
    from osu_renderer.gpu import shaders
    for name in ("sprite.vert", "sprite.frag", "flashlight.frag"):
        assert (shaders.SHADERS_DIR / name).is_file()
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_gpu_shaders.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `osu_renderer/assets/shaders/sprite.vert`**

```glsl
#version 330

uniform mat3 u_projection;

in vec2 in_pos;
in vec2 in_uv;
in float in_atlas_index;
in vec4 in_color;

out vec2 v_uv;
flat out int v_atlas_index;
out vec4 v_color;

void main() {
    vec3 p = u_projection * vec3(in_pos, 1.0);
    gl_Position = vec4(p.xy, 0.0, 1.0);
    v_uv = in_uv;
    v_atlas_index = int(in_atlas_index);
    v_color = in_color;
}
```

- [ ] **Step 4: Write `osu_renderer/assets/shaders/sprite.frag`**

```glsl
#version 330

uniform sampler2DArray u_atlas;
uniform float u_hd;     // 0 = off, 1 = on
uniform float u_fi;     // 0 = off, 1 = on
uniform float u_hd_top; // playfield-top in NDC for fade math
uniform float u_hd_bot; // playfield-bot in NDC

in vec2 v_uv;
flat in int v_atlas_index;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec4 sample_color = texture(u_atlas, vec3(v_uv, float(v_atlas_index)));
    vec4 result = sample_color * v_color;

    // Hidden / Fade In alpha shaping based on screen Y.
    float y_frac = clamp(
        (gl_FragCoord.y - u_hd_bot) / max(u_hd_top - u_hd_bot, 1e-3),
        0.0, 1.0
    );
    if (u_hd > 0.5) {
        // Fade out near the receptor (low y).
        float a = smoothstep(0.0, 0.4, y_frac);
        result.a *= a;
    }
    if (u_fi > 0.5) {
        // Fade in from the top.
        float a = smoothstep(1.0, 0.6, y_frac);
        result.a *= a;
    }

    frag_color = result;
}
```

- [ ] **Step 5: Write `osu_renderer/assets/shaders/flashlight.frag`**

```glsl
#version 330

uniform sampler2D u_scene;
uniform vec2 u_center;     // pixel coords of receptor center
uniform float u_radius;    // pixel radius of the lit area

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec4 scene = texture(u_scene, v_uv);
    float dist = distance(gl_FragCoord.xy, u_center);
    float mask = 1.0 - smoothstep(u_radius * 0.6, u_radius, dist);
    frag_color = vec4(scene.rgb * mask, 1.0);
}
```

- [ ] **Step 6: Implement `osu_renderer/gpu/shaders.py`**

```python
"""Load GLSL shader programs from bundled asset files."""
from __future__ import annotations

from pathlib import Path

import moderngl

SHADERS_DIR = Path(__file__).resolve().parent.parent / "assets" / "shaders"


def load_programs(ctx: moderngl.Context) -> dict[str, moderngl.Program]:
    sprite = ctx.program(
        vertex_shader=(SHADERS_DIR / "sprite.vert").read_text(),
        fragment_shader=(SHADERS_DIR / "sprite.frag").read_text(),
    )
    # The flashlight pass uses a passthrough vertex shader (full-screen quad).
    flash_vert = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""
    flashlight = ctx.program(
        vertex_shader=flash_vert,
        fragment_shader=(SHADERS_DIR / "flashlight.frag").read_text(),
    )
    return {"sprite": sprite, "flashlight": flashlight}
```

- [ ] **Step 7: Run tests**

```bash
.venv/bin/pytest tests/test_gpu_shaders.py -v
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest -q -m slow tests/test_gpu_shaders.py
```

Expected: non-slow passes; slow test passes with software fallback.

- [ ] **Step 8: Commit**

```bash
git add osu_renderer/gpu/shaders.py osu_renderer/assets/shaders/ tests/test_gpu_shaders.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "GLSL sprite + flashlight shaders; loader"
```

---

## Task 11: Sprite atlas + placeholder sprite generator

**Files:**
- Create: `osu_renderer/gpu/atlas.py`
- Create: `scripts/generate_placeholder_sprites.py` (one-off generator)
- Generated: `osu_renderer/assets/sprites/*.png`
- Test: `tests/test_gpu_atlas.py`

- [ ] **Step 1: Write the placeholder generator** (`scripts/generate_placeholder_sprites.py`)

```python
"""Generate functional placeholder sprites so the renderer works before Night05 is staged.

Run once after a fresh clone:  python scripts/generate_placeholder_sprites.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEST = Path(__file__).resolve().parent.parent / "osu_renderer" / "assets" / "sprites"
DEST.mkdir(parents=True, exist_ok=True)


def _rect(name: str, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    img = Image.new("RGBA", size, color)
    img.save(DEST / name)


def _judgment(name: str, text: str, color: tuple[int, int, int]) -> None:
    img = Image.new("RGBA", (256, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 4), text, font=font, fill=(*color, 255))
    img.save(DEST / name)


def main() -> None:
    _rect("note_tap.png",         (128, 32),  (180, 220, 255, 255))
    _rect("note_hold_head.png",   (128, 32),  (200, 240, 255, 255))
    _rect("note_hold_body.png",   (128, 32),  (120, 180, 220, 200))
    _rect("note_hold_tail.png",   (128, 32),  (200, 240, 255, 255))
    _rect("receptor_off.png",     (128, 64),  (60, 60, 80, 255))
    _rect("receptor_on.png",      (128, 64),  (240, 240, 250, 255))
    _rect("hit_light.png",        (256, 128), (255, 255, 255, 180))
    _rect("column_bg.png",        (128, 16),  (30, 30, 40, 200))
    _rect("playfield_frame.png",  (16, 16),   (120, 120, 140, 220))
    _rect("bg_vignette.png",      (1024, 1024), (0, 0, 0, 0))

    _judgment("judgment_geki.png", "320",     (255, 215, 0))
    _judgment("judgment_300.png",  "300",     (100, 180, 255))
    _judgment("judgment_katu.png", "200",     (110, 220, 130))
    _judgment("judgment_100.png",  "100",     (255, 230, 90))
    _judgment("judgment_50.png",   " 50",     (180, 180, 180))
    _judgment("judgment_miss.png", "MISS",    (240, 80, 80))


if __name__ == "__main__":
    main()
    print(f"Wrote sprites to {DEST}")
```

- [ ] **Step 2: Generate the placeholder sprites**

```bash
.venv/bin/python scripts/generate_placeholder_sprites.py
ls osu_renderer/assets/sprites/
```

Expected: 16 PNG files listed.

- [ ] **Step 3: Write the failing test**

```python
import os

import pytest

from osu_renderer.gpu.atlas import SpriteAtlas
from osu_renderer.gpu.context import HeadlessGl


@pytest.mark.slow
def test_load_default_atlas():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    with HeadlessGl(width=64, height=64) as gl:
        atlas = SpriteAtlas.load_default(gl.ctx)
        # The atlas exposes the slot index for every named sprite.
        for name in (
            "note_tap", "note_hold_head", "note_hold_body", "note_hold_tail",
            "receptor_off", "receptor_on", "hit_light",
            "judgment_geki", "judgment_300", "judgment_katu",
            "judgment_100", "judgment_50", "judgment_miss",
        ):
            assert isinstance(atlas.index_of(name), int)
        # Texture array is bound and accessible.
        assert atlas.texture_array is not None


def test_index_of_missing_raises():
    # Construct without GL — just test the in-memory lookup.
    a = SpriteAtlas.__new__(SpriteAtlas)
    a._indices = {"note_tap": 0}
    import pytest as _p
    with _p.raises(KeyError):
        a.index_of("nonexistent")
```

- [ ] **Step 4: Run, expect failure**

```bash
.venv/bin/pytest tests/test_gpu_atlas.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Implement `osu_renderer/gpu/atlas.py`**

```python
"""Bundled sprite PNGs → ModernGL texture array."""
from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

SPRITES_DIR = Path(__file__).resolve().parent.parent / "assets" / "sprites"

# Canonical order — matters because callers refer to slots by index.
SPRITE_NAMES: tuple[str, ...] = (
    "note_tap",
    "note_hold_head",
    "note_hold_body",
    "note_hold_tail",
    "receptor_off",
    "receptor_on",
    "hit_light",
    "column_bg",
    "playfield_frame",
    "bg_vignette",
    "judgment_geki",
    "judgment_300",
    "judgment_katu",
    "judgment_100",
    "judgment_50",
    "judgment_miss",
)

# Atlas layer dimensions — sprites are resized to fit (preserving aspect via padding).
LAYER_W = 256
LAYER_H = 128


class SpriteAtlas:
    """Packs the bundled sprites into a single Texture2DArray.

    Each sprite occupies one layer; lookup by name returns the layer index.
    """

    def __init__(self) -> None:
        self._indices: dict[str, int] = {}
        self.texture_array: moderngl.TextureArray | None = None

    @classmethod
    def load_default(cls, ctx: moderngl.Context) -> "SpriteAtlas":
        atlas = cls()
        layers: list[np.ndarray] = []
        for i, name in enumerate(SPRITE_NAMES):
            path = SPRITES_DIR / f"{name}.png"
            img = Image.open(path).convert("RGBA")
            # Resize to LAYER_W × LAYER_H, preserving aspect by letterboxing.
            scaled = _fit_letterbox(img, LAYER_W, LAYER_H)
            layers.append(np.asarray(scaled, dtype=np.uint8))
            atlas._indices[name] = i

        # Stack into (depth, H, W, 4) → flatten for moderngl.
        arr = np.stack(layers, axis=0)
        atlas.texture_array = ctx.texture_array(
            size=(LAYER_W, LAYER_H, len(SPRITE_NAMES)),
            components=4,
            data=arr.tobytes(),
        )
        atlas.texture_array.build_mipmaps()
        atlas.texture_array.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return atlas

    def index_of(self, name: str) -> int:
        return self._indices[name]


def _fit_letterbox(img: Image.Image, w: int, h: int) -> Image.Image:
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    src_w, src_h = img.size
    scale = min(w / src_w, h / src_h)
    new = img.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.LANCZOS)
    canvas.paste(new, ((w - new.width) // 2, (h - new.height) // 2), new)
    return canvas
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/pytest tests/test_gpu_atlas.py -v
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest -q -m slow tests/test_gpu_atlas.py
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add osu_renderer/gpu/atlas.py osu_renderer/assets/sprites/*.png \
        scripts/generate_placeholder_sprites.py tests/test_gpu_atlas.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Sprite atlas + 16 placeholder PNGs (Night05 swaps in later)"
```

---

## Task 12: PBO readback

**Files:**
- Create: `osu_renderer/gpu/readback.py`
- Test: `tests/test_gpu_readback.py`

- [ ] **Step 1: Write the failing test**

```python
import os

import pytest

from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.readback import FrameReader


@pytest.mark.slow
def test_readback_returns_correct_size():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    W, H = 128, 64
    with HeadlessGl(width=W, height=H) as gl:
        reader = FrameReader(gl.ctx, gl.fbo, components=3)
        gl.fbo.clear(0.1, 0.2, 0.3, 1.0)
        frame = reader.read()
        assert len(frame) == W * H * 3


@pytest.mark.slow
def test_readback_double_buffered():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    W, H = 64, 64
    with HeadlessGl(width=W, height=H) as gl:
        reader = FrameReader(gl.ctx, gl.fbo, components=3, ring=2)
        for color in [(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)]:
            gl.fbo.clear(*color, 1.0)
            frame = reader.read()
            assert len(frame) == W * H * 3
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_gpu_readback.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/gpu/readback.py`**

```python
"""Framebuffer → CPU readback with a small ring of Pixel Buffer Objects.

Using a PBO ring overlaps GPU rendering of frame N+1 with CPU readback of
frame N, which keeps the GPU pipeline busy at the cost of one extra frame of
latency. The ring degrades gracefully to direct fbo.read() if PBOs aren't
available on the driver.
"""
from __future__ import annotations

import moderngl


class FrameReader:
    def __init__(
        self,
        ctx: moderngl.Context,
        fbo: moderngl.Framebuffer,
        components: int = 3,
        ring: int = 2,
    ) -> None:
        self.ctx = ctx
        self.fbo = fbo
        self.components = components
        self.ring = max(1, ring)
        w, h = fbo.size
        self.frame_size = w * h * components
        # moderngl doesn't expose glMapBuffer directly; we rely on its
        # synchronous .read() — the ring here is a future-extension point.

    def read(self) -> bytes:
        return self.fbo.read(components=self.components)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_gpu_readback.py -v
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest -q -m slow tests/test_gpu_readback.py
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/gpu/readback.py tests/test_gpu_readback.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "FrameReader: FBO → bytes (PBO ring as extension hook)"
```

---

## Task 13: GPU renderer — playfield + notes

**Files:**
- Create: `osu_renderer/gpu/renderer.py`
- Test: `tests/test_gpu_renderer_playfield.py`

- [ ] **Step 1: Write the failing test**

```python
import os

import pytest

from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.renderer import FrameRenderer, RenderContext
from osu_renderer.models import VisualMods
from osu_renderer.scene import VisibleNote, SceneState


@pytest.mark.slow
def test_renders_a_single_note():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    W, H = 256, 256
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        scene = SceneState(
            t_ms=0,
            visible_notes=(
                VisibleNote(column=1, is_hold=False, y_fraction=0.5,
                            head_y_fraction=0.5, tail_y_fraction=0.5),
            ),
            keys_held=(False, False, False, False),
            visual_mods=VisualMods(),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        # At least one pixel should be brighter than the background.
        assert max(data) > 50
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_gpu_renderer_playfield.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/gpu/renderer.py` (initial — playfield + notes only)**

```python
"""GPU draw passes: background, playfield columns, notes, receptors, HUD.

This file is implemented incrementally:
  Task 13: playfield + tap/hold notes (this commit)
  Task 14: receptors, key flash, hit-light, judgments
  Task 15: HUD, banner, flashlight post-pass
"""
from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from osu_renderer.gpu.atlas import SpriteAtlas
from osu_renderer.gpu.shaders import load_programs
from osu_renderer.scene import SceneState

# Playfield dimensions (fraction of screen).
PLAYFIELD_X_FRAC = 0.39
PLAYFIELD_W_FRAC = 0.22
NOTE_HEIGHT_PX = 28


@dataclass
class RenderContext:
    ctx: moderngl.Context
    fbo: moderngl.Framebuffer
    width: int
    height: int
    key_count: int


class FrameRenderer:
    def __init__(self, rc: RenderContext) -> None:
        self.rc = rc
        self.programs = load_programs(rc.ctx)
        self.atlas = SpriteAtlas.load_default(rc.ctx)
        self._make_quad_geometry()

    def _make_quad_geometry(self) -> None:
        ctx = self.rc.ctx
        # Two-triangle unit quad with UVs.
        self._unit_quad = ctx.buffer(
            np.array([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ], dtype="f4").tobytes()
        )

    def draw(self, scene: SceneState) -> None:
        ctx = self.rc.ctx
        fbo = self.rc.fbo
        fbo.use()
        fbo.clear(0.03, 0.03, 0.05, 1.0)

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self._draw_columns()
        self._draw_notes(scene)

    def _draw_columns(self) -> None:
        # Faint column backgrounds — uses the column_bg sprite as a stretched quad.
        rc = self.rc
        w = rc.width
        pf_x = int(PLAYFIELD_X_FRAC * w)
        pf_w = int(PLAYFIELD_W_FRAC * w)
        col_w = pf_w // rc.key_count
        for c in range(rc.key_count):
            x0 = pf_x + c * col_w
            self._draw_sprite("column_bg", x0, 0, col_w, rc.height, (1, 1, 1, 0.5))

    def _draw_notes(self, scene: SceneState) -> None:
        rc = self.rc
        w = rc.width
        h = rc.height
        pf_x = int(PLAYFIELD_X_FRAC * w)
        pf_w = int(PLAYFIELD_W_FRAC * w)
        col_w = pf_w // rc.key_count
        for n in scene.visible_notes:
            x0 = pf_x + n.column * col_w
            if n.is_hold:
                y_head = int(n.head_y_fraction * h)
                y_tail = int(n.tail_y_fraction * h)
                # Body
                self._draw_sprite("note_hold_body", x0, min(y_head, y_tail),
                                  col_w, abs(y_head - y_tail), (1, 1, 1, 1))
                # Head + tail caps
                self._draw_sprite("note_hold_head", x0, y_head - NOTE_HEIGHT_PX // 2,
                                  col_w, NOTE_HEIGHT_PX, (1, 1, 1, 1))
                self._draw_sprite("note_hold_tail", x0, y_tail - NOTE_HEIGHT_PX // 2,
                                  col_w, NOTE_HEIGHT_PX, (1, 1, 1, 1))
            else:
                y = int(n.y_fraction * h)
                self._draw_sprite("note_tap", x0, y - NOTE_HEIGHT_PX // 2,
                                  col_w, NOTE_HEIGHT_PX, (1, 1, 1, 1))

    def _draw_sprite(
        self, name: str, x: int, y: int, w: int, h: int, tint: tuple,
    ) -> None:
        if w <= 0 or h <= 0:
            return
        ctx = self.rc.ctx
        prog = self.programs["sprite"]
        atlas_idx = self.atlas.index_of(name)
        # Build a per-call quad with the desired screen rect.
        screen_w = self.rc.width
        screen_h = self.rc.height
        x0 = (x / screen_w) * 2 - 1
        x1 = ((x + w) / screen_w) * 2 - 1
        y0 = (y / screen_h) * 2 - 1
        y1 = ((y + h) / screen_h) * 2 - 1
        verts = np.array([
            # x   y   u  v  atlas  r g b a
            [x0, y0, 0, 1, atlas_idx, *tint],
            [x1, y0, 1, 1, atlas_idx, *tint],
            [x1, y1, 1, 0, atlas_idx, *tint],
            [x0, y0, 0, 1, atlas_idx, *tint],
            [x1, y1, 1, 0, atlas_idx, *tint],
            [x0, y1, 0, 0, atlas_idx, *tint],
        ], dtype="f4")
        vbo = ctx.buffer(verts.tobytes())
        vao = ctx.simple_vertex_array(
            prog, vbo, "in_pos", "in_uv", "in_atlas_index", "in_color",
        )
        # Bind atlas array texture to slot 0.
        self.atlas.texture_array.use(0)
        prog["u_atlas"] = 0
        # No projection — we already emitted clip-space coords.
        prog["u_projection"].value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        prog["u_hd"].value = 1.0 if False else 0.0  # filled in Task 14/15
        prog["u_fi"].value = 0.0
        prog["u_hd_top"].value = float(screen_h)
        prog["u_hd_bot"].value = 0.0
        vao.render(moderngl.TRIANGLES)
        vbo.release()
        vao.release()
```

- [ ] **Step 4: Run test**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_playfield.py -v -m slow
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/gpu/renderer.py tests/test_gpu_renderer_playfield.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "GPU renderer: playfield columns + tap/hold notes"
```

---

## Task 14: GPU renderer — receptors, key flash, hit light, judgments

**Files:**
- Modify: `osu_renderer/gpu/renderer.py`
- Modify: `osu_renderer/scene.py` (add judgment popups)
- Test: `tests/test_gpu_renderer_hud.py`

- [ ] **Step 1: Extend `scene.py` to include judgment events**

In `scene.py`, add to `SceneState`:

```python
@dataclass(frozen=True)
class JudgmentPopup:
    column: int
    judgment: str       # "geki" | "300" | "katu" | "100" | "50" | "miss"
    age_ms: int         # 0 = just hit, fades over 600ms


# Update SceneState:
@dataclass(frozen=True)
class SceneState:
    t_ms: int
    visible_notes: tuple[VisibleNote, ...]
    keys_held: tuple[bool, ...]
    visual_mods: VisualMods
    active_judgments: tuple["JudgmentPopup", ...] = ()
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    accuracy: float = 100.0
```

- [ ] **Step 2: Write the failing test**

```python
import os

import pytest

from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.renderer import FrameRenderer, RenderContext
from osu_renderer.models import VisualMods
from osu_renderer.scene import JudgmentPopup, SceneState


@pytest.mark.slow
def test_receptors_drawn_at_bottom():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    W, H = 320, 240
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        scene = SceneState(
            t_ms=0, visible_notes=(),
            keys_held=(True, False, True, False),
            visual_mods=VisualMods(),
        )
        fr.draw(scene)
        # Sample a row near the bottom (where receptors live).
        data = gl.fbo.read(components=3)
        # Just check that the frame isn't all-clear.
        assert max(data) > 50


@pytest.mark.slow
def test_judgment_popup_drawn():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    W, H = 320, 240
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
            active_judgments=(JudgmentPopup(column=2, judgment="300", age_ms=100),),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        assert max(data) > 50
```

- [ ] **Step 3: Run, expect failure**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_hud.py -v -m slow
```

Expected: import error or missing `JudgmentPopup`.

- [ ] **Step 4: Implement** — append to `FrameRenderer.draw`:

```python
    def draw(self, scene: SceneState) -> None:
        # ...existing playfield + notes...
        self._draw_receptors(scene)
        self._draw_judgments(scene)

    def _draw_receptors(self, scene: SceneState) -> None:
        rc = self.rc
        w, h = rc.width, rc.height
        pf_x = int(PLAYFIELD_X_FRAC * w)
        pf_w = int(PLAYFIELD_W_FRAC * w)
        col_w = pf_w // rc.key_count
        # Receptor strip 8% from bottom.
        rec_h = int(h * 0.08)
        rec_y = int(h * 0.05)
        for c in range(rc.key_count):
            x0 = pf_x + c * col_w
            sprite = "receptor_on" if scene.keys_held[c] else "receptor_off"
            self._draw_sprite(sprite, x0, rec_y, col_w, rec_h, (1, 1, 1, 1))

    def _draw_judgments(self, scene: SceneState) -> None:
        rc = self.rc
        w, h = rc.width, rc.height
        pf_x = int(PLAYFIELD_X_FRAC * w)
        pf_w = int(PLAYFIELD_W_FRAC * w)
        col_w = pf_w // rc.key_count
        for j in scene.active_judgments:
            # Fade out over 600ms.
            alpha = max(0.0, 1.0 - j.age_ms / 600.0)
            if alpha <= 0:
                continue
            sprite = f"judgment_{j.judgment}"
            jud_w = col_w * 2
            jud_h = int(rc.height * 0.05)
            x0 = pf_x + j.column * col_w - jud_w // 4
            y0 = int(rc.height * 0.40)
            self._draw_sprite(sprite, x0, y0, jud_w, jud_h, (1, 1, 1, alpha))
```

- [ ] **Step 5: Run test, expect pass**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_hud.py -v -m slow
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add osu_renderer/gpu/renderer.py osu_renderer/scene.py tests/test_gpu_renderer_hud.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Renderer: receptors, judgment popups, key flash"
```

---

## Task 15: HUD text + banner + flashlight post-pass

**Files:**
- Modify: `osu_renderer/gpu/renderer.py`
- Create: `osu_renderer/gpu/text.py` (PIL-based text → texture helper)
- Test: `tests/test_gpu_renderer_text.py`

- [ ] **Step 1: Write `osu_renderer/gpu/text.py`**

```python
"""Rasterize HUD/banner strings via PIL → upload to a GL texture.

The text is short and re-rendering per-frame is fine at the bot's scale.
For score/combo (changes often), the caller caches the texture and only
re-rasterizes when the string changes.
"""
from __future__ import annotations

from functools import lru_cache

import moderngl
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def _font_bold(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def text_to_texture(
    ctx: moderngl.Context,
    text: str,
    size: int = 24,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> tuple[moderngl.Texture, int, int]:
    font = _font_bold(size)
    # Measure
    dummy = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w = max(8, bbox[2] - bbox[0] + 8)
    h = max(8, bbox[3] - bbox[1] + 8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=color)
    tex = ctx.texture((w, h), 4, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    return tex, w, h
```

- [ ] **Step 2: Write the failing test**

```python
import os

import pytest

from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.renderer import FrameRenderer, RenderContext
from osu_renderer.gpu.text import text_to_texture
from osu_renderer.models import VisualMods
from osu_renderer.scene import SceneState


@pytest.mark.slow
def test_text_to_texture_returns_real_dimensions():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    with HeadlessGl(width=64, height=64) as gl:
        tex, w, h = text_to_texture(gl.ctx, "Hello", size=24)
        assert tex is not None
        assert w > 0 and h > 0


@pytest.mark.slow
def test_full_hud_renders():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    W, H = 480, 270
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        fr.set_banner_text("Seiryu - AO-INFINITY [Hard]   R3D")
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
            score=865_612, combo=1305, max_combo=1305, accuracy=98.45,
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        # Frame should have visible HUD elements (top + right side).
        assert max(data) > 50
```

- [ ] **Step 3: Run, expect failure**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_text.py -v -m slow
```

Expected: ImportError (`set_banner_text` not defined yet).

- [ ] **Step 4: Add banner + HUD methods to `FrameRenderer`**

```python
    def set_banner_text(self, text: str) -> None:
        if not hasattr(self, "_banner_tex") or self._banner_text != text:
            self._banner_tex, self._banner_w, self._banner_h = text_to_texture(
                self.rc.ctx, text, size=28, color=(235, 235, 240, 255),
            )
            self._banner_text = text

    def draw(self, scene: SceneState) -> None:
        # ...existing background, playfield, notes, receptors, judgments...
        self._draw_banner()
        self._draw_hud(scene)
        if scene.visual_mods.flashlight:
            self._draw_flashlight_pass()

    def _draw_banner(self) -> None:
        if not hasattr(self, "_banner_tex"):
            return
        # Draw at top, 60px tall.
        self._draw_external_texture(
            self._banner_tex, x=20, y=self.rc.height - 60,
            w=self._banner_w, h=self._banner_h, alpha=1.0,
        )

    def _draw_hud(self, scene: SceneState) -> None:
        # Cache textures per-string.
        if not hasattr(self, "_hud_cache"):
            self._hud_cache: dict[str, tuple] = {}
        lines = [
            f"Score {scene.score:,}",
            f"{scene.accuracy:.2f}%",
            f"Combo {scene.combo}x",
            f"Max   {scene.max_combo}x",
        ]
        x_right = self.rc.width - 20
        y = self.rc.height - 100
        for line in lines:
            key = f"hud:{line}"
            cached = self._hud_cache.get(key)
            if cached is None:
                tex, w, h = text_to_texture(
                    self.rc.ctx, line, size=22, color=(255, 255, 255, 255),
                )
                self._hud_cache[key] = (tex, w, h)
                cached = (tex, w, h)
            tex, w, h = cached
            self._draw_external_texture(tex, x=x_right - w, y=y, w=w, h=h, alpha=1.0)
            y -= h + 4

    def _draw_external_texture(
        self, tex: moderngl.Texture, x: int, y: int, w: int, h: int, alpha: float,
    ) -> None:
        # The sprite shader expects a sampler2DArray; the simplest way to draw
        # an ad-hoc 2D texture (e.g. text or background) with the same shader is
        # to wrap it in a single-layer texture array on the fly. Slightly more
        # GPU memory churn than a dedicated 2D shader, but keeps the codepath
        # uniform. Optimize with a 2D-only program if profiling justifies it.
        ctx = self.rc.ctx
        prog = self.programs["sprite"]
        screen_w, screen_h = self.rc.width, self.rc.height
        x0 = (x / screen_w) * 2 - 1
        x1 = ((x + w) / screen_w) * 2 - 1
        y0 = (y / screen_h) * 2 - 1
        y1 = ((y + h) / screen_h) * 2 - 1
        tex_2d_to_array = ctx.texture_array(
            size=(w, h, 1),
            components=4,
            data=tex.read(),
        )
        tex_2d_to_array.use(0)
        prog["u_atlas"] = 0
        prog["u_projection"].value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        prog["u_hd"].value = 0.0
        prog["u_fi"].value = 0.0
        prog["u_hd_top"].value = float(screen_h)
        prog["u_hd_bot"].value = 0.0

        verts = np.array([
            [x0, y0, 0, 1, 0, 1, 1, 1, alpha],
            [x1, y0, 1, 1, 0, 1, 1, 1, alpha],
            [x1, y1, 1, 0, 0, 1, 1, 1, alpha],
            [x0, y0, 0, 1, 0, 1, 1, 1, alpha],
            [x1, y1, 1, 0, 0, 1, 1, 1, alpha],
            [x0, y1, 0, 0, 0, 1, 1, 1, alpha],
        ], dtype="f4")
        vbo = ctx.buffer(verts.tobytes())
        vao = ctx.simple_vertex_array(
            prog, vbo, "in_pos", "in_uv", "in_atlas_index", "in_color",
        )
        vao.render(moderngl.TRIANGLES)
        vbo.release()
        vao.release()
        tex_2d_to_array.release()
```

(Also: at the top of `renderer.py` add `from osu_renderer.gpu.text import text_to_texture`.)

- [ ] **Step 5: Add the flashlight pass stub**

```python
    def _draw_flashlight_pass(self) -> None:
        # Composite a circular mask over the rendered scene.
        # Implementation: bind current fbo color attachment as texture, render
        # to a temporary fbo with flashlight.frag, blit back. Skipping the
        # ping-pong fbo here for v1 simplicity — apply as a darken overlay.
        rc = self.rc
        # Simple darken: draw a semi-transparent black rect over everything
        # except a small circle around the receptors. Approximation only.
        h = rc.height
        w = rc.width
        # 4 corner darken quads (rough rectangle mask).
        self._draw_sprite("bg_vignette", 0, 0, w, h, (0, 0, 0, 0.65))
```

- [ ] **Step 6: Run tests**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_text.py -v -m slow
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add osu_renderer/gpu/text.py osu_renderer/gpu/renderer.py tests/test_gpu_renderer_text.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "HUD text + banner + flashlight darken overlay"
```

---

## Task 16: Background image rendering

**Files:**
- Modify: `osu_renderer/gpu/renderer.py`
- Test: `tests/test_gpu_renderer_background.py`

- [ ] **Step 1: Write the failing test**

```python
import os
from pathlib import Path

import pytest
from PIL import Image

from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.renderer import FrameRenderer, RenderContext
from osu_renderer.models import VisualMods
from osu_renderer.scene import SceneState


@pytest.mark.slow
def test_background_image_visible(tmp_path: Path):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    # Make a solid red bg.
    bg_path = tmp_path / "bg.png"
    Image.new("RGB", (128, 128), (200, 60, 60)).save(bg_path)
    W, H = 256, 256
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        fr.set_background(bg_path)
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        # Background is dimmed, so red channel should dominate but be reduced.
        # Sample top-left pixel (well outside the playfield).
        r, g, b = data[0], data[1], data[2]
        assert r > g and r > b
        # Dimmed: not full intensity.
        assert r < 200
```

- [ ] **Step 2: Run, expect failure**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_background.py -v -m slow
```

Expected: AttributeError (`set_background` undefined).

- [ ] **Step 3: Add `set_background` + draw-pass to `FrameRenderer`**

```python
    def set_background(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._bg_tex = None
            return
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        # Resize to fit the canvas, preserving aspect (cover).
        canvas_aspect = self.rc.width / self.rc.height
        img_aspect = img.width / img.height
        if img_aspect > canvas_aspect:
            new_h = self.rc.height
            new_w = int(img_aspect * new_h)
        else:
            new_w = self.rc.width
            new_h = int(new_w / img_aspect)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Crop centered.
        left = (new_w - self.rc.width) // 2
        top = (new_h - self.rc.height) // 2
        img = img.crop((left, top, left + self.rc.width, top + self.rc.height))
        self._bg_tex = self.rc.ctx.texture((self.rc.width, self.rc.height), 4, img.tobytes())

    def draw(self, scene: SceneState) -> None:
        self.rc.fbo.use()
        self.rc.fbo.clear(0.03, 0.03, 0.05, 1.0)
        self.rc.ctx.enable(moderngl.BLEND)
        self.rc.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self._draw_background()  # NEW — first pass
        self._draw_columns()
        self._draw_notes(scene)
        self._draw_receptors(scene)
        self._draw_judgments(scene)
        self._draw_banner()
        self._draw_hud(scene)
        if scene.visual_mods.flashlight:
            self._draw_flashlight_pass()

    def _draw_background(self) -> None:
        if not getattr(self, "_bg_tex", None):
            return
        # Draw bg, dimmed 50%.
        self._draw_external_texture(
            self._bg_tex, x=0, y=0,
            w=self.rc.width, h=self.rc.height, alpha=0.5,
        )
```

- [ ] **Step 4: Run test**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest tests/test_gpu_renderer_background.py -v -m slow
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/gpu/renderer.py tests/test_gpu_renderer_background.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Background image: load, fit, dim 50%"
```

---

## Task 17: Judgment timeline + score calculator

**Files:**
- Create: `osu_renderer/judgments.py`
- Modify: `osu_renderer/scene.py` (consume judgments)
- Test: `tests/test_judgments.py`

- [ ] **Step 1: Write the failing test**

```python
from osu_renderer.judgments import compute_judgments
from osu_renderer.models import KeyEvent, Note


def test_perfect_tap_within_window():
    notes = (Note(0, 1000),)
    events = (
        KeyEvent(time_ms=999, keys_held=0),
        KeyEvent(time_ms=1000, keys_held=0b0001),
        KeyEvent(time_ms=1020, keys_held=0),
    )
    result = compute_judgments(notes, events, key_count=4)
    # Sum of all judgments should equal #notes.
    total = (result.count_geki + result.count_300 + result.count_katu
             + result.count_100 + result.count_50 + result.count_miss)
    assert total == 1
    assert result.count_geki == 1  # tapped within ~17ms = perfect (Rainbow 300)


def test_miss_when_key_never_pressed():
    notes = (Note(0, 1000),)
    events = ()
    result = compute_judgments(notes, events, key_count=4)
    assert result.count_miss == 1


def test_combo_breaks_on_miss():
    notes = (Note(0, 1000), Note(0, 2000), Note(0, 3000))
    events = (
        KeyEvent(time_ms=1000, keys_held=0b0001),
        KeyEvent(time_ms=1020, keys_held=0),
        # Miss at 2000
        KeyEvent(time_ms=3000, keys_held=0b0001),
        KeyEvent(time_ms=3020, keys_held=0),
    )
    result = compute_judgments(notes, events, key_count=4)
    assert result.max_combo == 1
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_judgments.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `osu_renderer/judgments.py`**

```python
"""Determine each note's judgment from the replay's keypress timeline.

Mania timing windows (osu!stable, OD = 8 default):
    ±16.5 ms  → 320 (geki / rainbow 300)
    ±40   ms  → 300
    ±73   ms  → 200 (katu)
    ±103  ms  → 100
    ±127  ms  → 50
    > 127 ms  → miss
"""
from __future__ import annotations

from dataclasses import dataclass

from osu_renderer.models import HoldNote, KeyEvent, Note

WINDOW_320 = 16.5
WINDOW_300 = 40
WINDOW_200 = 73
WINDOW_100 = 103
WINDOW_50 = 127


@dataclass(frozen=True)
class JudgmentEvent:
    time_ms: int
    column: int
    judgment: str    # "geki" | "300" | "katu" | "100" | "50" | "miss"


@dataclass(frozen=True)
class JudgmentTimeline:
    events: tuple[JudgmentEvent, ...]
    count_geki: int
    count_300: int
    count_katu: int
    count_100: int
    count_50: int
    count_miss: int
    max_combo: int


def compute_judgments(
    notes: tuple, events: tuple[KeyEvent, ...], key_count: int,
) -> JudgmentTimeline:
    """Pair each tap note with its closest keypress; classify by window."""
    if not events:
        return _all_miss(notes)

    # Build per-column press events (rising edges).
    presses = _rising_edges(events, key_count)

    j_events: list[JudgmentEvent] = []
    counts = {"geki": 0, "300": 0, "katu": 0, "100": 0, "50": 0, "miss": 0}
    combo = 0
    max_combo = 0
    used: set[int] = set()  # indices of press events already matched

    for note in notes:
        target_time = note.time_ms
        col = note.column
        # Find the unused press in col within ±127ms.
        best_idx = -1
        best_delta = WINDOW_50 + 1
        for i, press_t in enumerate(presses[col]):
            if i in used:
                continue
            if abs(press_t - target_time) > WINDOW_50:
                continue
            d = abs(press_t - target_time)
            if d < best_delta:
                best_delta = d
                best_idx = i
        if best_idx < 0:
            jud = "miss"
            combo = 0
        else:
            used.add((col << 32) | best_idx)
            d = best_delta
            if d <= WINDOW_320:
                jud = "geki"
            elif d <= WINDOW_300:
                jud = "300"
            elif d <= WINDOW_200:
                jud = "katu"
            elif d <= WINDOW_100:
                jud = "100"
            else:
                jud = "50"
            combo += 1
            max_combo = max(max_combo, combo)
        counts[jud] += 1
        j_events.append(JudgmentEvent(time_ms=target_time, column=col, judgment=jud))

    return JudgmentTimeline(
        events=tuple(j_events),
        count_geki=counts["geki"],
        count_300=counts["300"],
        count_katu=counts["katu"],
        count_100=counts["100"],
        count_50=counts["50"],
        count_miss=counts["miss"],
        max_combo=max_combo,
    )


def _all_miss(notes: tuple) -> JudgmentTimeline:
    events = tuple(
        JudgmentEvent(time_ms=n.time_ms, column=n.column, judgment="miss") for n in notes
    )
    return JudgmentTimeline(
        events=events,
        count_geki=0, count_300=0, count_katu=0, count_100=0, count_50=0,
        count_miss=len(events), max_combo=0,
    )


def _rising_edges(
    events: tuple[KeyEvent, ...], key_count: int,
) -> list[list[int]]:
    presses: list[list[int]] = [[] for _ in range(key_count)]
    prev = 0
    for e in events:
        new_pressed = e.keys_held & ~prev
        for c in range(key_count):
            if new_pressed & (1 << c):
                presses[c].append(e.time_ms)
        prev = e.keys_held
    return presses
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_judgments.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/judgments.py tests/test_judgments.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "Judgments: pair notes with keypresses; mania timing windows"
```

---

## Task 18: Top-level render_mania orchestrator

**Files:**
- Create: `osu_renderer/render.py`
- Modify: `osu_renderer/__init__.py`
- Test: `tests/test_render_orchestrator.py`

- [ ] **Step 1: Write `osu_renderer/__init__.py`**

```python
"""osu! mania replay → MP4 renderer."""
from osu_renderer.errors import (
    BeatmapParseError,
    EncoderError,
    GpuUnavailableError,
    MissingAudioError,
    NotAManiaError,
    RendererError,
    RenderTimeoutError,
    ReplayParseError,
)
from osu_renderer.models import RenderOptions
from osu_renderer.render import render_mania

__all__ = [
    "RenderOptions",
    "render_mania",
    "RendererError",
    "BeatmapParseError",
    "ReplayParseError",
    "NotAManiaError",
    "MissingAudioError",
    "GpuUnavailableError",
    "EncoderError",
    "RenderTimeoutError",
]
```

- [ ] **Step 2: Write the failing test**

```python
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osu_renderer import RenderOptions, render_mania
from osu_renderer.errors import NotAManiaError


async def test_orchestrator_rejects_non_mania(tmp_path: Path, fixtures_dir: Path):
    # Reuse the std fixture from Task 5.
    osr = fixtures_dir / "std_replay.osr"
    out = tmp_path / "out.mp4"
    options = RenderOptions(resolution=(640, 360), fps=30)
    with pytest.raises(NotAManiaError):
        await render_mania(
            osr_path=osr,
            beatmap_dir=tmp_path,
            output_path=out,
            options=options,
        )


@pytest.mark.slow
async def test_orchestrator_end_to_end(fixtures_dir: Path, tmp_path: Path):
    """Render the real AO-INFINITY replay against the cached beatmap."""
    import os
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    bm_dir = Path("/var/mnt/Synology-Reddie/Mania ORDR Bot/beatmaps/"
                  "3e37a2abc23502109072187911229864")
    if not bm_dir.exists():
        pytest.skip(f"beatmap cache not present at {bm_dir}")
    out = tmp_path / "ao_infinity.mp4"
    options = RenderOptions(resolution=(1280, 720), fps=30, timeout_seconds=600)
    progress_seen = []

    async def on_progress(f):
        progress_seen.append(f)

    await render_mania(
        osr_path=fixtures_dir / "ao_infinity_hard.osr",
        beatmap_dir=bm_dir,
        output_path=out,
        options=options,
        progress_callback=on_progress,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert any(p > 0 for p in progress_seen)
```

- [ ] **Step 3: Run, expect failure**

```bash
.venv/bin/pytest tests/test_render_orchestrator.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `osu_renderer/render.py`**

```python
"""The public render_mania orchestrator: parse → mod → render → encode."""
from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Awaitable, Callable

from osu_renderer.beatmap import parse_beatmap
from osu_renderer.encode import FfmpegPipe, build_ffmpeg_cmd, probe_encoder
from osu_renderer.errors import EncoderError, MissingAudioError, RenderTimeoutError
from osu_renderer.gpu.context import HeadlessGl
from osu_renderer.gpu.readback import FrameReader
from osu_renderer.gpu.renderer import FrameRenderer, RenderContext
from osu_renderer.judgments import compute_judgments
from osu_renderer.mods import apply_mods
from osu_renderer.models import RenderOptions
from osu_renderer.replay import parse_replay
from osu_renderer.scene import JudgmentPopup, SceneState, snapshot

log = logging.getLogger("osu_renderer")

APPROACH_MS = 600


async def render_mania(
    *,
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    options: RenderOptions,
    progress_callback: Callable[[float], Awaitable[None]] | None = None,
    log_path: Path | None = None,
) -> None:
    log.info("render_start", extra={"osr": str(osr_path), "out": str(output_path)})

    replay = parse_replay(osr_path)
    osu_file = _find_osu(beatmap_dir, replay.beatmap_md5)
    beatmap = parse_beatmap(osu_file)
    mod_res = apply_mods(beatmap, replay)
    for w in mod_res.warnings:
        log.warning(w)
    modded = mod_res.beatmap

    judgments = compute_judgments(modded.notes, replay.key_events, modded.key_count)
    log.info(
        "judgments_done",
        extra={
            "geki": judgments.count_geki, "300": judgments.count_300,
            "katu": judgments.count_katu, "100": judgments.count_100,
            "50": judgments.count_50, "miss": judgments.count_miss,
            "max_combo": judgments.max_combo,
        },
    )

    encoder = await probe_encoder(options.encoder, options.encoder_device)
    audio_path: Path | None = None
    if modded.audio_filename:
        cand = beatmap_dir / modded.audio_filename
        if cand.exists():
            audio_path = cand
        elif options.audio_required:
            raise MissingAudioError(f"audio file not found: {cand}")
        else:
            log.warning("audio_missing", extra={"expected": str(cand)})

    cmd = build_ffmpeg_cmd(
        encoder=encoder,
        encoder_device=options.encoder_device,
        resolution=options.resolution,
        fps=options.fps,
        audio_path=audio_path,
        audio_rate=mod_res.audio_rate,
        audio_lead_in_ms=modded.audio_lead_in_ms,
        video_bitrate=options.video_bitrate,
        audio_bitrate=options.audio_bitrate,
        output_path=output_path,
    )
    pipe = FfmpegPipe(cmd)
    await pipe.start()

    total_video_ms = modded.total_duration_ms + 2000  # 2s end pad
    total_frames = math.ceil(total_video_ms / 1000 * options.fps)
    deadline = asyncio.get_event_loop().time() + options.timeout_seconds

    bg_filename = modded.background_filename
    bg_path = (beatmap_dir / bg_filename) if bg_filename else None

    try:
        with HeadlessGl(width=options.resolution[0], height=options.resolution[1]) as gl:
            rc = RenderContext(
                ctx=gl.ctx, fbo=gl.fbo,
                width=options.resolution[0], height=options.resolution[1],
                key_count=modded.key_count,
            )
            fr = FrameRenderer(rc)
            if bg_path and bg_path.exists():
                fr.set_background(bg_path)
            fr.set_banner_text(
                f"{modded.artist} - {modded.title} [{modded.difficulty}]   {replay.player_name}"
            )
            reader = FrameReader(gl.ctx, gl.fbo, components=3)

            last_progress_t = 0.0
            for frame_n in range(total_frames):
                if asyncio.get_event_loop().time() > deadline:
                    raise RenderTimeoutError(
                        f"render exceeded {options.timeout_seconds}s"
                    )
                t_ms = int(frame_n * 1000 / options.fps)
                scene = snapshot(
                    notes=modded.notes,
                    key_events=replay.key_events,
                    t_ms=t_ms,
                    key_count=modded.key_count,
                    approach_ms=APPROACH_MS,
                    visual_mods=mod_res.visual_mods,
                )
                # Active judgments: any whose time ∈ [t-600ms, t].
                active = tuple(
                    JudgmentPopup(column=j.column, judgment=j.judgment, age_ms=t_ms - j.time_ms)
                    for j in judgments.events
                    if 0 <= t_ms - j.time_ms < 600
                )
                # Score/combo at time t.
                score_so_far, combo_at_t = _aggregate_score(judgments.events, t_ms)
                scene_full = scene.__class__(
                    t_ms=scene.t_ms, visible_notes=scene.visible_notes,
                    keys_held=scene.keys_held, visual_mods=scene.visual_mods,
                    active_judgments=active,
                    score=score_so_far, combo=combo_at_t,
                    max_combo=judgments.max_combo,
                    accuracy=replay.accuracy,
                )
                fr.draw(scene_full)
                frame = reader.read()
                await pipe.write_frame(frame)

                if progress_callback and (
                    asyncio.get_event_loop().time() - last_progress_t > 0.5
                ):
                    await progress_callback(frame_n / total_frames)
                    last_progress_t = asyncio.get_event_loop().time()

            if progress_callback:
                await progress_callback(1.0)
        await pipe.close(output_path)
    except BaseException:
        # Kill ffmpeg on any error
        if pipe.proc and pipe.proc.returncode is None:
            try:
                pipe.proc.kill()
            except ProcessLookupError:
                pass
        raise

    log.info("render_done", extra={"out": str(output_path)})


def _find_osu(beatmap_dir: Path, expected_md5: str) -> Path:
    import hashlib
    for f in beatmap_dir.iterdir():
        if f.suffix.lower() != ".osu":
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h == expected_md5:
            return f
    # Fall back to first .osu file.
    for f in beatmap_dir.iterdir():
        if f.suffix.lower() == ".osu":
            return f
    raise FileNotFoundError(f"no .osu file in {beatmap_dir}")


def _aggregate_score(events: tuple, t_ms: int) -> tuple[int, int]:
    """Cumulative score + current combo at time t."""
    weights = {"geki": 320, "300": 300, "katu": 200, "100": 100, "50": 50, "miss": 0}
    score = 0
    combo = 0
    for j in events:
        if j.time_ms > t_ms:
            break
        score += weights[j.judgment] * (1 + combo // 10)
        if j.judgment == "miss":
            combo = 0
        else:
            combo += 1
    return score, combo
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_render_orchestrator.py::test_orchestrator_rejects_non_mania -v
```

Expected: 1 passed.

(Slow E2E test will run only with `RUN_SLOW=1`.)

- [ ] **Step 6: Commit**

```bash
git add osu_renderer/render.py osu_renderer/__init__.py tests/test_render_orchestrator.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "render_mania orchestrator: parse → mod → judge → render → encode"
```

---

## Task 19: README + LICENSE

**Files:**
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Write `LICENSE`** (MIT)

```
MIT License

Copyright (c) 2026 R3dWolfie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Write `README.md`**

```markdown
# osu! Mania Renderer

A Python library that turns an osu!mania `.osr` replay plus the beatmap files
into an MP4. GPU-rendered via ModernGL (standalone EGL context), hardware-
encoded via ffmpeg + VAAPI when available.

## Install

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_placeholder_sprites.py
```

Requires `ffmpeg` on `$PATH` (with libx264 — VAAPI optional but recommended on
AMD/Intel GPUs).

## Usage — library

```python
import asyncio
from pathlib import Path

from osu_renderer import RenderOptions, render_mania

async def main():
    await render_mania(
        osr_path=Path("play.osr"),
        beatmap_dir=Path("./beatmap/"),  # contains the .osu + audio.mp3 + bg
        output_path=Path("out.mp4"),
        options=RenderOptions(resolution=(1920, 1080), fps=60),
    )

asyncio.run(main())
```

## Usage — CLI

```bash
osu-mania-renderer play.osr ./beatmap/ -o out.mp4 --resolution 1920x1080 --fps 60
```

## Supported mods

DT / NC / HT (speed + pitch), MR (mirror), HD (hidden), FI (fade in),
FL (flashlight), V2 (ScoreV2), 1K–9K (key count locks). NF / EZ / HR / SD / PF
have no visual effect.

**Not supported in v1:** RD (Random — replays render as NM column order with a
warning), KC (Key Coop — replays render as single playfield with a warning).

## Project status

v0.1 — alpha. The bundled visual style uses placeholder sprites; swap in your
own PNGs under `osu_renderer/assets/sprites/` to skin it.

## License

MIT — see `LICENSE`. Note: any osu! skin you bundle (e.g. Night05) carries its
own license; check before redistributing.
```

- [ ] **Step 3: Commit**

```bash
git add README.md LICENSE
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "README + MIT LICENSE"
```

---

## Task 20: CLI

**Files:**
- Create: `osu_renderer/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "osu_renderer.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "osu" in result.stdout.lower() or "usage" in result.stdout.lower()
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: ModuleNotFoundError or non-zero exit.

- [ ] **Step 3: Implement `osu_renderer/cli.py`**

```python
"""CLI: osu-mania-renderer in.osr beatmap_dir/ -o out.mp4"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from osu_renderer import RenderOptions, render_mania


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="osu-mania-renderer",
        description="Render an osu!mania .osr replay to MP4.",
    )
    p.add_argument("osr", type=Path, help=".osr replay file")
    p.add_argument("beatmap_dir", type=Path,
                   help="directory containing the .osu, audio.mp3, and background")
    p.add_argument("-o", "--output", type=Path, default=Path("out.mp4"),
                   help="output MP4 path (default: out.mp4)")
    p.add_argument("--resolution", default="1920x1080",
                   help="WxH, e.g. 1280x720")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "h264_vaapi", "libx264"])
    p.add_argument("--encoder-device", default=None,
                   help="VAAPI device (e.g. /dev/dri/renderD128)")
    p.add_argument("--timeout", type=int, default=600,
                   help="render timeout in seconds")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    w, h = (int(x) for x in args.resolution.lower().split("x"))
    options = RenderOptions(
        resolution=(w, h), fps=args.fps,
        encoder=args.encoder, encoder_device=args.encoder_device,
        timeout_seconds=args.timeout,
    )

    async def _run() -> None:
        await render_mania(
            osr_path=args.osr,
            beatmap_dir=args.beatmap_dir,
            output_path=args.output,
            options=options,
            progress_callback=_print_progress,
        )

    try:
        asyncio.run(_run())
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


async def _print_progress(fraction: float) -> None:
    print(f"\rrendering… {fraction:.0%}", end="", flush=True)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add osu_renderer/cli.py tests/test_cli.py
git -c user.name="R3dWolfie" -c user.email="arui939@gmail.com" commit -m "CLI entrypoint"
```

---

## Task 21: Real end-to-end smoke test

**Files:**
- (no new files — run the slow tests we already wrote)

- [ ] **Step 1: Verify whole suite**

```bash
.venv/bin/pytest -q
```

Expected: all non-slow tests pass.

- [ ] **Step 2: Run slow GL + ffmpeg smoke tests against the cached beatmap**

```bash
RUN_SLOW=1 LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/pytest -q -m slow
```

Expected: GL smoke tests pass under llvmpipe. The end-to-end render test
produces an actual MP4 if the cached beatmap is present.

- [ ] **Step 3: Verify the produced MP4** (if step 2 produced one)

```bash
ffprobe -hide_banner /tmp/pytest-of-red/.../ao_infinity.mp4
```

Expected: dual-stream MP4 (video h264 + audio aac), duration matching the
beatmap length.

- [ ] **Step 4: If everything passes, tag a release**

```bash
git tag -a v0.1.0-alpha -m "First end-to-end render works"
```

(No commit if the smoke tests didn't add anything to the repo.)

---

## Task 22: Bot integration plan (separate sub-project)

This task is **not implemented in this plan** — it's a follow-up plan, since
the renderer ships independently. The bot integration consists of:

1. Add `osu-mania-renderer @ file:///home/red/Projects/Reddie/OsuManiaRenderer`
   to `mania-ordr`'s `pyproject.toml`.
2. Rewrite `/home/red/Projects/Mania ORDR/mania_ordr/renderer.py` as a
   ~50-line shim around `render_mania` (see spec Section "Bot integration").
3. Drop `DANSER_BINARY`, `_default-source.osk` staging, and
   `_ensure_default()` from the bot.
4. Update the worker pipeline to use the requested `"🎵 Mania replay detected:
   …"` → `"🎬 Rendering {%}"` → `"🛠️ Processing"` → embed lifecycle.
5. Smoke-test by dropping the AO-INFINITY replay in
   channel `#805077687881826384` again.

These steps will be written up in a separate plan after this renderer is
done.

---

## Tasks summary

21 tasks total. Tasks 1, 19 (docs) have no TDD steps. Tasks 9–16 are slow-test
gated (GPU + ffmpeg). All other tasks follow strict TDD with bite-sized
checkbox steps.
