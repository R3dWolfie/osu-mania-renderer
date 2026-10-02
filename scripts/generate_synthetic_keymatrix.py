"""Generate deterministic, genuinely playable native Mania 1K..18K fixtures.

Run with the repository Python:
    python scripts/generate_synthetic_keymatrix.py --root /tmp/r3d-keymatrix
Add --render for both production paths (720p sweep + representative 1080p).
Optional real-fixture renders require all three explicit inputs:
--real-replay, --real-beatmap-dir, and --real-skin-dir. Encoding defaults to
--encoder auto; --encoder-device is forwarded only when explicitly supplied.
Outputs are isolated under --root; existing outputs are never replaced.
No generated binaries belong in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import struct
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.judgments import compute_judgments
from osu_mania_renderer_v2.beatmap.replay import parse_replay
from osu_mania_renderer_v2.gpu.legacy_stage_geometry import legacy_stage_topology

REPRESENTATIVE = (4, 7, 10, 11, 12, 13, 18)


def objects(keys: int) -> tuple[tuple[int, int, int], ...]:
    """(column, press, release): taps, overlapping staggered holds, chords."""
    if not 1 <= keys <= 18:
        raise ValueError("keys must be in 1..18")
    result = [(c, 1800 + c * 120, 1840 + c * 120) for c in range(keys)]
    result += [(c, 4200 + c * 90, 4800 + c * 90) for c in range(keys)]
    topology = legacy_stage_topology(keys)
    boundary = (topology.stages[0].end_column - 1, topology.stages[-1].first_column)
    result += [(c, 6800, 6840) for c in sorted(set(boundary))]
    centres = {
        s.first_column + (s.special_column if s.special_column is not None else s.column_count // 2)
        for s in topology.stages
    }
    result += [(c, 7400, 7440) for c in sorted(centres)]
    result += [(c, 8000, 8040) for c in range(keys)]
    return tuple(sorted(result, key=lambda o: (o[1], o[0])))


def beatmap_bytes(keys: int) -> bytes:
    lines = []
    for column, press, release in objects(keys):
        x = int((column + 0.5) * 512 / keys)
        if release - press > 40:
            lines.append(f"{x},192,{press},128,0,{release}:0:0:0:0:")
        else:
            lines.append(f"{x},192,{press},1,0,0:0:0:0:")
    text = (
        f"""osu file format v14

[General]
AudioFilename: silence.wav
AudioLeadIn: 0
Mode: 3

[Metadata]
Title: Synthetic lane alignment matrix
Artist: R3D diagnostics
Creator: deterministic generator
Version: {keys:02d}K
BeatmapID: 0
BeatmapSetID: -1

[Difficulty]
HPDrainRate: 5
CircleSize: {keys}
OverallDifficulty: 5
ApproachRate: 5
SliderMultiplier: 1.4
SliderTickRate: 1

[TimingPoints]
0,500,4,1,0,50,1,0

[HitObjects]
"""
        + "\n".join(lines)
        + "\n"
    )
    return text.encode("utf-8")


def _uleb_string(value: str) -> bytes:
    data = value.encode("utf-8")
    result = bytearray([0x0B])
    length = len(data)
    while True:
        byte = length & 0x7F
        length >>= 7
        result.append(byte | (0x80 if length else 0))
        if not length:
            return bytes(result) + data


def replay_bytes(keys: int, beatmap_md5: str) -> bytes:
    changes: dict[int, list[tuple[int, bool]]] = {0: [], 9000: []}
    for column, press, release in objects(keys):
        changes.setdefault(press, []).append((column, True))
        changes.setdefault(release, []).append((column, False))
    frames, previous, mask = [], 0, 0
    for time, transitions in sorted(changes.items()):
        for column, down in transitions:
            if down:
                mask |= 1 << column
            else:
                mask &= ~(1 << column)
        frames.append(f"{time - previous}|{mask}|0|0,")
        previous = time
    frames.append("-12345|0|0|0,")
    raw = "".join(frames).encode("ascii")
    blob = lzma.compress(raw, format=lzma.FORMAT_ALONE)
    # ScoreV2 explicitly records heads AND tails. The raw simulator must
    # independently prove every event is MAX before any header reconciliation.
    scoring_count = len(objects(keys)) + keys
    result = bytearray(struct.pack("<Bi", 3, 20260101))
    for value in (beatmap_md5, "Synthetic keymatrix", hashlib.md5(raw).hexdigest()):
        result += _uleb_string(value)
    result += struct.pack(
        "<6hihbi", 0, 0, 0, scoring_count, 0, 0, 1000000, scoring_count, 1, 1 << 29
    )
    result += _uleb_string("")
    result += struct.pack("<qi", 0, len(blob)) + blob + struct.pack("<q", 0)
    return bytes(result)


def write_new(path: Path, data: bytes) -> None:
    """Idempotent generation is allowed, but never replace different data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def make_skin(root: Path) -> Path:
    skin = root / "skin"
    skin.mkdir(parents=True, exist_ok=True)
    settings = [
        "[General]",
        "Name: Synthetic lane geometry",
        "Author: R3D diagnostics",
        "Version: 2.7",
    ]
    for keys in range(1, 19):
        layout = legacy_stage_topology(keys)
        boundary = {s.first_column for s in layout.stages} | {
            s.end_column - 1 for s in layout.stages
        }
        settings += [
            "",
            "[Mania]",
            f"Keys:{keys}",
            "HitPosition:402",
            "LightPosition:413",
            "KeysUnderNotes:1",
            "NoteBodyStyle:0",
            "ColourColumnLine:80,80,80,255",
            "ColumnLineWidth:" + ",".join(["2"] * (keys + 1)),
        ]
        for c in range(keys):
            colour = (
                (255, 200, 40)
                if layout.column_kind(c) == "S"
                else ((255, 80, 120) if c in boundary else (40, 190, 220))
            )
            stem = f"k{keys:02d}-c{c:02d}"
            for role, size in (
                ("key", (60, 180)),
                ("keyD", (60, 180)),
                ("note", (60, 24)),
                ("body", (60, 40)),
                ("tail", (60, 24)),
            ):
                path = skin / f"{stem}-{role}.png"
                image = Image.new("RGBA", size, (*colour, 100 if role == "key" else 220))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(*colour, 255), width=2)
                draw.line((29, 0, 29, size[1]), fill=(255, 255, 255, 255), width=2)
                if role.startswith("key"):
                    draw.text((4, 4), str(c + 1), fill="white")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                write_new(path, buffer.getvalue())
            settings += [
                f"KeyImage{c}:{stem}-key",
                f"KeyImage{c}D:{stem}-keyD",
                f"NoteImage{c}:{stem}-note",
                f"NoteImage{c}H:{stem}-note",
                f"NoteImage{c}L:{stem}-body",
                f"NoteImage{c}T:{stem}-tail",
                f"Colour{c + 1}:15,20,30,255",
            ]
    write_new(skin / "skin.ini", ("\n".join(settings) + "\n").encode())
    for name, size, colour in (
        ("mania-stage-left", (4, 768), (60, 255, 60, 255)),
        ("mania-stage-right", (4, 768), (255, 60, 255, 255)),
        ("mania-stage-hint", (60, 2), (255, 255, 255, 255)),
        ("mania-stage-bottom", (1, 1), (0, 0, 0, 0)),
        ("mania-stage-light", (60, 200), (30, 40, 60, 80)),
        ("lightingN", (60, 100), (0, 0, 0, 0)),
        ("lightingL", (60, 100), (0, 0, 0, 0)),
    ):
        image = Image.new("RGBA", size, colour)
        if name.startswith("lighting"):
            ImageDraw.Draw(image).rectangle((28, 0, 31, 99), fill=(100, 255, 100, 180))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        write_new(skin / f"{name}.png", buffer.getvalue())
    return skin


def generate(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for subdir in (
        "comparisons",
        "logs",
        "renders/monolithic",
        "renders/wiki",
        "renders/real-fixture",
    ):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    skin = make_skin(root)
    manifest = {
        "generator": "scripts/generate_synthetic_keymatrix.py",
        "skin": str(skin),
        "fixtures": [],
    }
    for keys in range(1, 19):
        folder = root / "fixtures" / f"{keys:02d}K"
        osu = folder / f"synthetic-{keys:02d}k.osu"
        osr = folder / f"synthetic-{keys:02d}k.osr"
        data = beatmap_bytes(keys)
        md5 = hashlib.md5(data).hexdigest()
        write_new(osu, data)
        write_new(osr, replay_bytes(keys, md5))
        audio = folder / "silence.wav"
        if not audio.exists():
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                stream.writeframes(bytes(8000 * 2 * 10))
        beatmap, replay = parse_beatmap(osu), parse_replay(osr)
        timeline = compute_judgments(beatmap.notes, replay.key_events, keys, overall_difficulty=5)
        assert replay.beatmap_md5 == md5
        assert all(e.judgment == "geki" and e.hit_offset_ms == 0 for e in timeline.events)
        assert timeline.count_geki == replay.count_geki
        manifest["fixtures"].append(
            {
                "keys": keys,
                "osu": str(osu),
                "osr": str(osr),
                "md5": md5,
                "objects": len(beatmap.notes),
                "raw_MAX_events": timeline.count_geki,
                "raw_misses": timeline.count_miss,
                "final_chord_ms": 8000,
            }
        )
    write_new(root / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def render_matrix(
    root: Path,
    manifest: dict,
    *,
    real_replay: Path | None = None,
    real_beatmap_dir: Path | None = None,
    real_skin_dir: Path | None = None,
    encoder: str = "auto",
    encoder_device: str | None = None,
) -> None:
    jobs = []
    for fixture in manifest["fixtures"]:
        keys = fixture["keys"]
        for resolution in ("1280x720", "1920x1080") if keys in REPRESENTATIVE else ("1280x720",):
            for mode in ("monolithic", "wiki"):
                jobs.append(
                    (
                        mode,
                        fixture["osr"],
                        str(Path(fixture["osu"]).parent),
                        manifest["skin"],
                        resolution,
                        root / "renders" / mode / f"{keys:02d}K-{resolution}.mp4",
                        True,
                    )
                )
    if real_replay is not None:
        for mode in ("monolithic", "wiki"):
            jobs.append(
                (
                    mode,
                    str(real_replay),
                    str(real_beatmap_dir),
                    str(real_skin_dir),
                    "1920x1080",
                    root / "renders/real-fixture" / f"real-fixture-{mode}.mp4",
                    False,
                )
            )

    def run(job):
        mode, replay, beatmaps, skin, resolution, output, synthetic = job
        if output.exists():
            raise FileExistsError(f"Refusing to replace render {output}")
        command = [
            sys.executable,
            "-m",
            "osu_mania_renderer_v2.cli",
            replay,
            beatmaps,
            "--skin-dir",
            skin,
            "-o",
            str(output),
            "--resolution",
            resolution,
            "--fps",
            "60",
            "--encoder",
            encoder,
            "--timeout",
            "1200",
        ]
        if encoder_device is not None:
            command += ["--encoder-device", encoder_device]
        if synthetic:
            command += [
                "--no-result-screen",
                "--no-loudnorm",
                "--no-replay-hitsounds",
                "--no-combo-break",
                "--no-score",
                "--no-mods",
                "--no-hp-bar",
                "--no-hit-error",
                "--no-key-counter",
                "--no-progress",
                "--no-ur",
            ]
        env = dict(os.environ)
        env["OSU_USE_WIKI_RENDERER"] = "1" if mode == "wiki" else "0"
        log = root / "logs" / f"{mode}-{output.stem}.log"
        with log.open("x") as stream:
            result = subprocess.run(
                command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"Render failed; see {log}")
        print(f"Rendered {output}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, jobs))


def contact_sheets(root: Path) -> None:
    def frame(video, width=640):
        data = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                "4.5",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:-1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return image.crop((0, image.height - int(width * 0.33), width, image.height))

    for mode in ("monolithic", "wiki"):
        sheet = Image.new("RGB", (1920, 6 * 235), "#101018")
        draw = ImageDraw.Draw(sheet)
        for index, keys in enumerate(range(1, 19)):
            image = frame(root / "renders" / mode / f"{keys:02d}K-1280x720.mp4")
            x, y = index % 3 * 640, index // 3 * 235
            draw.text(
                (x + 8, y + 5), f"{keys:02d}K {mode} (4.5s; receptor / hold centres)", fill="white"
            )
            sheet.paste(image, (x, y + 22))
        path = root / "comparisons" / f"01K-18K-{mode}.png"
        if path.exists():
            raise FileExistsError(path)
        sheet.save(path)
    sheet = Image.new("RGB", (1920, len(REPRESENTATIVE) * 340), "#101018")
    draw = ImageDraw.Draw(sheet)
    for row, keys in enumerate(REPRESENTATIVE):
        for col, mode in enumerate(("monolithic", "wiki")):
            draw.text((col * 960 + 8, row * 340 + 5), f"{keys:02d}K {mode} 1920x1080", fill="white")
            sheet.paste(
                frame(root / "renders" / mode / f"{keys:02d}K-1920x1080.mp4", 960),
                (col * 960, row * 340 + 22),
            )
    path = root / "comparisons" / "representative-monolithic-vs-wiki.png"
    if path.exists():
        raise FileExistsError(path)
    sheet.save(path)


def geometry_report(root: Path, label: str = "after") -> None:
    """Print/write all source coordinates and the actual hit-light rect centre.

    This probe needs no GL. Importing FrameRenderer before this module allows
    running it against a preserved package snapshot for before/after evidence.
    """
    from types import SimpleNamespace

    from osu_mania_renderer_v2.beatmap.skin_ini import parse_skin_ini
    from osu_mania_renderer_v2.gpu.renderer import FrameRenderer

    ini = parse_skin_ini(root / "skin")
    results, lines = [], []
    for keys in range(1, 19):
        fr = object.__new__(FrameRenderer)
        fr.rc = SimpleNamespace(width=1920, height=1080, key_count=keys)
        fr.mania_section = ini.mania_for_keycount(keys)
        fr.skin_ini = ini
        fr._is_argon_default = lambda: False
        fr.atlas = SimpleNamespace(global_native_size=lambda _: (60, 100))
        fr._compute_playfield_geometry()
        layout = fr.stage_layout
        rows = []
        lines += [
            f"\n{keys:02d}K @1920x1080",
            "global stage local       x   width  centre stage_x stage_w stage_right light_centre",
        ]
        for c in range(keys):
            stage, local = layout.locate(c)
            x, w = fr.col_x[c], fr.col_w[c]
            light = fr._legacy_lighting_rect("lighting_n", c=c, x0=x, cw=w, centre_y=100)
            center = light[0] + light[2] / 2
            row = dict(
                global_column=c,
                stage=stage.index,
                local_column=local,
                x=x,
                width=w,
                center=x + w / 2,
                stage_x=stage.x,
                stage_width=stage.width,
                stage_right=stage.right,
                light_center=center,
                light_center_error=center - (x + w / 2),
            )
            rows.append(row)
            lines.append(
                f"{c:6d} {stage.index:5d} {local:5d} {x:7.3f} {w:7.3f} {x + w / 2:7.3f} {stage.x:7.3f} {stage.width:7.3f} {stage.right:10.3f} {center:12.3f}"
            )
        gap = None
        if len(layout.stages) == 2:
            left, right = layout.stages
            gap = dict(
                stage_one_right=left.right,
                stage_two_left=right.x,
                visible_column_gap=fr.col_x[right.first_column] - left.right,
                stage_separation=layout.separation,
                boundary_column_spacing=fr.col_x[right.first_column] - right.x,
            )
            lines.append("gap: " + json.dumps(gap, sort_keys=True))
        results.append(dict(keys=keys, columns=rows, gap=gap))
    write_new(
        root / "logs" / f"geometry-{label}.json", (json.dumps(results, indent=2) + "\n").encode()
    )
    write_new(root / "logs" / f"geometry-{label}.txt", ("\n".join(lines) + "\n").encode())
    print(f"Wrote 1K..18K geometry-{label} probes", flush=True)


def verify_outputs(root: Path, *, encoder: str = "auto", encoder_device: str | None = None) -> dict:
    """Probe expected videos and record requested encoder/device settings."""
    expected = []
    for keys in range(1, 19):
        for resolution in ((1280, 720), (1920, 1080)) if keys in REPRESENTATIVE else ((1280, 720),):
            for mode in ("monolithic", "wiki"):
                expected.append(
                    (
                        root
                        / "renders"
                        / mode
                        / f"{keys:02d}K-{resolution[0]}x{resolution[1]}.mp4",
                        resolution,
                    )
                )
    for mode in ("monolithic", "wiki"):
        video = root / "renders/real-fixture" / f"real-fixture-{mode}.mp4"
        if video.exists():
            expected.append((video, (1920, 1080)))
    result = {
        "fps": 60,
        "requested_encoder": encoder,
        "device": encoder_device,
        "videos": [],
    }
    for video, resolution in expected:
        data = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        assert (data["width"], data["height"]) == resolution
        assert data["r_frame_rate"] == "60/1" and data["codec_name"] == "h264"
        assert int(data["nb_frames"]) > 0
        result["videos"].append(
            {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), **data}
        )
    write_new(root / "render-manifest.json", (json.dumps(result, indent=2) + "\n").encode())
    print(f"Verified {len(expected)} H.264 60fps videos", flush=True)
    return result


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-replay", type=Path)
    parser.add_argument("--real-beatmap-dir", type=Path)
    parser.add_argument("--real-skin-dir", type=Path)
    parser.add_argument("--encoder", default="auto")
    parser.add_argument("--encoder-device")
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    real_inputs = (args.real_replay, args.real_beatmap_dir, args.real_skin_dir)
    if any(value is not None for value in real_inputs) and not all(
        value is not None for value in real_inputs
    ):
        parser.error(
            "--real-replay, --real-beatmap-dir, and --real-skin-dir must be supplied together"
        )
    manifest = generate(args.root)
    print(
        f"Validated {len(manifest['fixtures'])} genuine zero-offset replay fixtures in {args.root}",
        flush=True,
    )
    if args.render:
        render_matrix(
            args.root,
            manifest,
            real_replay=args.real_replay,
            real_beatmap_dir=args.real_beatmap_dir,
            real_skin_dir=args.real_skin_dir,
            encoder=args.encoder,
            encoder_device=args.encoder_device,
        )
    if args.contact_sheets:
        contact_sheets(args.root)
    if args.geometry:
        geometry_report(args.root)
    if args.verify or args.render:
        verify_outputs(args.root, encoder=args.encoder, encoder_device=args.encoder_device)


if __name__ == "__main__":
    main()
