"""Portable keymatrix CLI configuration; never invoke the actual renderer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import generate_synthetic_keymatrix as keymatrix


REAL_OPTIONS = ("--real-replay", "--real-beatmap-dir", "--real-skin-dir")


@pytest.mark.parametrize("mask", range(1, 7))
def test_incomplete_real_group_fails_before_generation(tmp_path, monkeypatch, capsys, mask):
    root = tmp_path / "not-created"
    monkeypatch.setattr(keymatrix, "generate", lambda _: pytest.fail("generation must not start"))
    argv = ["--root", str(root)]
    for index, option in enumerate(REAL_OPTIONS):
        if mask & (1 << index):
            argv += [option, str(tmp_path / f"missing-input-{index}")]
    with pytest.raises(SystemExit) as error:
        keymatrix.main(argv)
    assert error.value.code == 2
    assert "must be supplied together" in capsys.readouterr().err
    assert not root.exists()


def test_root_required_and_help_needs_no_generation(monkeypatch, capsys):
    monkeypatch.setattr(keymatrix, "generate", lambda _: pytest.fail("generation must not start"))
    with pytest.raises(SystemExit) as error:
        keymatrix.main([])
    assert error.value.code == 2
    assert "required: --root" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        keymatrix.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for option in (*REAL_OPTIONS, "--root", "--encoder", "--encoder-device"):
        assert option in help_text


@pytest.mark.parametrize("real_fixture", [False, True])
@pytest.mark.parametrize("encoder,device", [("auto", None), ("h264_vaapi", "explicit-device")])
def test_main_forwards_explicit_configuration(tmp_path, monkeypatch, real_fixture, encoder, device):
    manifest = {"fixtures": []}
    generated, rendered, verified = [], [], []
    monkeypatch.setattr(keymatrix, "generate", lambda root: generated.append(root) or manifest)
    monkeypatch.setattr(
        keymatrix, "render_matrix", lambda *args, **kwargs: rendered.append((args, kwargs))
    )
    monkeypatch.setattr(
        keymatrix, "verify_outputs", lambda *args, **kwargs: verified.append((args, kwargs))
    )
    argv = ["--root", str(tmp_path), "--render"]
    if encoder != "auto":
        argv += ["--encoder", encoder]
    if device is not None:
        argv += ["--encoder-device", device]
    inputs = [tmp_path / name for name in ("input.osr", "beatmap", "skin")]
    if real_fixture:
        for option, path in zip(REAL_OPTIONS, inputs):
            argv += [option, str(path)]
    keymatrix.main(argv)
    assert generated == [tmp_path]
    assert rendered == [
        (
            (tmp_path, manifest),
            dict(
                real_replay=inputs[0] if real_fixture else None,
                real_beatmap_dir=inputs[1] if real_fixture else None,
                real_skin_dir=inputs[2] if real_fixture else None,
                encoder=encoder,
                encoder_device=device,
            ),
        )
    ]
    assert verified == [((tmp_path,), dict(encoder=encoder, encoder_device=device))]


@pytest.mark.parametrize("real_fixture", [False, True])
@pytest.mark.parametrize("settings", [{}, {"encoder": "h264_vaapi", "encoder_device": "device"}])
def test_render_commands_are_portable(tmp_path, monkeypatch, real_fixture, settings):
    (tmp_path / "logs").mkdir()
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs["env"]["OSU_USE_WIKI_RENDERER"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(keymatrix.subprocess, "run", fake_run)
    manifest = dict(
        skin="synthetic-skin",
        fixtures=[dict(keys=1, osr="synthetic.osr", osu="beatmap/map.osu")],
    )
    real_inputs = (
        dict(
            real_replay=tmp_path / "real.osr",
            real_beatmap_dir=tmp_path / "real-map",
            real_skin_dir=tmp_path / "real-skin",
        )
        if real_fixture
        else {}
    )
    keymatrix.render_matrix(tmp_path, manifest, **real_inputs, **settings)
    assert len(commands) == (4 if real_fixture else 2)
    for command, wiki in commands:
        assert command[command.index("--encoder") + 1] == settings.get("encoder", "auto")
        if "encoder_device" in settings:
            assert command[command.index("--encoder-device") + 1] == settings["encoder_device"]
        else:
            assert "--encoder-device" not in command
        output = Path(command[command.index("-o") + 1])
        mode = "wiki" if wiki == "1" else "monolithic"
        if command[3] == str(real_inputs.get("real_replay")):
            assert command[4] == str(real_inputs["real_beatmap_dir"])
            assert command[command.index("--skin-dir") + 1] == str(real_inputs["real_skin_dir"])
            assert output == tmp_path / "renders/real-fixture" / f"real-fixture-{mode}.mp4"
        else:
            assert command[3:5] == ["synthetic.osr", "beatmap"]
            assert output == tmp_path / "renders" / mode / "01K-1280x720.mp4"


@pytest.mark.parametrize("settings", [{}, {"encoder": "h264_vaapi", "encoder_device": "device"}])
def test_verification_records_requested_settings_and_generic_real_names(
    tmp_path, monkeypatch, settings
):
    videos = []
    for keys in range(1, 19):
        sizes = ("1280x720", "1920x1080") if keys in keymatrix.REPRESENTATIVE else ("1280x720",)
        for size in sizes:
            for mode in ("monolithic", "wiki"):
                videos.append(tmp_path / "renders" / mode / f"{keys:02d}K-{size}.mp4")
    for mode in ("monolithic", "wiki"):
        videos.append(tmp_path / "renders/real-fixture" / f"real-fixture-{mode}.mp4")
    for video in videos:
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"mock video; ffprobe is stubbed")

    def fake_probe(command):
        video = Path(command[-1])
        assert command[0] == "ffprobe" and video in videos
        width, height = (
            (1920, 1080)
            if "1920x1080" in video.name or "real-fixture" in video.name
            else (1280, 720)
        )
        return json.dumps(
            dict(
                streams=[
                    dict(
                        codec_name="h264",
                        width=width,
                        height=height,
                        r_frame_rate="60/1",
                        nb_frames="60",
                    )
                ]
            )
        ).encode()

    monkeypatch.setattr(keymatrix.subprocess, "check_output", fake_probe)
    result = keymatrix.verify_outputs(tmp_path, **settings)
    assert result["requested_encoder"] == settings.get("encoder", "auto")
    assert result["device"] == settings.get("encoder_device")
    assert {entry["path"] for entry in result["videos"]} == {str(video) for video in videos}
    assert json.loads((tmp_path / "render-manifest.json").read_text()) == result
