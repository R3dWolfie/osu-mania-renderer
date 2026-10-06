"""Runtime selection and pipe failures without requiring ffmpeg, CUDA, or EGL."""
import asyncio
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from osu_mania_renderer_v2.errors import EncoderError
from osu_mania_renderer_v2.render import encode


@pytest.fixture
def encoder_probes(monkeypatch):
    calls = []
    results = {}
    listed = ["h264_nvenc", "h264_vaapi", "libx264", "libopenh264"]

    async def spawn(*command, **options):
        calls.append((command, options))
        if "-encoders" in command:
            output = "\n".join(f" V..... {name} encoder" for name in listed).encode()
            return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(output, b"")))
        name = command[command.index("-c:v") + 1]
        code, stderr = results.get(name, (0, b""))
        return SimpleNamespace(returncode=code, communicate=AsyncMock(return_value=(b"", stderr)))

    monkeypatch.setattr(encode, "_ffmpeg_prefix", lambda: ["host-ffmpeg"])
    monkeypatch.setattr(encode.asyncio, "create_subprocess_exec", spawn)
    return calls, results, listed


def attempted(calls):
    return [cmd[cmd.index("-c:v") + 1] for cmd, _ in calls if "-c:v" in cmd]


async def test_auto_rejects_compiled_nvenc_without_cuda_and_uses_working_vaapi(
        encoder_probes, caplog):
    calls, results, _ = encoder_probes
    results["h264_nvenc"] = (1, b"Cannot load libcuda.so.1\n")
    assert await encode.probe_encoder("auto", "/dev/dri/renderD128") == "h264_vaapi"
    assert attempted(calls) == ["h264_nvenc", "h264_vaapi"]
    vaapi, options = calls[-1]
    assert vaapi[0] == "host-ffmpeg"
    assert vaapi[vaapi.index("-vaapi_device") + 1] == "/dev/dri/renderD128"
    assert "format=nv12,hwupload" in vaapi
    assert options["stdin"] == asyncio.subprocess.PIPE
    assert vaapi[-3:] == ("-f", "null", "-")  # No test video or output file.
    assert "Cannot load libcuda.so.1" in caplog.text


async def test_auto_uses_software_when_both_hardware_encoders_fail(encoder_probes):
    calls, results, _ = encoder_probes
    results.update(h264_nvenc=(1, b"No CUDA"), h264_vaapi=(1, b"No device"))
    assert await encode.probe_encoder("auto", "/dev/dri/renderD128") == "libx264"
    assert attempted(calls) == ["h264_nvenc", "h264_vaapi", "libx264"]


async def test_working_nvenc_retains_first_preference(encoder_probes):
    calls, _, _ = encoder_probes
    assert await encode.probe_encoder("auto", "/dev/dri/renderD128") == "h264_nvenc"
    assert attempted(calls) == ["h264_nvenc"]


async def test_auto_tries_openh264_when_x264_cannot_start(encoder_probes):
    calls, results, _ = encoder_probes
    results.update(h264_nvenc=(1, b"No CUDA"), h264_vaapi=(1, b"No device"),
                   libx264=(1, b"Cannot start x264"))
    assert await encode.probe_encoder("auto", "/dev/dri/renderD128") == "libopenh264"
    assert attempted(calls)[-2:] == ["libx264", "libopenh264"]


async def test_auto_does_not_probe_encoders_absent_from_ffmpeg(encoder_probes):
    calls, _, listed = encoder_probes
    listed[:] = ["libx264"]
    assert await encode.probe_encoder("auto", "/dev/dri/renderD128") == "libx264"
    assert attempted(calls) == ["libx264"]


async def test_auto_without_vaapi_device_falls_back_to_software(encoder_probes, monkeypatch):
    calls, results, _ = encoder_probes
    results["h264_nvenc"] = (1, b"No CUDA")
    monkeypatch.setattr(encode.Path, "exists", lambda _: False)
    assert await encode.probe_encoder("auto", None) == "libx264"
    assert attempted(calls) == ["h264_nvenc", "libx264"]


@pytest.mark.parametrize("amf_works", [True, False])
async def test_windows_hardware_preference_is_retained(encoder_probes, monkeypatch, amf_works):
    calls, results, listed = encoder_probes
    listed.extend(("h264_amf", "h264_qsv"))
    results["h264_nvenc"] = (1, b"No CUDA")
    if not amf_works:
        results["h264_amf"] = (1, b"No AMD device")
    monkeypatch.setattr(encode.sys, "platform", "win32")
    assert await encode.probe_encoder("auto", None) == ("h264_amf" if amf_works else "h264_qsv")
    assert attempted(calls) == (["h264_nvenc", "h264_amf"] if amf_works else
                                ["h264_nvenc", "h264_amf", "h264_qsv"])


async def test_no_working_encoder_reports_actual_failures(encoder_probes):
    _, results, listed = encoder_probes
    results.update({name: (1, f"Cannot initialise {name}".encode()) for name in listed})
    with pytest.raises(EncoderError, match="No usable H.264 encoder") as error:
        await encode.probe_encoder("auto", "/dev/dri/renderD128")
    assert "Cannot initialise h264_nvenc" in str(error.value)


@pytest.mark.parametrize("encoder", ["libx264", "h264_nvenc", "h264_vaapi"])
async def test_explicit_encoder_remains_an_explicit_override(encoder_probes, encoder):
    calls, _, _ = encoder_probes
    assert await encode.probe_encoder(encoder, None) == encoder
    assert not calls


async def test_runtime_probe_timeout_kills_and_reaps_child(monkeypatch):
    process = SimpleNamespace(returncode=None, killed=False)

    async def communicate(_=None):
        if not process.killed:
            raise TimeoutError
        process.returncode = -9
        return b"", b"driver stalled"

    def kill():
        process.killed = True

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill = kill
    monkeypatch.setattr(encode, "_ffmpeg_prefix", lambda: ["host-ffmpeg"])
    monkeypatch.setattr(encode.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    usable, reason = await encode._encoder_usable("h264_nvenc", None)
    assert not usable and "timed out" in reason
    assert process.killed and process.returncode == -9
    assert process.communicate.await_count == 2


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_encoder_query_timeout_or_cancellation_reaps_process(monkeypatch, failure):
    process = SimpleNamespace(returncode=None)
    process.communicate = AsyncMock(side_effect=[failure(), (b"", b"")])
    process.kill = lambda: setattr(process, "returncode", -9)
    monkeypatch.setattr(encode.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    if failure is TimeoutError:
        with pytest.raises(EncoderError, match="encoder query timed out"):
            await encode.probe_encoder("auto", None)
    else:
        with pytest.raises(asyncio.CancelledError):
            await encode.probe_encoder("auto", None)
    assert process.returncode == -9 and process.communicate.await_count == 2


async def failed_pipe():
    # A tiny Python child substitutes for ffmpeg: it prints the real failure
    # shape and exits without reading stdin. This exercises actual pipe cleanup.
    pipe = encode.FfmpegPipe([sys.executable, "-c",
        "import sys; sys.stderr.write('Cannot load libcuda.so.1\\n'); sys.exit(7)"])
    await pipe.start()
    await asyncio.wait_for(pipe.proc.wait(), 3)
    fd = pipe._stdin_fd
    lease = SimpleNamespace(mv=b"frame", done=threading.Event())
    await pipe.write_frame(lease)
    assert await asyncio.to_thread(lease.done.wait, 2)
    assert isinstance(pipe._werr, BrokenPipeError)
    return pipe, fd


@pytest.mark.parametrize("on_close", [False, True])
async def test_broken_pipe_reports_ffmpeg_stderr_and_cleans_writer(tmp_path, on_close):
    pipe, fd = await failed_pipe()
    try:
        with pytest.raises(EncoderError, match="Cannot load libcuda.so.1") as error:
            if on_close:
                await pipe.close(tmp_path / "failed.mp4")
            else:
                await pipe.write_frame(b"another frame")
        assert "7" in str(error.value)
        assert pipe._stdin_fd is None and pipe._writer is None
        assert pipe.proc.returncode == 7
        with pytest.raises(OSError):
            os.fstat(fd)
        with pytest.raises(EncoderError, match="Cannot load libcuda.so.1"):
            await pipe.close(tmp_path / "failed.mp4")  # Repeated cleanup retains the diagnostic.
    finally:
        # Keep even the initial failing regression free of leaked descriptors.
        if pipe._writer is not None:
            pipe._join_writer()
        if pipe._stdin_fd is not None:
            os.close(pipe._stdin_fd)
        await pipe.proc.communicate()


async def test_successful_pipe_close_preserves_frame_bytes_and_captures_stderr(tmp_path):
    output = tmp_path / "output.mp4"
    pipe = encode.FfmpegPipe([sys.executable, "-c",
        "import sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read()); "
        "sys.stderr.write('test diagnostic\\n')", str(output)])
    await pipe.start()
    await pipe.write_frame(b"first frame")
    await pipe.write_frame(b"second frame")
    await pipe.close(output)
    assert output.read_bytes() == b"first framesecond frame"
    assert pipe._stdin_fd is None and pipe._writer is None and pipe.proc.returncode == 0
    assert b"test diagnostic" in pipe._stderr_log


async def test_close_reports_nonzero_exit_even_when_frames_were_consumed(tmp_path):
    pipe = encode.FfmpegPipe([sys.executable, "-c",
        "import sys; sys.stdin.buffer.read(); "
        "sys.stderr.write('Output permission denied\\n'); sys.exit(3)"])
    await pipe.start()
    await pipe.write_frame(b"frame")
    with pytest.raises(EncoderError, match="ffmpeg exit code 3: Output permission denied"):
        await pipe.close(tmp_path / "failed.mp4")
    assert pipe._stdin_fd is None and pipe._writer is None
