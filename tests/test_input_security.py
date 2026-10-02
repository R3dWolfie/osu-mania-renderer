import lzma
import tempfile
from pathlib import Path

from osu_mania_renderer_v2.security import (
    bounded_lzma_decompress,
    ffmpeg_file_input_args,
    safe_related_file,
    safe_skin_file,
)


def test_bounded_lzma_decompress_rejects_bomb():
    compressed = lzma.compress(b"x" * (2 * 1024 * 1024))
    try:
        bounded_lzma_decompress(compressed, max_output=1024 * 1024)
    except ValueError as exc:
        assert "output limit" in str(exc)
    else:
        raise AssertionError("oversized LZMA stream was accepted")


def test_related_and_skin_lookups_are_confined():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        media = tmp_path / "Media"
        media.mkdir()
        song = media / "Song.MP3"
        song.write_bytes(b"x")
        assert safe_related_file(tmp_path, "media/song.mp3") == song
        assert safe_related_file(tmp_path, "../Song.MP3") is None
        assert safe_related_file(tmp_path, "/etc/passwd") is None
        assert safe_skin_file(tmp_path, "media/song.mp3") == song
        assert safe_skin_file(tmp_path, "..\\Song.MP3") is None


def test_ffmpeg_file_input_args_forces_file_protocol():
    with tempfile.TemporaryDirectory() as directory:
        args = ffmpeg_file_input_args(Path(directory) / "background.webm")
        assert args[:4] == ["-protocol_whitelist", "file", "-f", "matroska"]
