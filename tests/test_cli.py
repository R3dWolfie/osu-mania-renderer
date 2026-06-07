import subprocess
import sys


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "osu_mania_renderer_v2.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "osu" in result.stdout.lower() or "usage" in result.stdout.lower()
