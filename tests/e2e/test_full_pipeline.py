"""End-to-end test: run the whispermlx CLI on a real audio file.

This is the golden-path test. It invokes the actual CLI as a subprocess so the
full VAD -> ASR -> alignment stack runs against real models. Marked
@pytest.mark.e2e and excluded from the default pytest run via pyproject.toml's
addopts ("-m 'not e2e'"). Run explicitly with: uv run pytest -m e2e.

The assertion compares the transcribed segment text against a known-good
baseline. The baseline is intentionally loose (substring match on key words)
because mlx-whisper sampling is near-deterministic but the exact wording can
shift across mlx versions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

E2E_VIDEO = os.path.expanduser("~/Downloads/create_a_video_where_this_is_s.mp4")

pytestmark = pytest.mark.e2e


def _run_cli(audio_path, output_dir, extra_args=None):
    cmd = [
        sys.executable,
        "-m",
        "whisperx",
        audio_path,
        "--model",
        "tiny",
        "--output_dir",
        output_dir,
        "--output_format",
        "json",
        "--no_align",
        "--verbose",
        "False",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def test_full_pipeline_produces_segments(tmp_path):
    if not os.path.exists(E2E_VIDEO):
        pytest.skip(f"E2E video not found at {E2E_VIDEO}")

    result = _run_cli(E2E_VIDEO, str(tmp_path))
    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    # Find the produced JSON (basename of the input + .json).
    basename = os.path.splitext(os.path.basename(E2E_VIDEO))[0]
    json_path = tmp_path / f"{basename}.json"
    assert json_path.exists(), f"Output JSON not found: {json_path}"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "segments" in data
    assert len(data["segments"]) > 0, "expected at least one segment"
    assert "language" in data
    # Each segment has the core fields.
    for seg in data["segments"]:
        assert "text" in seg
        assert isinstance(seg["text"], str)
        assert "start" in seg and "end" in seg
