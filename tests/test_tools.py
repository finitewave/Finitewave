import numpy as np
import pytest
from pathlib import Path
from finitewave.tools.animation_2d_builder import Animation2DBuilder

def test_animation_builder_2d(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        frame = np.random.rand(100, 100)
        np.save(frames_dir / f"frame_{i}.npy", frame)

    builder = Animation2DBuilder()

    try:
        builder.write(
            path=frames_dir,
            animation_name="test_video",
            shape=(100, 100),
            clear=False,
            prog_bar=False
        )
    except Exception as e:
        pytest.fail(f"Write failed with error: {e}")

    output = tmp_path / "test_video.mp4"
    assert output.exists()
