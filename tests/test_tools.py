import numpy as np
import pytest
from pathlib import Path
from finitewave.tools.animation_2d_builder import Animation2DBuilder
from finitewave.tools.animation_3d_builder import Animation3DBuilder

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

def test_animation_builder_3d(tmp_path):
    builder = Animation3DBuilder()
    
    folder = tmp_path / "slices"
    folder.mkdir()

    shape = (10, 10, 10)
    mask = np.ones(shape, dtype=bool)

    for i in range(3):
        data = np.random.rand(mask[mask > 0].shape[0])
        np.save(folder / f"{i:03d}.npy", data)

    builder.write(
        path=folder,
        mask=mask,
        animation_name="test_3d_animation",
        path_save=tmp_path,
        prog_bar=False,
        scalar_bar=False,
    )

    output = tmp_path / "test_3d_animation.mp4"
    assert output.exists()