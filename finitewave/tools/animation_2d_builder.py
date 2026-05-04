from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _require_tools():
    # natsort
    try:
        from natsort import natsorted
    except ImportError as e:
        raise ImportError(
            "Natural frame sorting requires optional dependencies.\n"
            "Install with:\n"
            "  pip install 'finitewave[tools]'"
        ) from e

    # ffmpeg-python
    try:
        import ffmpeg
    except ImportError as e:
        raise ImportError(
            "Animation building requires ffmpeg-python.\n"
            "Install with:\n"
            "  pip install 'finitewave[tools]'"
        ) from e

    # system ffmpeg
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg binary not found (missing `ffmpeg` in PATH).\n\n"
            "Install FFmpeg:\n"
            "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Conda: conda install -c conda-forge ffmpeg\n"
            "  - Windows: winget install Gyan.FFmpeg\n"
        )

    return natsorted, ffmpeg


class Animation2DBuilder:

    def write(
        self,
        path,
        path_save=None,
        animation_name='animation',
        mask=None,
        shape_scale=1,
        fps=12,
        clim=(0, 1),
        shape=(100, 100),
        cmap="coolwarm",
        prog_bar=False
    ):
        """
        Write an animation from a folder with snapshots.

        Parameters
        ----------
        path : str or Path
            Path to the folder with snapshots.
        path_save : str or Path, optional
            Path to save the animation file. If None, it will be saved in the
            parent directory of `path`.
        animation_name : str
            Name of the animation file.
        mask : ndarray
            Mask to apply to the frames.
        shape_scale : int
            Scale factor for the frames.
        fps : int
            Frames per second.
        clim : list
            Color limits for the colormap.
        shape : tuple
            Shape of the frames.
        cmap : str
            Matplotlib colormap to use.
        prog_bar : bool
            Show progress bar.
        """
        natsorted, ffmpeg = _require_tools()

        path = Path(path)

        if path_save is None:
            path_save = path.parent

        path_save = Path(path_save).joinpath(f"{animation_name}.mp4")

        files = list(natsorted(path.glob("*.npy")))
        if not files:
            raise FileNotFoundError(f"No .npy frames found in {path}")

        height, width = (np.array(shape) * shape_scale).astype(int)
        cmap_obj = plt.get_cmap(cmap)

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                   s=f'{width}x{height}', framerate=fps)
            .output(path_save.as_posix(), pix_fmt='yuv420p')
            .global_args("-loglevel", "error") # to print only errors from ffmpeg (as we put quiet=False)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=False, pipe_stderr=False, quiet=False) # there was a deadlock problem when using quiet=True
        )

        try:
            for file in tqdm(files, desc='Building animation', disable=not prog_bar):
                frame = np.load(file)

                mask_ = (frame < clim[0]) | (frame > clim[1])
                if mask is not None:
                    mask_ |= mask

                frame = (frame - clim[0]) / (clim[1] - clim[0])
                frame[mask_] = np.nan

                rgb = (cmap_obj(frame, bytes=True)[:, :, :3]).astype("uint8")

                if shape_scale > 1:
                    rgb = np.repeat(np.repeat(rgb, shape_scale, axis=0),
                                    shape_scale, axis=1)

                process.stdin.write(rgb.tobytes())

        finally:
            if process.stdin:
                process.stdin.close()
            process.wait()