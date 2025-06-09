from pathlib import Path
import shutil
from natsort import natsorted
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Animation2DBuilder:
    def __init__(self):
        pass

    def write(self, path, animation_name='animation', mask=None, shape_scale=1,
              fps=12, clim=[0, 1], shape=(100, 100), cmap="coolwarm",
              clear=False, prog_bar=False):
        """
        Write an animation from a folder with snapshots.

        Parameters
        ----------
        path : str or Path
            Path to the folder with snapshots.
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
        clear : bool
            Clear the snapshot folder after writing the animation.
        prog_bar : bool
            Show progress bar.
        """
        path = Path(path)
        path_save = path.parent.joinpath(animation_name).with_suffix(".mp4")

        files = natsorted(path.glob("*.npy"))

        height, width = np.array(shape) * shape_scale
        cmap = plt.get_cmap(cmap)

        with (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                   s=f'{width}x{height}', framerate=fps)
            .output(path_save.as_posix(), pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        ) as process:
            # Write frames to FFmpeg process
            for file in tqdm(files, desc='Building animation',
                             disable=not prog_bar):
                frame = np.load(file.with_suffix(".npy"))
                # Normalize the frame data to the colormap
                mask_ = (frame < clim[0]) | (frame > clim[1])

                if mask is not None:
                    mask_ |= mask

                frame = (frame - clim[0]) / (clim[1] - clim[0])
                frame[mask_] = np.nan

                frame = (cmap(frame, bytes=True)[:, :, :3]).astype("uint8")

                # Upscale the frame if necessary
                if shape_scale > 1:
                    frame = np.repeat(np.repeat(frame, shape_scale, axis=0),
                                      shape_scale, axis=1)

                process.stdin.write(frame.tobytes())

        # Close the FFmpeg process
        process.stdin.close()
        process.wait()

        if clear:
            shutil.rmtree(path)
