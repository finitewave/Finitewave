from pathlib import Path
import shutil
from natsort import natsorted
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Animation2DBuilder:
    def __init__(self):
        self.path = None
        self.path_save = None
        self.scalar_mask = None
        self.prog_bar = False
        self.animation_name = 'animation'

    def load_scalar(self, path, scalar_mask=None, nan_mask=None):
        """
        Load a scalar field value from path and apply mask if provided.

        Parameters
        ----------
        path : str
            Path to the NumPy file containing the scalar data.
        scalar_mask : ndarray, optional
            Mask indicating where scalar values should be placed
            in the output array. If given, the output will be reshaped
            according to the mask.
        nan_mask : ndarray, optional
            Mask indicating where to set NaN values in the output array.

        Returns
        -------
        ndarray
            The loaded scalar data, possibly reshaped according to the mask.
        """

        scalar = np.load(path).astype(np.float32)

        if scalar_mask is None:
            if nan_mask is not None:
                scalar[nan_mask] = np.nan
            return scalar

        if scalar_mask.shape == scalar.shape:
            if nan_mask is not None:
                scalar[nan_mask] = np.nan
            return scalar

        if scalar_mask[scalar_mask > 0].shape == scalar.shape:
            scalar_mesh = np.zeros_like(scalar_mask, dtype=float)
            scalar_mesh[scalar_mask > 0] = scalar

            if nan_mask is not None:
                scalar_mesh[nan_mask] = np.nan
            return scalar_mesh

        raise ValueError("Mask and scalar must have the same shape, or scalar"
                         + " must have the same shape as mask[mask > 0]")

    def collect_frames(self, path):
        """
        Collect and sort frame files from the specified directory.

        Parameters
        ----------
        path : str
            Directory path containing the frame files.

        Returns
        -------
        list of Path
            Sorted list of frame file paths.
        """
        files = natsorted(Path(path).glob("*.npy"))

        if len(files) == 0:
            raise ValueError("No files found")

        return files

    def write(self, mask=None, shape_scale=1, fps=12, clim=[0, 1],
              cmap="RdBu_r", **kwargs):
        """
        Write an animation from a folder with snapshots.

        Parameters
        ----------
        mask : ndarray
            Mask to apply to the frames.
        shape_scale : int
            Scale factor for the frames.
        fps : int
            Frames per second.
        clim : list
            Color limits for the colormap.
        cmap : str
            Matplotlib colormap to use.
        """
        path = Path(self.path)
        files = self.collect_frames(path)

        path_save = self.path_save

        if path_save is None:
            path_save = path.parent

        path_save = Path(path_save).joinpath(f"{self.animation_name}.mp4")

        image = self.load_scalar(files[0], self.scalar_mask)
        height, width = np.array(image.shape) * shape_scale
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
                             disable=not self.prog_bar):
                frame = self.load_scalar(file, self.scalar_mask)
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
