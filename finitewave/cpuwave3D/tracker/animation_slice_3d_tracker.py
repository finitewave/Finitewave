from pathlib import Path
import numpy as np

from finitewave.cpuwave2D.tracker.animation_2d_tracker import (
    Animation2DTracker
)
from finitewave.tools.animation_2d_builder import Animation2DBuilder


class AnimationSlice3DTracker(Animation2DTracker):
    """
    A class to track and save 2D frames of a 3D cardiac tissue model simulation
    for animation purposes.

    This tracker periodically saves the state of a specified target array from
    the model to disk as NumPy files, which can later be used to create
    animations.

    Attributes
    ----------
    slice_x : int
        The x-coordinate of the slice to capture.
    slice_y : int
        The y-coordinate of the slice to capture.
    slice_z : int
        The z-coordinate of the slice to capture.
    """

    def __init__(self):
        super().__init__()
        self.slice_x = None
        self.slice_y = None
        self.slice_z = None

    def initialize(self, model):
        self.model = model
        self._frame_counter = 0

        if np.count_nonzero([self.slice_x, self.slice_y, self.slice_z]) != 1:
            message = "Exactly one slice must be specified."
            raise ValueError(message)

        super().initialize(model)

    def _track(self):
        """
        Saves frames based on the specified step interval and target array.

        The frames are saved in the specified directory as NumPy files.
        """
        values = self.model.__dict__[self.variable_name]

        frame = self.select_frame(values)

        np.save(Path(self.path, self.dir_name, str(self._frame_counter)
                     ).with_suffix(".npy"), frame.astype(self.frame_type))

        self._frame_counter += 1

    def select_frame(self, array):
        """
        Selects the slice from the 3D array based on the specified slice
        coordinates.

        Parameters
        ----------
        array : ndarray
            The 3D array to slice.

        Returns
        -------
        ndarray
            The 2D slice of the 3D array.
        """
        if self.slice_x is not None:
            return array[self.slice_x, :, :]

        if self.slice_y is not None:
            return array[:, self.slice_y, :]

        if self.slice_z is not None:
            return array[:, :, self.slice_z]

    def write(self, shape_scale=1, fps=12, cmap="coolwarm", clim=[0, 1],
              clear=False, prog_bar=True):
        """
        Creates an animation from the saved frames using the Animation2DBuilder
        class. Fibrosis and boundaries will be shown in black.

        Parameters
        ----------
        shape_scale : int, optional
            Scale factor for the frame size. The default is 5.
        fps : int, optional
            Frames per second for the animation. The default is 12.
        cmap : str, optional
            Color map for the animation. The default is 'coolwarm'.
        clim : list, optional
            Color limits for the animation. The default is [0, 1].
        clear : bool, optional
            Clear the snapshot folder after creating the animation.
            The default is False.
        prog_bar : bool, optional
            Show a progress bar during the animation creation.
            The default is True.
        """
        animation_builder = Animation2DBuilder()
        path = Path(self.path, self.dir_name)
        mask = self.select_frame(self.model.cardiac_tissue.mesh) != 1

        animation_builder.write(path,
                                animation_name=self.file_name,
                                mask=mask,
                                shape_scale=shape_scale,
                                fps=fps,
                                clim=clim,
                                shape=mask.shape,
                                cmap=cmap,
                                clear=clear,
                                prog_bar=prog_bar)
