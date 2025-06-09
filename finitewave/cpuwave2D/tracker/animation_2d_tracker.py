from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker
from finitewave.tools import Animation2DBuilder


class Animation2DTracker(Tracker):
    """
    A class to track and save frames of a 2D cardiac tissue model simulation
    for animation purposes.

    This tracker periodically saves the state of a specified target array from
    the model to disk as NumPy files, which can later be used to create
    animations.

    Attributes
    ----------
    dir_name : str
        Directory for saving frames.
    variable_name : str
        Name of the target array to capture.
    frame_type : str
        Default frame format settings.
    overwrite : bool
        Overwrite existing frames.
    file_name : str
        Name of the animation file.

    """

    def __init__(self):
        """
        Initializes the Animation2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.dir_name = "animation"   # Directory for saving frames
        self.variable_name = "u"      # Name of the target array to capture
        self.frame_type = "float64"   # Default frame format settings
        self._frame_counter = 0       # Internal frame counter
        self.overwrite = True         # Overwrite existing frames
        self.file_name = "animation"  # Name of the animation file

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and sets up
        directories for saving frames.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        self._frame_counter = 0  # Reset frame counter

        if not Path(self.path, self.dir_name).is_dir():
            Path(self.path, self.dir_name).mkdir(parents=True)

        if self.overwrite:
            for file in Path(self.path, self.dir_name).glob("*.npy"):
                file.unlink()

    def _track(self):
        """
        Saves frames based on the specified step interval and target array.

        The frames are saved in the specified directory as NumPy files.
        """
        frame = self.model.__dict__[self.variable_name]
        dir_path = Path(self.path, self.dir_name)

        np.save(dir_path.joinpath(str(self._frame_counter)
                                  ).with_suffix(".npy"),
                frame.astype(self.frame_type))

        self._frame_counter += 1

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
        mask = self.model.cardiac_tissue.mesh != 1

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
