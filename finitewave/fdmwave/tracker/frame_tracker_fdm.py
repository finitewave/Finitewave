from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker
from finitewave.tools import Animation2DBuilder


class FrameTrackerFDM(Tracker):
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
    var_name : str
        Name of the target array to capture.
    frame_type : str
        Default frame format settings.
    overwrite : bool
        Overwrite existing frames.
    mask_output : bool
        Mask output to save only myocardium (mesh == 1).
    """

    def __init__(self):
        """
        Initializes the FrameTrackerFDM with default parameters.
        """
        Tracker.__init__(self)
        self.dir_name = "snapshots"   # Directory for saving frames
        self.var_name = "u"           # Name of the target array to capture
        self.frame_type = "float64"   # Default frame format settings
        self._frame_counter = 0       # Internal frame counter
        self.overwrite = True         # Overwrite existing frames
        self.mask_output = False      # Mask output to save only myocardium (mesh == 1)

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

        if self.mask_output:
            self.mask = self.model.cardiac_tissue.mesh > 0

    def _track(self):
        """
        Saves frames based on the specified step interval and target array.

        The frames are saved in the specified directory as NumPy files.
        """
        frame = self.model.__dict__[self.variable_name].copy()
        if self.mask_output:
            frame = frame[self.mask]
        else:
            frame[self.model.cardiac_tissue.mesh != 1] = np.nan

        dir_path = Path(self.path, self.dir_name)

        np.save(
            dir_path.joinpath(str(self._frame_counter)).with_suffix(".npy"),
            frame.astype(self.frame_type),
        )

        self._frame_counter += 1
