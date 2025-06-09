from pathlib import Path
import numpy as np

from finitewave.tools.animation_2d_builder import Animation2DBuilder
from .local_activation_time_2d_tracker import LocalActivationTime2DTracker


class PeriodAnimation2DTracker(LocalActivationTime2DTracker):
    """
    A class to track the periods of activation for each cell in a 2D cardiac
    tissue model.

    This class extends Animation2DTracker to create and save a period map that
    shows the time interval between successive activations of each cell that
    crosses a given threshold. The period map is saved at each time step.

    Attributes
    ----------
    dir_name : str
        The directory name to save the period maps.
    file_name : str
        The file name for saving the period maps.
    overwrite : bool
        Whether to overwrite existing period maps.
    period_map : np.ndarray
        The array to store activation periods.
    """

    def __init__(self):
        """
        Initializes the PeriodMap2DTracker with default parameters.
        """
        super().__init__()

        self.dir_name = "period"   # Directory to save the period maps
        self.file_name = "period"  # File name for saving the period maps
        self.overwrite = False     # Overwrite existing period maps
        self._frame_counter = 0    # Counter to track the current frame number

        self.period_map = np.ndarray  # Array to store activation periods

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and preallocates
        memory for tracking.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        # Initialize the period map and state arrays
        self.model = model
        self._frame_counter = 0
        self.period_map = - np.ones_like(self.model.u)
        self._activated = np.full(self.model.u.shape, 0, dtype=bool)

        if not Path(self.path, self.dir_name).is_dir():
            Path(self.path, self.dir_name).mkdir(parents=True)

        if self.overwrite:
            for file in Path(self.path, self.dir_name).glob("*.npy"):
                file.unlink()

    def _track(self):
        """
        Tracks the activation periods at each time step of the simulation.

        This method calculates the time interval between successive activations
        for each cell, updates the period map, and saves it to a file.
        """
        cross_mask = self.cross_threshold()
        self.period_map = np.where(cross_mask, self.model.t, self.period_map)

        np.save(Path(self.path, self.dir_name, str(self._frame_counter)
                     ).with_suffix(".npy"), self.period_map)
        self._frame_counter += 1

    def write(self, shape_scale=3, fps=10, clim=None, cmap="viridis",
              clear=True, prog_bar=True):
        """
        Creates an animation from the saved period maps.

        Parameters
        ----------
        shape_scale : int, optional
            The scaling factor for the shape of the period map.
        fps : int, optional
            The frames per second for the animation.
        clim : list, optional
            The color limits for the animation.
        cmap : str, optional
            The color map for the animation.
        clear : bool, optional
            Whether to clear the directory before saving the animation.
        prog_bar : bool, optional
            Whether to show a progress bar during the animation creation.
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
