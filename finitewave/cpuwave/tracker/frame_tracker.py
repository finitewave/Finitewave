from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class FrameTracker(Tracker):
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
    output_dtype : str
        Default frame format settings.
    overwrite : bool
        Overwrite existing frames.
    """

    def __init__(self, aggregate=False, dir_name="snapshots", var_name="u",
                 overwrite=True, output_dtype="float32", **kwargs):
        """
        Initializes the FrameGridTracker with default parameters.

        Parameters
        ----------
        aggregate : bool, optional
            Whether to aggregate frames into a single array.
            If False, frames will be saved individually.
        dir_name : str, optional
            Directory name for saving frames (default is "snapshots").
        var_name : str, optional
            Name of the target array to capture (default is "u").
        overwrite : bool, optional
            Whether to overwrite existing frames (default is True).
        output_dtype : dtype, optional
            Data type for the saved frames (default is "float32").
            If None, it will be set to the simulation's default floating-point type.
        **kwargs
            Additional keyword arguments for the base Tracker class.
        """
        super().__init__(**kwargs)
        self.aggregate = aggregate
        self.dir_name = dir_name
        self.var_name = var_name
        self.output_dtype = output_dtype
        self.overwrite = overwrite
        self.frames = None

    def initialize(self, simulation):
        """
        Initializes the tracker with the simulation model and sets up
        directories for saving frames.

        Parameters
        ----------
        simulation : object
            The cardiac tissue model object containing the data to be tracked.
        """
        super().initialize(simulation)

        if self.aggregate:
            self._make_array()
        else:
            self._make_dir()

    @property
    def output(self):
        """
        Returns the tracked frames.

        Returns
        -------
        np.ndarray
            The array containing the tracked frames if aggregation is enabled.
            Otherwise, returns None since frames are saved individually.
        """
        if self.aggregate:
            return self.frames
        return None
    
    def _make_array(self):
        t_max = min(self.simulation.t_max, self.end_time)
        t_min = self.start_time
        dt = self.simulation.dt
        n_frames = int((t_max - t_min) / (self.step * dt)) + 1
        var_data = self.simulation.cardiac_model.__dict__[self.var_name]

        self.frames = np.zeros((n_frames, *var_data.shape))

    def _make_dir(self):
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
        frame = self.simulation.cardiac_model.__dict__[self.var_name]
        

        if self.aggregate:
            self.frames[self.tracking_counter] = frame.astype(self.output_dtype)
            return

        dir_path = Path(self.path, self.dir_name)
        np.save(
            dir_path.joinpath(str(self.tracking_counter)).with_suffix(".npy"),
            frame.astype(self.output_dtype),
        )
