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

    def __init__(self, aggregate=False, path=".", dir_name="frames", var_name="u",
                 overwrite=True, output_dtype="float32", keep_shape=False,
                 **kwargs):
        """
        Initializes the FrameGridTracker with default parameters.

        Parameters
        ----------
        aggregate : bool, optional
            Whether to aggregate frames into a single array.
            If False, frames will be saved individually.
        path : str, optional
            Base path for saving frames (default is current directory).
        dir_name : str, optional
            Directory name for saving frames (default is "frames").
        var_name : str, optional
            Name of the target array to capture (default is "u").
        overwrite : bool, optional
            Whether to overwrite existing frames (default is True).
        output_dtype : dtype, optional
            Data type for the saved frames (default is "float32").
            If None, it will be set to the simulation's default floating-point type.
        keep_shape : bool, optional
            Whether to keep the original shape of the tracked variable in the output.
             If False, the output will be flattened array with length equal to
             the number of tissue nodes (default is False).
        **kwargs
            Additional keyword arguments for the base Tracker class.
        """
        super().__init__(**kwargs)
        self.aggregate = aggregate
        self.path = "."
        self.dir_name = dir_name
        self.var_name = var_name
        self.output_dtype = output_dtype
        self.overwrite = overwrite
        self.frames = None
        self.keep_shape = keep_shape

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

        if self.keep_shape:
            output_shape = self.simulation.cardiac_tissue.mesh.shape
        else:
            output_shape = self.simulation.cardiac_model.__dict__[self.var_name].shape

        self.frames = np.zeros((n_frames, *output_shape), dtype=self.output_dtype)

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
        frame_raw = self.simulation.cardiac_model.__dict__[self.var_name]
        frame_raw = np.asarray(frame_raw, dtype=self.output_dtype)

        if self.keep_shape:
            frame = self._reshape_to_mesh(frame_raw)
        else:
            frame = frame_raw

        if self.aggregate:
            self.frames[self.tracking_counter] = frame
            return

        dir_path = Path(self.path, self.dir_name)
        np.save(
            dir_path.joinpath(str(self.tracking_counter)).with_suffix(".npy"),
            frame,
        )

    def _reshape_to_mesh(self, frame):
        """
        Reshapes a flattened frame back to the original mesh shape.

        Parameters
        ----------
        frame : np.ndarray
            The flattened frame to be reshaped.

        Returns
        -------
        np.ndarray
            The reshaped frame with the original mesh shape.
        """
        output_shape = self.simulation.cardiac_tissue.mesh.shape
        tissue_indexes = self.simulation.cardiac_tissue.tissue_indexes
        reshaped_frame = np.full(output_shape, np.nan, dtype=self.output_dtype)
        reshaped_frame.flat[tissue_indexes] = frame
        return reshaped_frame
