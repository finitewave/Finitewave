from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class MultiVariableTracker(Tracker):
    """
    A class to track multiple variables at a specific node(s) of cardiac tissue.

    This tracker monitors user-defined variables at a specified node(s) and
    records their values over time.

    Attributes
    ----------
    node_inds : array-like
        The indices of one or multiple nodes in the tissue where the variables are tracked.
    var_list : list of str
        A list of variable names to be tracked.
    vars_data : dict
        A dictionary where each key is a variable name, and the value is
        an array of its tracked values over time.
    tracking_times : np.ndarray
        An array of time points corresponding to the tracked variable values.
    """
    def __init__(self, node_inds=None, var_list=None, start_time=0, end_time=np.inf, step=1):
        """
        Initializes the MultiVariableTracker with default parameters.

        Parameters
        ----------
        node_inds : array-like
            The indices of the node(s) where the variables are tracked. List of
            lists can be used to track multiple nodes.
        var_list : list of str
            A list of variable names to be tracked. If None, all state variables in the model will be tracked.
        start_time : float, optional
            The time at which tracking will begin. Default is 0.
        end_time : float, optional
            The time at which tracking will end. Default is infinity.
        step : int, optional
            The frequency at which tracking will occur. Default is 1.
        """
        super().__init__(start_time=start_time, end_time=end_time, step=step)
        self.node_inds = node_inds
        self.var_list = var_list
        self.vars_data = {}

    def initialize(self, simulation):
        """
        Initializes the tracker with the simulation model and precomputes
        necessary values for each variable.

        Parameters
        ----------
        simulation : object
            The simulation object containing the cardiac tissue model object
            containing the data to be tracked.
        """
        super().initialize(simulation)
        if self.node_inds is None:
            raise ValueError("Node indices must be provided for tracking.")

        self.vars_data = {}
        self.model = simulation.cardiac_model

        if self.var_list is None:
            self.var_list = self.model.state_vars

        self._node_inds = self._compute_node_inds(simulation.cardiac_tissue, simulation.cardiac_model)
        self._node_inds = self.simulation.backend.wrap_indexes(self._node_inds)

        # Initialize storage for each variable to be tracked
        for var_name in self.var_list:
            if not hasattr(self.model, var_name):
                raise ValueError(f"Variable '{var_name}' not found in model.")
            
            var_val = getattr(self.model, var_name)
            
            if var_val.size < self._node_inds.max() + 1:
                msg = (f"Some node indices are out of bounds for variable " +
                       f"'{var_name}' with size {var_val.size}.")
                raise ValueError(msg)
            
            var_val = np.zeros((len(self.tracking_times), len(self._node_inds)),
                                dtype=np.float64)
            self.vars_data[var_name] = self.simulation.backend.wrap(var_val)

    def _compute_node_inds(self, cardiac_tissue, cardiac_model):
        """
        Computes the cell indices for tracking based on the mesh and memory
        saving settings.

        Parameters
        ----------
        cardiac_tissue : object
            The cardiac tissue object containing the mesh information.
        cardiac_model : object
            The cardiac model object containing the memory saving settings.

        Returns
        -------
        list or list of lists with two indices
            The computed cell indices for tracking.
        """
        mesh = cardiac_tissue.mesh

        if len(mesh.shape) == 1:
            flat_ind = np.atleast_1d(self.node_inds)
            return flat_ind
        
        flat_ind = np.ravel_multi_index(np.atleast_2d(self.node_inds).T, mesh.shape)

        if not cardiac_model.memory_save:
            return flat_ind

        ind = - np.ones(mesh.size, dtype=int)
        ind[cardiac_model.tissue_indexes] = np.arange(cardiac_model.tissue_indexes.size)
        flat_ind = ind[flat_ind]

        if np.any(flat_ind < 0):
            non_tissue_inds = np.array(self.node_inds)[flat_ind < 0]
            raise ValueError(f"Specified nodes {non_tissue_inds} are not part of the tissue.")

        return flat_ind

    def _track(self):
        """
        Tracks and stores the values of each specified variable at each time step.

        This method should be called at each time step of the simulation.
        """
        for var_name in self.var_list:
            var_values = self.model.__dict__[var_name]
            var_val = self.simulation.backend.select_values(var_values, self._node_inds)
            self.vars_data[var_name] = self.simulation.backend.set_values(
                self.vars_data[var_name], self.tracking_counter, var_val
            )
        
    @property
    def output(self):
        """
        Returns the tracked variables data.

        Returns
        -------
        dict
            A dictionary where each key is a variable name, and the value is
            an array of its tracked values over time.
        """
        vars_data = {}
        for var_name in self.var_list:
            vars_data[var_name] = np.squeeze(self.vars_data[var_name])
        return vars_data

    def write(self, path=".", dir_name="tracked_data"):
        """
        Saves the tracked variables to disk as NumPy files.

        Parameters
        ----------
        path : str, optional
            The directory path where the tracked data will be saved.
            Default is the current directory.
        dir_name : str, optional
            The name of the directory where the tracked data will be saved.
            Default is "tracked_data".
        """
        if not Path(path, dir_name).is_dir():
            Path(path, dir_name).mkdir(parents=True)

        vars_data = self.output

        for var_name in self.var_list:
            np.save(Path(path, dir_name, f"{var_name}.npy"), vars_data[var_name])
