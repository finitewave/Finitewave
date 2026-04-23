from pathlib import Path
import numpy as np


class StateSaver:
    """ This class provides functionality to save the state of a
    simulation model, including all relevant variables specified in the model's
    ``state_vars`` attribute.

    Attributes
    ----------
    path : str
        Directory path where the simulation state will be saved.
    passed : bool
        Whether the state has been saved.
    model : CardiacModel
        The model instance for which the state will be saved or loaded.
    time : float
        The time at which to save the state of the simulation.
    node_inds : int
        The index of the node to save.
    """

    def __init__(self, path=".", time=-1, node_inds=None):
        """
        Initializes the state keeper with the given path.

        Parameters
        ----------
        path : str, optional
            The directory path where the simulation state will be saved.
            The default is ".".
        time : float, optional
            The time at which to save the state of the simulation.
            The default is -1 (save at the end of the simulation).
        node_inds : int, optional
            The index of the node to save.
            The default is None (save all nodes).
        """
        self.path = path
        self.passed = False
        self.model = None
        self.time = time
        self.node_inds = node_inds

    def initialize(self, simulation):
        """
        Initializes the state keeper with the given model.

        Parameters
        ----------
        simulation : CardiacSimulation
            The simulation instance for which the state will be saved or loaded.
        """
        self.simulation = simulation
        self.passed = self.path == ""

        if self.node_inds is not None:
            self._node_inds = self._compute_flat_inds()

    def save(self):
        """
        Saves the state of the given model to the specified ``path``
        directory.

        This method creates the necessary directories if they do not exist and
        saves each variable listed in the model's ``state_vars`` attribute as
        a numpy file.
        """
        if self.passed:
            return

        if self.time < 0 and self.simulation.t < self.simulation.t_max:
            return

        if self.time >= 0 and self.simulation.t < self.time:
            return

        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True, exist_ok=True)

        for var in self.simulation.cardiac_model.state_vars:
            var_data = getattr(self.simulation.cardiac_model, var)
            self._save_variable(Path(self.path).joinpath(var + ".npy"), var_data)

        self.passed = True

    def _save_variable(self, var_path, var):
        """
        Saves a variable to a numpy file.

        Parameters
        ----------
        var_path : str
            The file path where the variable will be saved.

        var : numpy.ndarray
            The variable to be saved.
        """
        if self.node_inds is not None:
            var = np.array(var)
            var = var.flat[self._node_inds]

        np.save(var_path, var)

    def _compute_flat_inds(self):
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
        cardiac_model = self.simulation.cardiac_model
        mesh = self.simulation.cardiac_tissue.mesh

        flat_ind = np.ravel_multi_index(np.atleast_2d(self.node_inds).T, mesh.shape)

        ind = - np.ones(mesh.size, dtype=int)
        ind[cardiac_model.tissue_indexes] = np.arange(cardiac_model.tissue_indexes.size)
        flat_ind = ind[flat_ind]

        if np.any(flat_ind < 0):
            non_tissue_inds = np.array(self.node_inds)[flat_ind < 0]
            raise ValueError(f"Specified nodes {non_tissue_inds} are not part of the tissue.")

        return flat_ind


class StateSaverCollection(StateSaver):
    """ This class saves multiple states of a simulation model.

    Attributes
    ----------
    savers : list
        List of StateSaver objects.
    """
    def __init__(self):
        super().__init__()
        self.savers = []

    def initialize(self, simulation):
        """ Initializes the state saver collection with the given simulation.

        Parameters
        ----------
        simulation : CardiacSimulation
            The simulation instance for which the state will be saved or loaded.
        """
        for saver in self.savers:
            saver.initialize(simulation)

    def save(self):
        """ Applies the save method to each StateSaver object in the
        collection.
        """
        for saver in self.savers:
            saver.save()
