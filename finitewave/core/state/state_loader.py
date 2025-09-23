from pathlib import Path
import numpy as np


class StateLoader:
    """ This class provides functionality to load the state of a simulation
    model, including all relevant variables specified in the model's
    ``state_vars`` attribute.
    Attributes
    ----------
    path : str
        Directory path from where the simulation state will be loaded.
    passed : bool
        Whether the state has been loaded.
    model : CardiacModel
        The model instance for which the state will be saved or loaded.
    single_value : bool
        Whether the state variables are single values.
    """

    def __init__(self, path=""):
        """
        Initializes the state keeper with the given path.

        Parameters
        ----------
        path : str, optional
            The directory path from where the simulation state will be loaded.
        """
        self.path = path
        self.passed = True
        self.model = None
        self.single_value = False

    def initialize(self, model):
        """
        Initializes the state keeper with the given model.

        Parameters
        ----------
        model : CardiacModel
            The model instance for which the state will be saved or loaded.
        """
        self.model = model
        self.passed = self.path == ""

        if not Path(self.path).exists():
            message = (f"Unable to load state from {self.path}. " +
                       "Directory does not exist.")
            raise FileNotFoundError(message)

    def load(self):
        """
        Loads the state from the specified ``path`` directory and sets
        it in the given model.

        This method loads each variable listed in the model's ``state_vars``
        attribute from numpy files and sets these variables in the model.
        """
        if self.passed:
            return

        for var in self.model.state_vars:
            val = self._load_variable(Path(self.path).joinpath(var + ".npy"))
            setattr(self.model, var, val)

        self.model.update_state()
        self.passed = True

    def _load_variable(self, var_path):
        """
        Loads a state variable from a numpy file.

        Parameters
        ----------
        var_path : str
            The file path from which the variable will be loaded.

        Returns
        -------
        numpy.ndarray
            The variable loaded from the file.
        """
        if self.single_value:
            val = np.load(var_path)
            return (np.ones_like(self.model.u) *
                    val).astype(self.model.npfloat)
        return np.load(var_path)
