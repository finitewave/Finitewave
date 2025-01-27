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
        The model instance for which the state will be saved or saved.
    """

    def __init__(self, path="."):
        """
        Initializes the state keeper with the given path.

        Parameters
        ----------
        path : str, optional
            The directory path where the simulation state will be saved.
        """
        self.path = path
        self.passed = False
        self.model = None

    def initialize(self, model):
        """
        Initializes the state keeper with the given model.

        Parameters
        ----------
        model : CardiacModel
            The model instance for which the state will be saved or saved.
        """
        self.model = model
        self.passed = self.path == ""

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

        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True, exist_ok=True)

        for var in self.model.state_vars:
            self._save_variable(Path(self.path).joinpath(var + ".npy"),
                                self.model.__dict__[var])

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
        np.save(var_path, var)
