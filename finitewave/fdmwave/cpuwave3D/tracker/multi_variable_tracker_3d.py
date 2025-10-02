import numpy as np
from finitewave.fdmwave.cpuwave2D.tracker.multi_variable_tracker_2d import (
    MultiVariableTracker2D
)


class MultiVariableTracker3D(MultiVariableTracker2D):
    def __init__(self):
        super().__init__()

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
        self.vars = {}
        self.simulation = simulation
        self.model = simulation.cardiac_model
        # Initialize storage for each variable to be tracked
        for var_ in self.var_list:
            if var_ not in self.model.__dict__:
                raise ValueError(f"Variable '{var_}' not found in model.")
            self.vars[var_] = []

        if self.cell_ind is None:
            self._meausure_mask = simulation.cardiac_tissue.mesh == 1
        else:
            self._meausure_mask = tuple(np.atleast_2d(self.cell_ind).T)
