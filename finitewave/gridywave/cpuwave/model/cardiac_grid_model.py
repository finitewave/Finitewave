import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel


class CardiacGridModel(CardiacModel):
    """
    Base class for cardiac grid models.

    Attributes
    ----------
    state_vars : list
        List of state variables to be saved and restored.
    memory_save : bool
        Whether to save memory by only storing the state variables at the
        tissue indexes (``mesh > 0``).
    """

    def __init__(self, memory_save=False):
        """
        Initializes the CardiacGridModel instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Whether to save memory by only storing the state variables at the
            tissue indexes (``mesh > 0``).
        """
        super().__init__()
        self.memory_save = memory_save
        self.state_vars = []

    def initialize(self, simulation):
        """
        Initializes the model for simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        shape = simulation.cardiac_tissue.mesh.shape

        if self.memory_save:
            shape = (len(simulation.cardiac_tissue.tissue_indexes), )

        self.u = self.init_u * np.ones(shape, dtype=simulation.npfloat)
        self.v = self.init_v * np.ones(shape, dtype=simulation.npfloat)
        self.rhs = np.zeros_like(self.u)
        self.compute_indexes(simulation.cardiac_tissue)

    def init_state_vars(self, shape, npfloat):
        """
        Initializes the state variables with the given initial values.

        Parameters
        ----------
        shape : tuple
            The shape of the state variables.
        npfloat : str
            The data type of the state variables.
        """
        for var in self.state_vars:
            init_val = getattr(self, "init_" + var)
            setattr(self, var, init_val * np.ones(shape, dtype=npfloat))

    def compute_indexes(self, cardiac_tissue):
        """
        Computes the myocyte indexes. If memory saving is enabled, also
        computes the tissue indexes.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The cardiac tissue object.
        """
        if self.memory_save:
            self.myo_indexes = cardiac_tissue.myo_on_tissue_indexes
            self.tissue_indexes = cardiac_tissue.tissue_indexes
            return

        self.myo_indexes = cardiac_tissue.myo_indexes
