import numpy as np
from warnings import warn

from finitewave.core.model.cardiac_model_base import CardiacModelBase


class CardiacModel(CardiacModelBase):
    """
    Base class for cardiac grid models.

    Attributes
    ----------
    state_vars : list
        List of state variables to be saved and restored.
    memory_save : bool
        Whether to save memory by only storing the state variables at the
        tissue indexes (``mesh > 0``).
    D_model : float
        Model-specific diffusion coefficient.
    u : np.ndarray
        Array representing the action potential (mV) across the tissue.
    rhs : np.ndarray
        Array representing the sum of the ionic currents multiplied by dt.
    myo_indexes : np.ndarray
        Array of myocyte indices corresponding to cardiac model arrays.
        If memory saving is enabled, the indexes correspond to
        ``mesh.flat[tissue_indexes] == 1``.
    tissue_indexes : np.ndarray
        Array of indices corresponding to the full tissue mesh.
        State variables and rhs correspond to mesh.flat[tissue_indexes]
    """

    def __init__(self, memory_save=False):
        """
        Initializes the CardiacModel instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Whether to save memory by only storing the state variables at the
            tissue indexes (``mesh > 0``).
        """
        super().__init__()
        self.memory_save = memory_save
        self.state_vars = []
        self.step = 1
        self.counter = 0

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
        
        if not self.memory_save:
            tissue_fraction = (len(simulation.cardiac_tissue.tissue_indexes) /
                               simulation.cardiac_tissue.mesh.size)
            if tissue_fraction < 0.5:
                warn(f"Tissue fraction is only {tissue_fraction:.2f}. " +
                     "Consider enabling memory saving for better performance.")

        self.init_state_vars(shape, simulation.npfloat)
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
        self.tissue_indexes = np.arange(cardiac_tissue.mesh.size)

    def build_prepacing(self, dt, n_beats, bcl, stim_duration, stim_amplitude):
        t_max = n_beats * bcl

        stim_values = np.zeros(int(t_max / dt), dtype=np.float64)

        for s in np.arange(n_beats):
            stim_start = s * bcl
            stim_end = stim_start + stim_duration
            
            start_idx = int(stim_start / dt)
            end_idx = int(stim_end / dt)
            stim_values[start_idx: end_idx] = dt * stim_amplitude

        return stim_values
