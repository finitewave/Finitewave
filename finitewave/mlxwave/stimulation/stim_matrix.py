from finitewave.core.stimulation.stim import Stim


class StimMatrix(Stim):
    """
    A class that applies a stimulus to a cardiac model based on a specified
    matrix.
    """
    def __init__(self, matrix):
        """
        Initializes the StimMatrix instance.

        Parameters
        ----------
        matrix : numpy.ndarray
            An array where the stimulus is applied.
        """
        super().__init__(0.0, 0.0)
        self.matrix = matrix > 0

    def initialize(self, simulation):
        """
        Initializes the stimulation by determining the indexes of the
        cardiac model where the voltage stimulus will be applied.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        """
        if self.matrix.shape != simulation.cardiac_tissue.mesh.shape:
            raise ValueError("The shape of the stimulation matrix does not "
                             "match the shape of the cardiac tissue mesh.")

        self.simulation = simulation
        tissue_indexes = simulation.cardiac_model.tissue_indexes
        mesh_mask = self.matrix.flat[tissue_indexes] > 0
        myo_indexes = simulation.cardiac_model.myo_indexes
        self.stim_indexes = myo_indexes[mesh_mask[myo_indexes] > 0]
