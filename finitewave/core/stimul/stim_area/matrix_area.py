import numpy as np


class MatrixArea:
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
        self.matrix = matrix > 0

    def build_stim_indexes(self, simulation):
        """
        Builds the indexes of the cardiac model where the stimulus will be applied.

        cardiac model where the voltage stimulus will be applied.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        """
        if self.matrix.shape != simulation.cardiac_tissue.mesh.shape:
            raise ValueError("The shape of the stimulation matrix does not "
                             "match the shape of the cardiac tissue mesh.")

        tissue_indexes = np.asarray(simulation.cardiac_model.tissue_indexes)
        myo_indexes = np.asarray(simulation.cardiac_model.myo_indexes)

        mesh_mask = self.matrix.flat[tissue_indexes] > 0
        stim_indexes = myo_indexes[mesh_mask[myo_indexes] > 0]
        stim_indexes = simulation.backend.wrap_indexes(stim_indexes)
        return stim_indexes
