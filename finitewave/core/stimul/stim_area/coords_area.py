import numpy as np
from scipy import spatial


class CoordsArea:
    """
    A class that applies a stimulus to specific coordinates in a cardiac model.

    Attributes
    ----------
    coords : numpy.ndarray
        The coordinates of the locations where the stimulus is applied.
    size : float
        The radius around each coordinate to include in the stimulation.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the stimulus is applied.
    """
    def __init__(self, coords, size):
        """
        Initializes the StimElectrodes instance.

        Parameters
        ----------
        coords : numpy.ndarray
            The coordinates of the electrodes where the stimulus is applied.
        size : float
            The radius around each coordinate to include in the stimulation.
        """
        self.coords = coords
        self.size = size

    def build_stim_indexes(self, simulation):
        """
        Initializes the stimulation indexes based on the simulation data.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance containing the cardiac model data.
        """
        myo_indexes = np.asarray(simulation.cardiac_model.myo_indexes)
        tissue_indexes = np.asanyarray(simulation.cardiac_model.tissue_indexes)
        myo_coords = np.asanyarray(simulation.cardiac_tissue.coords[tissue_indexes[myo_indexes]])
        stim_indexes = self.select_nodes(myo_coords, myo_indexes)
        stim_indexes = simulation.backend.wrap_indexes(stim_indexes)
        return stim_indexes

    def select_nodes(self, myo_coords, myo_indexes):
        """
        Selects the nodes in the cardiac model that are within the stimulation
        area.

        Parameters
        ----------
        myo_coords : numpy.ndarray
            The coordinates of the myocardial nodes.
        myo_indexes : numpy.ndarray
            The indexes of the myocardial nodes.

        Returns
        -------
        numpy.ndarray
            The indexes of the nodes that are within the stimulation area.
        """
        tree = spatial.KDTree(myo_coords)
        inds = tree.query_ball_point(self.coords, self.size)
        inds = np.unique(np.concatenate(inds)).astype(np.int32)
        return myo_indexes[inds]
