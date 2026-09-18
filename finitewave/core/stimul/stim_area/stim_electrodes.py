import numpy as np
from scipy import spatial


class StimElectrodes:
    """
    A class that applies a stimulus to specific electrodes in a cardiac model.

    Attributes
    ----------
    coords : numpy.ndarray
        The coordinates of the electrodes where the stimulus is applied.
    size : float
        The radius around each coordinate to include in the stimulation.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the stimulus is applied.
    tree : scipy.spatial.KDTree
        A KDTree for efficient spatial queries of the myocardial tissue.
    mask : numpy.ndarray
        The mask to apply to the myocardial tissue.
    """
    def __init__(self, coords, size, tree=None, mask=None):
        """
        Initializes the StimElectrodes instance.

        Parameters
        ----------
        coords : numpy.ndarray
            The coordinates of the electrodes where the stimulus is applied.
        size : float
            The radius around each coordinate to include in the stimulation.
        tree : scipy.spatial.KDTree, optional
            A KDTree for efficient spatial queries of the myocardial tissue.
            If None, a new KDTree is created.
        mask : numpy.ndarray, optional
            The mask to apply to the myocardial tissue. If None, no mask is applied.
        """
        self.coords = coords
        self.size = size
        self.tree = tree
        self.mask = mask

    def build_stim_indexes(self, simulation):
        """
        Initializes the stimulation indexes based on the simulation data.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance containing the cardiac model data.
        """
        myo_indexes = np.asarray(simulation.cardiac_tissue.myo_indexes)
        myo_coords = np.asarray(simulation.cardiac_tissue.myo_coords)
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
        myo_coords = myo_coords[self.mask] if self.mask is not None else myo_coords
        myo_indexes = myo_indexes[self.mask] if self.mask is not None else myo_indexes

        if self.tree is None:
            self.tree = spatial.KDTree(myo_coords)

        coords = np.atleast_2d(self.coords)
        inds = self.tree.query_ball_point(coords, self.size)
        inds = np.unique(np.concatenate(inds)).astype(np.int32)
        return myo_indexes[inds]
