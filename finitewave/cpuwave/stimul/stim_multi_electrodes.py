
import numpy as np
from finitewave.core.stimul.stim_area.stim_electrodes import StimElectrodes


class StimMultiElectrodes(StimElectrodes):
    """
    Stimulus sequence for multi-electrode stimulation protocol.

    This class defines a stimulus sequence that applies stimuli to multiple electrodes.
    """
    def __init__(self, coords, size, tree=None):
        """
        Initializes the StimMultiElectrodes instance.

        Parameters
        ----------
        coords : numpy.ndarray
            The coordinates of the electrodes where the stimulus is applied.
        size : float
            The radius around each coordinate to include in the stimulation.
        tree : spatial.KDTree, optional
            A pre-built KDTree for efficient neighbor searching (default is None).
        """
        super().__init__(coords, size)
        self.tree = tree

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
            The indexes of the selected nodes for stimulation.
        """
        if self.tree is None:
            self.tree = spatial.KDTree(myo_coords)
        
        inds = self.tree.query_ball_point(self.coords, self.size, workers=-1)
        inds = np.unique(np.concatenate(inds)).astype(np.int32)
        return myo_indexes[inds]

    