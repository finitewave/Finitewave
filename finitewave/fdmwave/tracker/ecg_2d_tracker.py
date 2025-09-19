from pathlib import Path
import numpy as np
from numba import njit, prange
from scipy.spatial import distance

from finitewave.core.tracker.tracker import Tracker


class ECG2DTracker(Tracker):
    """
    A class to compute and track electrocardiogram (ECG) signals from a 2D
    cardiac tissue model simulation.

    This tracker calculates ECG signals at specified measurement points by
    computing the potential differences across the cardiac tissue mesh and
    considering the inverse square of the distance from each measurement point.

    Attributes
    ----------
    measure_coords : np.ndarray
        An array of points (x, y, z) where ECG signals are measured.
    ecg : list
        The computed ECG signals.
    file_name : str
        The name of the file to save the computed ECG signals.
    u_tr : np.ndarray
        The updated potential values after diffusion.

    """

    def __init__(self, measure_coords=None):
        """
        Initializes the ECG2DTracker with default parameters.

        Parameters
        ----------
        distance_power : int, optional
            The power to which the distance is raised in the calculation of the
            ECG signal. The default is 1.
        """
        super().__init__()
        self.measure_coords = measure_coords
        self.ecg = []
        self.file_name = "ecg.npy"
        self.u_tr = None

    def initialize(self, model):
        """
        Initialize the ECG tracker with the model object.

        Parameters
        ----------
        model : CardiacModel3D
            The model object containing the simulation parameters.
        """
        self.model = model
        self.measure_coords = np.atleast_2d(self.measure_coords)
        self.ecg = []
        self.u_tr = np.zeros_like(model.u)

    def calc_ecg(self):
        """
        Calculate the ECG signal at the measurement points.

        Returns
        -------
        np.ndarray
            The computed ECG signal.
        """
        self.model.diffusion_kernel(self.u_tr,
                                    self.model.u,
                                    self.model.weights,
                                    self.model.cardiac_tissue.myo_indexes)
        ecg = compute_ecg(self.u_tr,
                          self.model.u,
                          self.measure_coords,
                          self.model.dr,
                          self.model.cardiac_tissue.myo_indexes)
        return ecg

    def _track(self):
        """
        Tracks and stores ECG signals at the specified intervals.

        This method should be called at each time step of the simulation.
        """
        ecg = self.calc_ecg()
        self.ecg.append(ecg)

    @property
    def output(self):
        """
        Get the computed ECG signals as a numpy array.

        Returns
        -------
        np.ndarray
            The computed ECG signals.
        """
        return np.array(self.ecg)

    def write(self):
        """
        Save the computed ECG signals to a file.

        The ECG signals are saved as a numpy array in the specified path.
        """
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True)

        np.save(Path(self.path).joinpath(self.file_name).with_suffix('.npy'),
                self.output)

@njit(parallel=True)
def compute_ecg(u_tr, u, coords, dr, indexes):
    """
    Performs isotropic diffusion on a 2D grid.

    Parameters
    ----------
    u_tr : numpy.ndarray
        A 2D array to store the updated potential values after diffusion.
    u : numpy.ndarray
        A 2D array representing the current potential values before diffusion.
    coord : tuple
        The coordinates of the measurement point.
    dr : float
        The spatial resolution of the grid.
    indexes : numpy.ndarray
        A 1D array of indices of the healthy tissue points.
    """
    n_j = u.shape[1]

    n_c = len(coords)
    ecg = np.zeros(n_c)

    for c in range(n_c):
        x, y, z = coords[c]
        ecg_ = 0

        for ind in prange(len(indexes)):
            ii = indexes[ind]
            i = ii // n_j
            j = ii % n_j

            d = (x - i)**2 + (y - j)**2 + (z)**2

            if d > 0:
                ecg_ += (u_tr[i, j] - u[i, j]) / (d * dr)

        ecg[c] = ecg_

    return ecg