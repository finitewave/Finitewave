from pathlib import Path
import numpy as np
from numba import njit, prange

from finitewave.core.tracker.tracker import Tracker


class ECGGridTracker(Tracker):
    """
    A class to compute and track electrocardiogram (ECG) signals from a 3D
    cardiac tissue model simulation.

    This tracker calculates ECG signals at specified measurement points by
    computing the potential differences across the cardiac tissue mesh and
    considering the inverse of the distance from each measurement point.

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

    def __init__(self, measure_coords=None, distance_power=1):
        super().__init__()
        self.measure_coords = measure_coords
        self.ecg = []
        self.file_name = "ecg.npy"
        self.u_tr = None
        self.distance_power = distance_power

    def initialize(self, simulation):
        """
        Initialize the ECG tracker with the simulation object.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        self.simulation = simulation
        self.measure_coords = np.atleast_2d(self.measure_coords)
        self.ecg = []

        myo_indexes = self.simulation.cardiac_model.myo_indexes
        mesh_indexes = self.simulation.cardiac_model.mesh_indexes

        myo_indexes_on_mesh = mesh_indexes[myo_indexes]

        self.myo_coords = self.unravel_index(
            myo_indexes_on_mesh, self.simulation.cardiac_tissue.mesh.shape)

        self.myo_indexes = np.zeros_like(self.simulation.cardiac_model.u,
                                         dtype=bool)
        self.myo_indexes[myo_indexes] = True

    def calc_ecg(self):
        """
        Calculate the ECG signal at the measurement points.

        Returns
        -------
        np.ndarray
            The computed ECG signal.
        """
        u = self.simulation.solver.u
        u_prev = self.simulation.solver.u_new
        rhs = self.simulation.solver.rhs
        dt = self.simulation.dt
        dr = self.simulation.dr

        # indexes = self.simulation.cardiac_model.myo_indexes
        # i, j, k = self.unravel_index(indexes, u.shape)

        tr_current = (u - u_prev - rhs).flat[self.myo_indexes] / dt
        ecg = calc_ecg(tr_current, self.measure_coords, *self.myo_coords, dr,
                       self.distance_power)
        return ecg

    def unravel_index(self, indexes, shape):
        if len(shape) == 2:
            i, j = np.unravel_index(indexes, shape)
            k = 0.
        elif len(shape) == 3:
            i, j, k = np.unravel_index(indexes, shape)
        else:
            raise ValueError("Unsupported mesh dimension")
        return i, j, k

    def _track(self):
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
        return np.squeeze(self.ecg)

    def write(self):
        """
        Save the computed ECG signals to a file.

        The ECG signals are saved as a numpy array in the specified path.
        """
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True)

        np.save(Path(self.path, self.file_name), self.output)


@njit(parallel=True, fastmath=True)
def calc_ecg(tr_current, coords, i, j, k, dr, distance_power=1):

    n_c = len(coords)
    ecg = np.zeros(n_c, dtype=tr_current.dtype)

    for c in prange(n_c):
        x, y, z = coords[c]
        dx = x - i
        dy = y - j
        dz = z - k
        d = np.sqrt(dx * dx + dy * dy + dz * dz) ** distance_power
        ecg[c] = np.sum(tr_current / (d * dr))

    return ecg
