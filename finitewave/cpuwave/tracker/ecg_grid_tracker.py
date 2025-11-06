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
        self.measure_coords = self.build_measure_coords(self.measure_coords)
        self.ecg = []

        mesh_shape = self.simulation.cardiac_tissue.mesh.shape
        myo_indexes = self.simulation.cardiac_model.myo_indexes
        mesh_indexes = self.simulation.cardiac_model.mesh_indexes

        self.myo_coords = self.build_myo_coords(myo_indexes, mesh_indexes,
                                                mesh_shape)
        self.myo_mask = self.build_myo_mask(myo_indexes, mesh_indexes)

    def build_myo_coords(self, myo_indexes, mesh_indexes, mesh_shape):
        global_myo_indexes = mesh_indexes[myo_indexes]
        myo_coords = np.unravel_index(global_myo_indexes, mesh_shape)

        if len(myo_coords) == 2:
            myo_coords = (myo_coords[0], myo_coords[1], 0)
        return myo_coords

    def build_measure_coords(self, coords):
        coords = np.atleast_2d(coords)
        coords = np.hstack((coords, np.zeros((coords.shape[0],
                                              3 - coords.shape[1]))))
        return coords.astype(self.simulation.npfloat)

    def build_myo_mask(self, myo_indexes, mesh_indexes):
        myo_mask = np.zeros_like(mesh_indexes, dtype=bool)
        myo_mask[myo_indexes] = True
        return myo_mask

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

        tr_current = (u - u_prev - dt * rhs).flat[self.myo_mask] / dt
        ecg = calc_ecg(tr_current, self.measure_coords, *self.myo_coords, dr,
                       self.distance_power)
        return ecg

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

    n = coords.shape[0]
    ecg = np.zeros(n, dtype=tr_current.dtype)

    for c in prange(n):
        x, y, z = coords[c]
        ds = (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2
        d = np.sqrt(ds) ** distance_power
        ecg[c] = np.sum(tr_current / (d * dr))

    return ecg
