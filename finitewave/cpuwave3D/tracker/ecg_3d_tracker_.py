from pathlib import Path
import numpy as np
from scipy import spatial

from finitewave.core.tracker.tracker import Tracker


class ECG3DTracker(Tracker):
    """
    A class to compute and track electrocardiogram (ECG) signals from a 3D
    cardiac tissue model simulation.

    This tracker calculates ECG signals at specified measurement points by
    computing the potential differences across the cardiac tissue mesh and
    considering the inverse of the distance from each measurement point.

    Parameters
    ----------
    memory_save (bool): A flag to enable memory saving mode. If True, the
        tracker will compute distances on the fly instead of precomputing them.
    batch_size (int): The number of tissue points to process in each batch when
        memory saving mode is enabled.

    Attributes
    ----------
    measure_coords : np.ndarray
        An array of points (x, y, z) where ECG signals are measured.
    ecg : list
        The computed ECG signals.
    memory_save : bool
        A flag to enable memory saving mode. If True, the tracker will compute
        distances on the fly instead of precomputing them. This mode is useful
        for large tissue meshes with high number of measurement points.
    dist_dtype : np.dtype
        The data type of the precomputed distances. To save memory, the
        distances are stored as float16 by default.
    batch_size : int
        The number of tissue points to process in each batch when memory saving
        mode is enabled.
    """

    def __init__(self, memory_save=False, batch_size=10, distance_power=1):
        """
        Initializes the ECG3DTracker with default parameters.
        
        Parameters
        ----------
        memory_save : bool, optional
            A flag to enable memory saving mode. If True, the tracker will
            compute distances on the fly instead of precomputing them. This
            mode is useful for large tissue meshes with high number of
            measurement points. The default is False.
        batch_size : int, optional
            The number of tissue points to process in each batch when memory
            saving mode is enabled. The default is 10.
        distance_power : int, optional
            The power to which the distance is raised in the calculation of the
            ECG signal. The default is 1.
        """
        super().__init__()
        self.measure_coords = np.array([[0, 0, 1]])
        self.ecg = []
        self.memory_save = memory_save
        self.dist_dtype = np.float16
        self.batch_size = batch_size
        self.file_name = "ecg.npy"
        self.distance_power = distance_power
        self.transmembrane_current = None

    def initialize(self, model):
        self.model = model
        self.measure_coords = np.atleast_2d(self.measure_coords)
        self.ecg = []
        self.tissue_mask = model.cardiac_tissue.mesh == 1
        self.transmembrane_current = np.zeros_like(model.u)

        if self.memory_save:
            self.tissue_coords = np.argwhere(self.tissue_mask)
            inds = np.arange(len(self.measure_coords))
            split_inds = inds[::self.batch_size][1:]
            self.splitted_coords = np.split(self.measure_coords, split_inds)
            return

        self.compute_distance()

    def compute_distance(self):
        self.distances = np.ones((len(self.measure_coords),
                                 np.count_nonzero(self.tissue_mask)),
                                 dtype=self.dist_dtype)

        tissue_coords = np.argwhere(self.tissue_mask)
        for i, point in enumerate(self.measure_coords):
            self.distances[i, :] = np.linalg.norm((point - tissue_coords),
                                                  axis=1
                                                  ).astype(self.dist_dtype)

        self.distances = self.distances ** self.distance_power

        if np.any(self.distances == 0):
            Warning("Measurement points are inside the tissue.")

    def uni_voltage(self, current):
        if self.memory_save:
            return self._uni_voltage_memory_save(current)

        return np.sum(current[self.tissue_mask] / self.distances, axis=1)

    def _uni_voltage_memory_save(self, current):
        """
        Compute the sum of the transmembrane current divided by the distance
        between the measurement points and the tissue points.

        Parameters
        ----------
        current : np.ndarray
            The transmembrane current array.

        Returns
        -------
        np.ndarray
            The computed potential difference at the measurement points.
        """
        ecg = []

        for coords in self.splitted_coords:
            distance = spatial.distance.cdist(coords, self.tissue_coords)
            distance = distance ** self.distance_power
            ecg.append(np.sum(current[self.tissue_mask] / distance, axis=1))

        return np.squeeze(np.column_stack(ecg))

    def calc_ecg(self):
        """
        Calculate the ECG signal at the measurement points.

        Returns
        -------
        np.ndarray
            The computed ECG signal.
        """
        # current = self.model.transmembrane_current
        self.model.diffusion_kernel(self.transmembrane_current,
                                    self.model.u,
                                    self.model.weights,
                                    self.model.cardiac_tissue.myo_indexes)
        current = self.transmembrane_current
        current[self.model.cardiac_tissue.mesh != 1] = 0
        return self.uni_voltage(current) / self.model.dr

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
        return np.array(self.ecg)

    def write(self):
        """
        Save the computed ECG signals to a file.

        The ECG signals are saved as a numpy array in the specified path.
        """
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True)

        np.save(Path(self.path, self.file_name), self.output)
