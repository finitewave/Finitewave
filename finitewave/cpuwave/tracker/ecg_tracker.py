from pathlib import Path
import numpy as np
import math

from finitewave.core.tracker.tracker import Tracker


class ECGTracker(Tracker):
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
    """

    def __init__(self, measure_coords=None, distance_power=1,
                 extracellular_conductivity=1.0, **kwargs):
        """Initialize the ECGTracker.
        
        Parameters
        ----------
        measure_coords : np.ndarray, optional
            An array of points (x, y, z) where ECG signals are measured.
        distance_power : float, optional
            The power to which the distance is raised in the inverse distance weighting. Default is 1.
        extracellular_conductivity : float, optional
            The extracellular conductivity. Default is 1.0.
        **kwargs
            Additional keyword arguments to pass to the base Tracker class.
        """
        super().__init__(**kwargs)
        self.measure_coords = measure_coords
        self.ecg = []
        self.file_name = "ecg.npy"
        self.distance_power = distance_power
        self.extracellular_conductivity = extracellular_conductivity

    def initialize(self, simulation):
        """
        Initialize the ECG tracker with the simulation object.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        super().initialize(simulation)
        self.simulation = simulation
        self.ecg_func = ecg_func(self.simulation.backend)
        self._measure_coords = self.build_measure_coords(self.measure_coords)
        self.ecg = []

        mesh_shape = self.simulation.cardiac_tissue.mesh.shape
        myo_indexes = self.simulation.cardiac_model.myo_indexes
        tissue_indexes = self.simulation.cardiac_model.tissue_indexes
        self.myo_indexes = tissue_indexes[myo_indexes]
        self.myo_coords = self.build_myo_coords(self.myo_indexes, mesh_shape)

    def build_myo_coords(self, myo_indexes, mesh_shape):
        myo_coords = np.unravel_index(myo_indexes, mesh_shape)

        if len(myo_coords) == 2:
            myo_coords = (myo_coords[0], myo_coords[1], np.zeros_like(myo_coords[0]))
        return self.simulation.backend.wrap(np.array(myo_coords))

    def build_measure_coords(self, coords):
        coords = np.atleast_2d(coords)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))))
        coords = self.simulation.backend.wrap(coords)
        return coords

    def build_myo_mask(self, myo_indexes, tissue_indexes):
        myo_mask = np.zeros_like(tissue_indexes, dtype=bool)
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
        u_old = self.simulation.solver.u_old
        u = self.simulation.cardiac_model.u
        rhs = self.simulation.solver.rhs
        dt = self.simulation.dt
        dr = self.simulation.cardiac_tissue.dr
        
        tr_current = (u - u_old - dt * rhs)
        tr_current = self.simulation.backend.select_values(tr_current, self.myo_indexes)
        ecg = self.ecg_func(self._measure_coords, tr_current, *self.myo_coords, dr, dt,
                            self.distance_power, self.extracellular_conductivity)
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


def ecg_func(backend):
    if backend.name == "numba":
        from numba import njit, prange

        @njit(parallel=True, fastmath=True)
        def calc_ecg_numba(coords, tr_current, i, j, k, dr, dt, distance_power=1.0, cond=1.0):

            n = coords.shape[0]
            ecg = np.empty(n, dtype=tr_current.dtype)

            for c in prange(n):
                x, y, z = coords[c]
                ds = (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2
                ds = np.where(ds == 0, 1, ds)
                d = np.sqrt(ds) ** distance_power
                ecg[c] = np.sum(tr_current / (d * dr * dt)) / (4 * math.pi * cond)

            return ecg
        
        return calc_ecg_numba
    
    if backend.name == "jax":
        import jax
        import jax.numpy as jnp

        @jax.jit
        def calc_ecg_jax(coords, tr_current, i, j, k, dr, dt, distance_power=1.0, cond=1.0):
            def single_ecg(_, coord):
                x, y, z = coord
                ds = jnp.maximum((i - x)**2 + (j - y)**2 + (k - z)**2, dr)
                d = jnp.sqrt(ds) ** distance_power
                result = jnp.sum(tr_current / (d * dt)) / (4 * jnp.pi * cond)
                return None, result

            # Scan loops on-device, keeping memory usage low
            _, ecg = jax.lax.scan(single_ecg, None, coords)
            return ecg
        
        return calc_ecg_jax
    
    if backend.name == "mlx":
        import mlx.core as mx
        import math

        @mx.compile
        def single_ecg(coord, tr_current, i, j, k, dr, dt, distance_power=1.0, cond=1.0):
            x, y, z = coord
            ds = mx.maximum((i - x)**2 + (j - y)**2 + (k - z)**2, dr)
            d = mx.sqrt(ds) ** distance_power
            result = mx.sum(tr_current / (d * dt)) / (4 * math.pi * cond)
            return result
        
        vmap_ecg = mx.vmap(single_ecg, in_axes=(0, None, None, None, None, None, None, None))
        compiled_ecg = mx.compile(vmap_ecg)


        def calc_ecg_mlx(coords, tr_current, i, j, k, dr, dt, distance_power=1.0, cond=1.0):
            dr = mx.array(dr)
            dt = mx.array(dt)
            distance_power = mx.array(distance_power)
            cond = mx.array(cond)

            num_coords = coords.shape[0]
            batch_size = 32
            ecg = mx.zeros(num_coords, dtype=tr_current.dtype)

            for start in range(0, num_coords, batch_size):
                end = min(start + batch_size, num_coords)
                batch_coords = coords[start:end]
                batch_results = compiled_ecg(batch_coords, tr_current, i, j, k, dr, distance_power, cond)
                ecg[start:end] = batch_results
                mx.eval(ecg)

            return ecg
        
        return calc_ecg_mlx