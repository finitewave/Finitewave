from pathlib import Path
import numpy as np
import math

from finitewave.core.tracker.tracker import Tracker


class EGMTracker(Tracker):
    """
    A class to compute and track electrocardiogram (egm) signals from a 3D
    cardiac tissue model simulation.

    This tracker calculates EGM signals at specified measurement points by
    computing the potential differences across the cardiac tissue mesh and
    considering the inverse of the distance from each measurement point.

    Attributes
    ----------
    lead_coords : np.ndarray
        An array of points (x, y, z) where EGM signals are measured.
    egm : list
        The computed EGM signals.
    file_name : str
        The name of the file to save the computed EGM signals.
    """

    def __init__(self, lead_coords=None, lead_fields=None, conductivity=1.0, **kwargs):
        """Initialize the EGMTracker.
        
        Parameters
        ----------
        lead_coords : np.ndarray, optional
            An array of points (x, y, z) where EGM signals are measured.
        lead_fields : np.ndarray, optional
            The field of leads for EGM signal calculation.
        conductivity : float, optional
            The conductivity of the medium. Default is 1.0.
        **kwargs
            Additional keyword arguments to pass to the base Tracker class.
        """
        super().__init__(**kwargs)
        self.lead_coords = lead_coords
        self.egms = []
        self.file_name = "egm.npy"
        self.lead_fields = lead_fields
        self.conductivity = conductivity

    def initialize(self, simulation):
        """
        Initialize the EGM tracker with the simulation object.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        super().initialize(simulation)
        self.simulation = simulation
        self.egms = []

        tissue_coords = self.simulation.cardiac_tissue.tissue_coords

        self.egm_func = egm_func(self.simulation.backend)

        if self.lead_fields is None:
            # tissue_coords = tissue_coords * self.simulation.cardiac_tissue.dr
            # lead_coords = np.atleast_2d(self.lead_coords) * self.simulation.cardiac_tissue.dr
            dr = self.simulation.cardiac_tissue.dr
            self.lead_fields = self.calc_lead_fields(tissue_coords, self.lead_coords, dr, self.conductivity)
            self.lead_fields[:, simulation.cardiac_tissue.fibro_indexes] = 0.0
        
        self.lead_fields = self.simulation.backend.wrap(np.array(self.lead_fields))

    def calc_lead_fields(self, tissue_coords, lead_coords, dr, conductivity=1.0):
        """
        Calculate the lead fields for the EGM signals based on the measurement
        coordinates and the cardiac tissue mesh.

        Returns
        -------
        np.ndarray
            The computed lead fields.
        """
        if tissue_coords.shape[1] == 2:
            tissue_coords = np.hstack([tissue_coords, np.zeros((tissue_coords.shape[0], 1))])

        lead_coords = np.atleast_2d(self.lead_coords)
        if lead_coords.shape[1] == 2:
            lead_coords = np.hstack([lead_coords, np.zeros((lead_coords.shape[0], 1))])

        distances = calc_distances(tissue_coords, lead_coords)
        distances[distances < 0.5] = 1.
        lead_fields = dr ** 3 / (4 * math.pi * conductivity * dr * distances)

        return lead_fields

    def calc_egm(self):
        """
        Calculate the EGM signal at the measurement points.

        Returns
        -------
        np.ndarray
            The computed egm signal.
        """
        u_old = self.simulation.solver.u_old
        u = self.simulation.cardiac_model.u
        rhs = self.simulation.solver.rhs
        dt = self.simulation.dt
        
        egm = self.egm_func(u, u_old, rhs, dt, self.lead_fields)
        return egm

    def _track(self):
        egm = self.calc_egm()
        self.egms.append(egm)

    @property
    def output(self):
        """
        Get the computed egm signals as a numpy array.

        Returns
        -------
        np.ndarray
            The computed egm signals.
        """
        return np.squeeze(self.egms)

    def write(self):
        """
        Save the computed egm signals to a file.

        The egm signals are saved as a numpy array in the specified path.
        """
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True)

        np.save(Path(self.path, self.file_name), self.output)


def calc_distances(tissue_coords, lead_coords):
    """
    Calculate the distances from measurement points to myocyte coordinates.

    Parameters
    ----------
    tissue_coords : np.ndarray
        The coordinates of the myocytes in the tissue mesh.
    lead_coords : np.ndarray
        The coordinates of the measurement points.

    Returns
    -------
    np.ndarray
        The computed distances.
    """
    n_tissue = tissue_coords.shape[0]
    n_leads = lead_coords.shape[0]
    distances = np.empty((n_leads, n_tissue), dtype=np.float64)

    for i in range(n_leads):
        distances[i, :] = np.linalg.norm(tissue_coords - lead_coords[i], axis=1)

    return distances


def egm_func(backend):
    if backend.name == "numba":
        from numba import njit, prange

        @njit(parallel=True, fastmath=True)
        def calc_egm_numba(u, u_old, rhs, dt, lead_field):
            tr_current = u - u_old - dt * rhs

            n = lead_field.shape[0]
            egms = np.empty(n, dtype=tr_current.dtype)

            for i in prange(n):
                egms[i] = np.sum(tr_current * lead_field[i, :]) / dt

            return egms
        
        return calc_egm_numba
    
    if backend.name == "jax":
        import jax
        import jax.numpy as jnp

        @jax.jit
        def calc_egm_jax(u, u_old, rhs, dt, lead_field):
            tr_current = u - u_old - dt * rhs
            egms = jnp.sum(tr_current[jnp.newaxis, :] * lead_field, axis=1) / dt
            return egms

        return calc_egm_jax

    if backend.name == "mlx":
        import mlx.core as mx
        
        @mx.compile
        def calc_egm_compiled(u, u_old, rhs, dt, lead_field):
            tr_current = u - u_old - dt * rhs
            egms = mx.sum(tr_current[mx.newaxis, :] * lead_field, axis=1) / dt
            return egms
        
        def calc_egm_mlx(u, u_old, rhs, dt, lead_field):
            egms = calc_egm_compiled(u, u_old, rhs, dt, lead_field)
            mx.eval(egms)
            return egms

        return calc_egm_mlx
