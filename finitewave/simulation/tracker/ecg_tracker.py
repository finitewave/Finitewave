from pathlib import Path
import math

import numpy as np
from scipy.spatial import KDTree

from finitewave.core.tracker.tracker import Tracker


class ECGTracker(Tracker):
    """Track the ECG signal at specified electrode positions.

    The ECG is computed as the approximate potential at the electrode positions
    due to the current sources in the tissue, using the formula:
    
        phi_e = 1 / (4 * pi * sigma) * sum(q_i / r_i)

    where
    ``phi_e`` is the potential at the electrode,
    ``sigma`` is the volume conductivity,
    ``q_i`` is the current source at tissue node ``i``, and
    ``r_i`` is the distance from tissue node ``i`` to the electrode.

    Parameters
    ----------
    lead_coords : array_like, shape (N, 2) or (N, 3)
        Electrode coordinates in grid units (for grid tissue) or physical
        coordinates (for element-mesh tissue).
    volume_conductivity : float
        The conductivity of the volume conductor surrounding the tissue.
    min_distance : float, optional
        Minimum distance for inverse-distance weighting. If None,
        defaults to 0.5 * dr for grid tissue or 0.1 mm for element-mesh tissue.
    **kwargs
        Additional keyword arguments passed to the tracker class.

    Attributes
    ----------
    diffusion_operator : backend-specific sparse matrix
        The diffusion operator K used to compute the current sources ``q = K @ u``.
        If None, the diffusion operator is computed from the simulation's spatial discretization.

    Notes
    -----
    By default, the ECG calculation uses the spatial discretization operator K,
    which is based on monodomain diffusion coefficients. ECG source currents,
    however, depend on intracellular conductivity.

    If the intracellular and monodomain operators differ by a scalar factor
    ``alpha = D_intracellular / D_monodomain``, this factor can be accounted for
    by dividing ``volume_conductivity`` by ``alpha``. A scalar correction is only
    valid when the two operators are proportional.
    """

    def __init__(self, lead_coords=None, volume_conductivity=1.0, min_distance=None, **kwargs):
        super().__init__(**kwargs)
        self.lead_coords = lead_coords
        self.volume_conductivity = volume_conductivity
        self.min_distance = min_distance
        self.diffusion_operator = None
        self.ecg = []
        self.file_name = "ecg.npy"

    def initialize(self, simulation):
        super().initialize(simulation)
        backend = simulation.backend
        tissue = simulation.cardiac_tissue

        if (not np.isfinite(self.volume_conductivity)
                or self.volume_conductivity <= 0):
            raise ValueError("volume_conductivity must be finite and positive.")

        is_grid = tissue.meta["type"] == "Grid"
        scale = tissue.dr if is_grid else 1.0

        self._min_distance = self.min_distance
        if self._min_distance is None:
            self._min_distance = 0.5 * scale if is_grid else 0.1

        if not np.isfinite(self._min_distance) or self._min_distance <= 0:
            raise ValueError("min_distance must be finite and positive.")

        source_coords = tissue.tissue_coords * scale
        source_coords = np.pad(source_coords, ((0, 0), (0, 3 - source_coords.shape[1])))
        lead_coords = self.build_lead_coords(self.lead_coords, source_coords, scale)

        self._lead_coords = backend.wrap_array(lead_coords)
        self._source_coords = backend.wrap_array(source_coords)
        source_mask = np.zeros(source_coords.shape[0], dtype=bool)
        source_mask[tissue.myo_indexes] = True
        self._source_mask = backend.wrap_mask(source_mask)
        self._ecg_scale = (scale ** 3 / (4 * math.pi * self.volume_conductivity))

        if self.diffusion_operator is None:
            K, _ = simulation.spatial_discretization.weights
            self._diffusion_operator = backend.wrap_sparse(K)
        else:
            self._diffusion_operator = self.diffusion_operator
        self.ecg_func = ecg_func(backend)
        self.ecg = []
        self._tracking_times = []
        self.tracking_counter = 0

    def build_lead_coords(self, coords, nodes, scale):
        """Validate and pad electrode coordinates to three dimensions."""

        if coords is None:
            raise ValueError("lead_coords must contain at least one electrode.")

        coords = np.atleast_2d(np.asarray(coords, dtype=float)) * scale
        if (coords.ndim != 2 or coords.shape[0] == 0
                or not 2 <= coords.shape[1] <= 3 or not np.all(np.isfinite(coords))):
            raise ValueError("lead_coords must be a finite (N, 2), or (N, 3) array.")

        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
        tree = KDTree(nodes)
        distances, _ = tree.query(coords, k=1, workers=-1)

        if np.any(distances < self._min_distance):
            msg = (f"Some electrodes are closer than min_distance = {self._min_distance} "
                   "to the tissue nodes.")
            raise ValueError(msg)

        return coords

    def calc_ecg(self):
        """Compute the electrode potentials from the current voltage state."""
        u = self.simulation.cardiac_model._u
        return self.ecg_func(
            self._diffusion_operator, u, self._source_mask, self._lead_coords,
            self._source_coords, self._ecg_scale)

    def _track(self):
        # Keep only small host-side electrode outputs in the recording history.
        self.ecg.append(np.asarray(self.calc_ecg()).copy())

    @property
    def output(self):
        return np.squeeze(np.asarray(self.ecg))

    def write(self, path=None):
        """Save the ECG history, defaulting to ``self.path`` or the current directory."""
        path = Path(path if path is not None else getattr(self, "path", "."))
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / self.file_name, self.output)


def ecg_func(backend):
    """Build a backend evaluator, sharing one K @ u across all electrodes."""
    matvec = backend.linalg.matvec
    if backend.name == "numba":
        from numba import njit, prange

        @njit(parallel=True)
        def reduce_ecg(q, mask, electrodes, nodes, scale_coef):
            out = np.empty(electrodes.shape[0], dtype=q.dtype)
            for e in prange(electrodes.shape[0]):
                result = 0.0
                for i in range(mask.size):
                    if mask[i]:
                        dx = nodes[i] - electrodes[e]
                        r = math.sqrt(np.sum(dx**2))
                        result += q[i] / r
                out[e] = result * scale_coef
            return out

        def calculate(K, u, mask, electrodes, nodes, scale_coef):
            q = matvec(K, u)
            # The CSR matvec leaves empty rows unwritten. Their product is zero.
            q[np.diff(K[0]) == 0] = 0.0
            return reduce_ecg(q, mask, electrodes, nodes, scale_coef)
        return calculate

    if backend.name == "jax":
        import jax
        import jax.numpy as jnp

        @jax.jit
        def calculate(K, u, mask, electrodes, nodes, scale_coef):
            q = matvec(K, u)

            def single_ecg(carry, electrode):
                r = jnp.sqrt(jnp.sum((nodes - electrode)**2, axis=1))

                return carry, jnp.sum(jnp.where(mask, q / r, 0.0)) * scale_coef

            return jax.lax.scan(single_ecg, None, electrodes)[1]

        return calculate

    if backend.name == "mlx":
        import mlx.core as mx

        @mx.compile
        def single_ecg(q, electrode, nodes, mask, scale_coef):
            r = mx.sqrt(mx.sum((nodes - electrode)**2, axis=1))
            return mx.sum(mx.where(mask, q / r, 0.0)) * scale_coef

        def calculate(K, u, mask, electrodes, nodes, scale_coef):
            q = matvec(K, u)
            mx.eval(q)
            results = []
            for e in range(electrodes.shape[0]):
                result = single_ecg(q, electrodes[e], nodes, mask, scale_coef)
                mx.eval(result)
                results.append(result)
            return mx.stack(results)
        return calculate

    raise ValueError(f"Unsupported ECG backend: {backend.name}")
