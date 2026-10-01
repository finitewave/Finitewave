"""ECG tracking with precomputed electrode-by-node lead fields."""

import numpy as np

from finitewave.core.tracker.tracker import Tracker
from .ecg_tracker import ECGTracker


class LeadFieldECGTracker(ECGTracker):
    """Compute ``lead_fields @ (K @ u)`` over active myocytes.
    IF the number of electrodes is small (the memory is enough to store the lead fields),
    this is faster than computing the ECG on-the-fly.

    Parameters
    ----------
    lead_coords : array_like, shape (N, 2) or (N, 3)
        Electrode coordinates in grid units (for grid tissue) or physical
        coordinates (for element-mesh tissue). For detailes refer to the ``ECGTracker`` class.
    lead_fields : array_like, shape (N, M)
        Precomputed lead fields for each electrode (N) and tissue node (M).
        If provided, ``lead_coords`` is ignored. The lead fields should be computed
        using the same tissue coordinates and volume conductivity as the simulation.
    volume_conductivity : float
        The conductivity of the volume conductor surrounding the tissue.
    min_distance : float, optional
        Minimum distance for inverse-distance weighting. If None,
        defaults to 0.5 * dr for grid tissue or 0.1 mm for element-mesh tissue.
    mono_to_intra_ratio : float, optional
        Ratio ``D_intracellular / D_monodomain`` (default 2.0). This is used
        to scale the diffusion operator K to approximate intracellular currents.
    **kwargs
        Additional keyword arguments passed to the tracker class.
    """

    def __init__(self, lead_coords=None, lead_fields=None,
                 volume_conductivity=1.0, min_distance=None,
                 mono_to_intra_ratio=2.0, **kwargs):
        super().__init__(lead_coords=lead_coords,
                         volume_conductivity=volume_conductivity,
                         min_distance=min_distance,
                         mono_to_intra_ratio=mono_to_intra_ratio, **kwargs)
        self.lead_fields = lead_fields
        self.file_name = "lead_field_ecg.npy"

    def initialize(self, simulation):
        backend = simulation.backend

        if self.lead_fields is not None:
            fields = self.initialize_from_fields(simulation)
        elif self.lead_coords is not None:
            fields = self.initialize_from_coords(simulation)
        else:
            raise ValueError("Must provide either lead_coords or lead_fields.")

        self._lead_fields = backend.wrap_array(fields)
        self.ecg_func = build_ecg_func(backend)

    def initialize_from_coords(self, simulation):
        """Initialize the tracker using electrode coordinates."""
        super().initialize(simulation)
        nodes = np.asarray(self._source_coords)
        electrodes = np.asarray(self._lead_coords)

        fields = np.empty((len(electrodes), len(nodes)))
        for i, electrode in enumerate(electrodes):
            fields[i] = self._ecg_scale / np.linalg.norm(nodes - electrode, axis=1)
        return fields

    def initialize_from_fields(self, simulation):
        """Initialize the tracker using precomputed lead fields."""
        Tracker.initialize(self, simulation)
        scale = self.compute_scaling_factor(simulation)
        fields = np.atleast_2d(np.asarray(self.lead_fields))
        self.build_diffusion_operator(simulation, scale)

        self.ecg = []
        self._tracking_times = []
        self.tracking_counter = 0

        return fields

    def calc_ecg(self):
        return self.ecg_func(self._diffusion_operator,
                             self.simulation.cardiac_model._u,
                             self._lead_fields)


def build_ecg_func(backend):
    """Compute the sparse source product once, then a dense lead-field product."""
    matvec = backend.linalg.matvec
    if backend.name == "numba":
        def calculate(K, u, fields):
            q = matvec(K, u)
            return fields @ q
        return calculate

    if backend.name == "jax":
        import jax

        @jax.jit
        def calculate(K, u, fields):
            return fields @ matvec(K, u)
        return calculate

    if backend.name == "mlx":
        import mlx.core as mx

        @mx.compile
        def calculate(K, u, fields):
            return fields @ matvec(K, u)
        return calculate
