from copy import copy

import numpy as np

from finitewave.core.stimul.stim_sequence import StimSequence
from finitewave.core.stimul.stim_area.stim_electrodes import StimElectrodes


class StimGroupedElectrodes(StimSequence):
    """Group electrode activation times into a sequence of stimuli.

    Call :meth:`add_stim` with a stimulus template to create one stimulus per
    occupied time bin. Each stimulus targets all electrodes in its bin and
    starts at the earliest supplied activation time in that bin.
    """

    def __init__(self, times, coords, size, time_step=1.0, tree=None, mask=None):
        """Define the electrode locations and activation schedule.

        Parameters
        ----------
        times : array_like, shape (n_electrodes,)
            Finite activation times, in the same units as the simulation time.
            Each entry corresponds to the same row in ``coords``.
        coords : array_like, shape (n_electrodes, n_dimensions)
            Electrode centers in the tissue coordinate system.
        size : float
            Electrode radius, in the same spatial units as ``coords``.
        time_step : float, optional
            Positive, finite bin width. Bins are left-closed and right-open,
            starting at ``floor(min(times))``. Default is 1.0. This does not
            change the simulation time step.
        tree : scipy.spatial.KDTree, optional
            Spatial index whose points must match the myocardial coordinates
            after applying ``mask``, in the same order. If omitted, each
            electrode area builds its tree when initialized.
        mask : numpy.ndarray, optional
            Boolean mask or integer indices selecting myocardial nodes before
            the spatial query. If omitted, all myocardial nodes are considered.

        Notes
        -----
        An empty schedule creates no stimuli. Grouping and input validation
        occur when :meth:`add_stim` is called.

        Examples
        --------
        >>> from finitewave.core.stimul.stim_type.stim_current import StimCurrent
        >>> protocol = StimGroupedElectrodes(
        ...     times=[0.2, 0.8, 2.0],
        ...     coords=[[1., 1.], [2., 2.], [3., 3.]], size=0.5)
        >>> protocol.add_stim(StimCurrent(time=0, curr_value=1, duration=0.1)) is protocol
        True
        >>> [stim.t for stim in protocol.sequence]
        [0.2, 2.0]
        """
        super().__init__()
        self.times = times
        self.time_step = time_step
        self.coords = coords
        self.size = size
        self.tree = tree
        self.mask = mask

    def add_stim(self, stim):
        """Append a copy of a stimulus template for each occupied time bin.

        Parameters
        ----------
        stim : Stim
            Uninitialized stimulus template, such as ``StimCurrent`` or
            ``StimVoltage``. Its time and stimulation area are replaced in
            each copy; other parameters, including amplitude and duration,
            are retained. The template itself is not modified. Copies are
            shallow, so any other mutable attributes remain shared.

        Returns
        -------
        StimGroupedElectrodes
            This sequence, allowing chained calls. Repeated calls append
            additional stimuli without clearing the existing sequence.
        """
        point_groups, time_groups = self.build_stim_groups(
            self.times, self.coords, self.time_step)
        for points, times in zip(point_groups, time_groups):
            grouped_stim = copy(stim)
            grouped_stim.t = float(times.min())
            grouped_stim.passed = False
            grouped_stim.stim_area = StimElectrodes(
                points, self.size, tree=self.tree, mask=self.mask)
            super().add_stim(grouped_stim)
        return self

    def build_stim_groups(self, times, coords, time_step=1.0):
        """Group electrode coordinates and activation times by time bin.

        Parameters
        ----------
        times : array_like, shape (n_electrodes,)
            Finite activation times; input order need not be chronological.
        coords : array_like, shape (n_electrodes, n_dimensions)
            Electrode centers corresponding to ``times``.
        time_step : float, optional
            Positive, finite bin width, default 1.0. Bins start at
            ``floor(min(times))`` and include their left boundary only.

        Returns
        -------
        point_groups : list of numpy.ndarray
            Coordinates in each occupied bin, ordered by increasing bin time.
        time_groups : list of numpy.ndarray
            Original activation times corresponding to ``point_groups``.
            Input order is preserved within each bin. Empty input returns
            two empty lists.

        Raises
        ------
        ValueError
            If times are not a finite one-dimensional array, coordinates do
            not have one row per time, or the bin width is not positive and
            finite.
        """
        times = np.asarray(times, dtype=float)
        coords = np.asarray(coords)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("times must be a finite one-dimensional array")
        if coords.ndim != 2 or len(coords) != len(times):
            raise ValueError("coords must be a two-dimensional array with one row per time")
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be positive and finite")
        if times.size == 0:
            return [], []

        bin_ids = np.floor((times - np.floor(times.min())) / time_step)
        point_groups = []
        time_groups = []
        for bin_id in np.unique(bin_ids):
            selected = bin_ids == bin_id
            point_groups.append(coords[selected])
            time_groups.append(times[selected])
        return point_groups, time_groups
