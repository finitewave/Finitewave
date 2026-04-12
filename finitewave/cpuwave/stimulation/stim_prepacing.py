import numpy as np


class StimPrepacing:
    """Class representing a prepacing sequence."""
    def __init__(self, dt):
        """
        Initializes the StimPrepacing instance.

        Parameters
        ----------
        dt : float
            Time step for the prepacing sequence.
        """
        self.dt = dt
        self._stim_sequence = []

    def add_stim(self, n_beats, cycle_length, curr_value, duration):
        """
        Adds a stimulus to the prepacing sequence.
        
        Parameters
        ----------
        n_beats : int
            Number of beats for the prepacing sequence.
        cycle_length : float
            Length of each cycle in the prepacing sequence.
        duration : float
            Duration of the stimulus.
        curr_value : float
            Amplitude of the stimulus current.
        """
        steps = int(n_beats * cycle_length / self.dt)

        stim_values = np.zeros(steps, dtype=np.float64)

        for s in np.arange(n_beats):
            stim_start = s * cycle_length
            stim_end = stim_start + duration
            
            start_idx = int(stim_start / self.dt)
            end_idx = int(stim_end / self.dt)
            stim_values[start_idx: end_idx] = curr_value

        self._stim_sequence.append(stim_values)

    @property
    def stim_sequence(self):
        """
        Returns
        -------
        np.ndarray
            Concatenated stimulus values for each time step in the prepacing sequence.
        """
        return np.concatenate(self._stim_sequence)
