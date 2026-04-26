import numpy as np


class StimSingleCell:
    """Class representing a single cell stimulus.
    
    Attributes
    ----------
    dt : float
        Time step for the single cell simulation.
    stim_current : np.ndarray
        The stimulus current sequence for the single cell simulation.
        The array size corresponds to the number of time steps in the simulation.
    """
    def __init__(self, dt):
        """
        Initializes the StimSingleCell instance.

        Parameters
        ----------
        dt : float
            Time step for the single cell simulation.
        """
        self.dt = dt
        self._stim_current = []

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

        stim_current = np.zeros(steps, dtype=np.float64)

        for s in np.arange(n_beats):
            stim_start = s * cycle_length
            stim_end = stim_start + duration
            
            start_idx = int(stim_start / self.dt)
            end_idx = int(stim_end / self.dt)
            stim_current[start_idx: end_idx] = curr_value

        self._stim_current.append(stim_current)

    @property
    def stim_current(self):
        """
        Returns
        -------
        np.ndarray
            Concatenated stimulus values for each time step in the prepacing sequence.
        """
        return np.concatenate(self._stim_current)
    
    @stim_current.setter
    def stim_current(self, stim_current):
        """
        Sets the stimulus current sequence.
        
        Parameters
        ----------
        stim_current : np.ndarray
            The stimulus current sequence.
        """
        self._stim_current = stim_current
