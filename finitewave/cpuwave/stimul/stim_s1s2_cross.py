
import numpy as np
from finitewave.core.stimul.stim_sequence import StimSequence
from finitewave.core.stimul.stim_impl.stim_voltage_coord import StimVoltageCoord


class StimS1S2Cross(StimSequence):
    """
    Stimulus sequence for S1-S2 cross-field stimulation protocol.

    This class defines a stimulus sequence that applies two stimuli (S1 and S2)
    in a cross-field pattern. The S1 stimulus is applied to a rectangular region
    of the tissue, while the S2 stimulus is applied to a different rectangular
    region after a specified delay.
    """

    def __init__(self, cardiac_tissue, s1_time, s2_time, voltage_value, axes=[0, 1]):
        """Initialize the S1-S2 cross-field stimulus sequence.

        Parameters
        ----------
        cardiac_tissue : CardiacTissueGrid
            The cardiac tissue grid on which to apply the stimuli.
        s1_time : float
            The time at which the S1 stimulus is applied.
        s2_time : float
            The time at which the S2 stimulus is applied.
        voltage_value : float
            The voltage value for both stimuli.
        axes : list of int, optional
            The axes along which to apply the stimuli (default is [0, 1]).
        """
        super().__init__()

        shape = cardiac_tissue.mesh.shape

        s1_region = np.zeros(2 * len(shape), dtype=int)
        s1_region[1::2] = shape
        s2_region = np.zeros(2 * len(shape), dtype=int)
        s2_region[1::2] = shape

        s1_region[2 * axes[0] + 1] = shape[axes[0]]//2
        s2_region[2 * axes[1] + 1] = shape[axes[1]]//2

        self.add_stim(StimVoltageCoord(s1_time, voltage_value, *s1_region))
        self.add_stim(StimVoltageCoord(s2_time, voltage_value, *s2_region))
