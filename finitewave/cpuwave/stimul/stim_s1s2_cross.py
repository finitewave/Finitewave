
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

    def __init__(self, cardiac_tissue, s1_time, s2_time, voltage_value):
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
        """
        super().__init__()

        n, m = cardiac_tissue.mesh.shape[:2]

        self.add_stim(StimVoltageCoord(s1_time, voltage_value,
                                       x_min=0, x_max=n//2,
                                       y_min=0, y_max=m))
        self.add_stim(StimVoltageCoord(s2_time, voltage_value,
                                       x_min=0, x_max=n,
                                       y_min=0, y_max=m//2))
