import numpy as np
from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoordFEM(StimVoltage):
    def __init__(self, time, volt_value, x1, x2, y1, y2, z1=0, z2=0):
        super().__init__(time, volt_value)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

    def initialize(self, simulation):
        super().initialize(simulation)
        tissue = simulation.cardiac_tissue
        mask = np.ones(tissue.coords.shape[0], dtype=bool)
        mask &= ((tissue.coords[:, 0] >= self.x1) &
                 (tissue.coords[:, 0] <= self.x2))
        mask &= ((tissue.coords[:, 1] >= self.y1) &
                 (tissue.coords[:, 1] <= self.y2))
        mask &= ((tissue.coords[:, 2] >= self.z1) &
                 (tissue.coords[:, 2] <= self.z2))
        self.mask = np.zeros_like(mask)
        self.mask[tissue.myo_indexes] = mask[tissue.myo_indexes]

    def stimulate(self, simulation):
        simulation.cardiac_model.u[self.mask] = self.volt_value
