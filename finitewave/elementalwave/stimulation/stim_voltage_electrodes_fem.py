import numpy as np
from scipy import spatial
from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageElectrodesFEM(StimVoltage):
    def __init__(self, time, volt_value, coords, size):
        super().__init__(time, volt_value)
        self.coords = coords
        self.size = size

    def initialize(self, model):
        self.stim_mask = self.locate_electrodes(model.cardiac_tissue)
        super().initialize(model)

    def stimulate(self, model):
        model.diffusion_model.u[self.stim_mask] = self.volt_value

    def locate_electrodes(self, cardiac_tissue):
        tissue_coords = cardiac_tissue.myo_coords
        tree = spatial.KDTree(tissue_coords)
        inds = tree.query_ball_point(self.coords, self.size)
        inds = np.unique(np.concatenate(np.atleast_2d(inds)))
        mask = np.zeros(tissue_coords.shape[0], dtype=bool)
        mask[inds] = True
        return mask
