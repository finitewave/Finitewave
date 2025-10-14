import numpy as np
from scipy import spatial
from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageElectrodes(StimVoltage):
    def __init__(self, time, volt_value, coords, size):
        super().__init__(time, volt_value)
        self.coords = coords
        self.size = size

    def initialize(self, simulation):
        super().initialize(simulation)
        myo_indexes = simulation.cardiac_model.myo_indexes.astype(np.int32)
        mesh_index = simulation.cardiac_model.mesh_indexes
        myo_coords = simulation.cardiac_tissue.coords[mesh_index[myo_indexes]]
        self.indexes = self.select_nodes(myo_coords, myo_indexes)

    def select_nodes(self, myo_coords, myo_indexes):
        tree = spatial.KDTree(myo_coords)
        inds = tree.query_ball_point(self.coords, self.size)
        inds = np.unique(np.concatenate(inds))
        return myo_indexes[inds]

    def stimulate(self, simulation):
        simulation.cardiac_model.u.flat[self.indexes] = self.volt_value
