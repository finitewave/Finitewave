from finitewave.gridywave.cpuwave2D.stimulation.stim_voltage_coord_2d import (
    StimVoltageCoord2D
)


class StimVoltageCoord3D(StimVoltageCoord2D):
    def __init__(self, time, volt_value, x1, x2, y1, y2, z1, z2):
        super().__init__(time, volt_value, x1, x2, y1, y2)
        self.z1 = z1
        self.z2 = z2

    def initialize(self, simulation):
        super().initialize(simulation)
        self.slices = (slice(self.x1, self.x2),
                       slice(self.y1, self.y2),
                       slice(self.z1, self.z2))
