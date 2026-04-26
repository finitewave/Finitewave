

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
tissue = fw.CardiacTissueGrid([n, n], dr=0.3)

model = fw.MitchellSchaeffer()

# induce spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=0, x_max=n, 
                                           y_min=0, y_max=5))
# set up the tracker:
lat_tracker = fw.ActivationTimeTracker(threshold=0.5, step=10)

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(lat_tracker)

# set up the simulation:
simulation = fw.CardiacSimulation(backend="mlx")
simulation.dt = 0.01
simulation.t_max = 100
simulation.cardiac_tissue = tissue
simulation.cardiac_model = model
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the simulation:
simulation.run()

plt.figure()
plt.imshow(lat_tracker.output, cmap="hsv", origin="lower")
plt.colorbar(label="Activation Time (ms)")
plt.title("Activation Time Map")
plt.show()
