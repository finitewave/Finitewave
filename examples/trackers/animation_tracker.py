
import matplotlib.pyplot as plt

import finitewave as fw
import numpy as np

# create a tissue of size 400x400 with cardiomycytes:
n = 200
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)
tissue.mesh[90:110, 90:110] = 0

# induce spiral wave:
stim_sequence = fw.StimS1S2Cross(tissue, s1_time=0, s2_time=35, voltage_value=1)

animation_tracker = fw.AnimationTracker(aggregate=True,
                                        path=".",
                                        dir_name="frames",
                                        var_name="u",
                                        output_dtype=np.float32,
                                        step=100,
                                        keep_shape=False)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(animation_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=100, backend="jax")
simulation.cardiac_model = fw.AlievPanfilov()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence
# run the model:
simulation.run()

animation_tracker.write(prog_bar=True, fps=12, clim=[0, 1], cmap="RdBu_r", clear=True,
                        upscale_factor=5)

