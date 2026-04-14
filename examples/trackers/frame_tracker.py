
import matplotlib.pyplot as plt

import finitewave as fw
import numpy as np

# create a tissue of size 400x400 with cardiomycytes:
n = 200
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 3, x_max=n//2 + 3,
                                           y_min=n//2 - 3, y_max=n//2 + 3))

frame_tracker = fw.FrameTracker(aggregate=True, var_name="u",
                                output_dtype=np.float32, step=1000)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(frame_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 31
simulation.cardiac_model = fw.BuenoOrovio()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence
# run the model:
simulation.run()

# show the potential map at the end of calculations:
fig, axs = plt.subplots(ncols=2, nrows=2)
for i, ax in enumerate(axs.flat):
    ax.imshow(frame_tracker.output[i])
    ax.set_title(f"t = {frame_tracker.tracking_times[i]:.2f} ms")
plt.tight_layout()
plt.show()
