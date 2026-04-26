
import matplotlib.pyplot as plt
import finitewave as fw


# create model object and set up parameters
cardiac_model = fw.AlievPanfilov()

# create a tissue of size 300x300 with cardiomycytes:
n = 300
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)

threshold_tracker = fw.LowThresholdTracker(node_inds=[n//2, n//2], threshold=0.03)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(threshold_tracker)

# set up stimulation parameters:
stim_s1 = fw.StimVoltageCoord(0, 1, x_min=0, x_max=n, y_min=0, y_max=5)
stim_s2 = fw.StimVoltageCoord(0, 1, x_min=n//2-5, x_max=n//2+5, y_min=0, y_max=n//2)

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim_s1)
stim_sequence.add_stim(fw.StimAdaptiveTime(stim_s2, threshold_tracker, delay=3))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=55, backend="mlx")
simulation.cardiac_model = cardiac_model
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

u = simulation.cardiac_model.output("u")

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(u, cmap="inferno")
plt.colorbar(label="Membrane Potential")
plt.show()
