

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 100
m = 10
tissue = fw.CardiacTissue([n, m], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentCoord(0, 5, 0.1, 0, 3, 0, m))

# set up tracker parameters:
node_inds = [[50, 5], [51, 5]]
threshold_tracker = fw.ThresholdTracker(node_inds, threshold=0.5)
act_pot_tracker = fw.ActionPotentialTracker(node_inds)

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(threshold_tracker)
tracker_sequence.add_tracker(act_pot_tracker)

# set up simulation parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=50)
simulation.cardiac_model = fw.AlievPanfilov()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# plot the threshold crossing status
time = threshold_tracker.tracking_times

plt.figure()
plt.plot(act_pot_tracker.tracking_times, act_pot_tracker.act_pot)
plt.plot(threshold_tracker.tracking_times, 
         np.ones(len(threshold_tracker.tracking_times)) * threshold_tracker.threshold)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Threshold Crossing Status')
plt.show()