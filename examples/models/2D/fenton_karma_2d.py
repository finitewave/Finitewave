
import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue:
n = 100
m = 5
tissue = fw.CardiacTissue2D([n, m])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 5, 0, m))

# create model object and set up parameters:
fentom_karma = fw.FentonKarma2D()
fentom_karma.dt = 0.01
fentom_karma.dr = 0.25
fentom_karma.t_max = 500
# add the tissue and the stim parameters to the model object:
fentom_karma.cardiac_tissue = tissue
fentom_karma.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
fentom_karma.tracker_sequence = tracker_sequence

# run the model:
fentom_karma.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * fentom_karma.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3")
plt.legend(title='Fenton-Karma')
plt.xlabel('Time (ms)')
plt.title('Action Potential')
plt.grid()
plt.show()
