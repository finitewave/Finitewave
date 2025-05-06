
import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue:
n = 100
k = 5
m = 3
tissue = fw.CardiacTissue3D([n, m, k])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 5, 0, m, 0, k))

# create model object and set up parameters:
bueno_orovio = fw.BuenoOrovio3D()
bueno_orovio.dt = 0.01
bueno_orovio.dr = 0.25
bueno_orovio.t_max = 500
# add the tissue and the stim parameters to the model object:
bueno_orovio.cardiac_tissue = tissue
bueno_orovio.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3, 1]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
bueno_orovio.tracker_sequence = tracker_sequence

# run the model:
bueno_orovio.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * bueno_orovio.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3_1")
plt.legend(title='Bueno-Orovio')
plt.xlabel('Time (ms)')
plt.title('Action Potential')
plt.grid()
plt.show()
