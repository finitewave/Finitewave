import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue of size 300x300 with cardiomycytes:
n = 100
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))
stim_sequence.add_stim(fw.StimVoltageCoord2D(400, 1, 0, n, 0, 5))

# create model object and set up parameters:
mitchell_schaeffer = fw.MitchellSchaeffer2D()
mitchell_schaeffer.dt = 0.01
mitchell_schaeffer.dr = 0.25
mitchell_schaeffer.t_max = 900
# add the tissue and the stim parameters to the model object:
mitchell_schaeffer.cardiac_tissue = tissue
mitchell_schaeffer.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[30, 30], [70, 70]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
mitchell_schaeffer.tracker_sequence = tracker_sequence

# run the model:
mitchell_schaeffer.run()

# plot the action potential
import numpy as np
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * mitchell_schaeffer.dt
plt.plot(time, action_pot_tracker.output[:, 0], label="cell_30_30")
plt.plot(time, action_pot_tracker.output[:, 1], label="cell_70_70")
plt.legend(title='Mitchell-Schaeffer')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Action Potential')
plt.grid()
plt.show()

# show the potential map at the end of calculations:
# plt.figure()
# plt.imshow(mitchell_schaeffer.u)
# plt.colorbar()
# plt.show()