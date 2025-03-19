
#
# The basic example of running simple simuations with the Aliev-Panfilov model.
# The model is a 2D model with isotropic stencil.
# The model is stimulated with a voltage pulse in the center of the tissue.
# Conductivity is set to 0.3 in the center of the tissue - this will deform the wavefront at the top of the square due to the slow propagation.
# 

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw


# number of nodes on the side
n = 100

tissue = fw.CardiacTissue3D([n, n, 10])
# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.conductivity = np.ones([n, n, 10])
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 1, 5, 1, n-1, 1, 9))

# create model object:
model = fw.TP063D()
# set up numerical parameters:
model.dt = 0.01
model.dr = 0.25
model.t_max = 400
# add the tissue and the stim parameters to the model object:
model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# measure action potential for cells 30, 30 and 40, 40
action_pot_tracker.cell_ind = [[30, 30, 5], [40, 40, 5]]
tracker_sequence.add_tracker(action_pot_tracker)

model.tracker_sequence = tracker_sequence

model.run()

# show the potential map at the end of calculations:
plt.imshow(model.u[:, :, 5])
plt.show()

# plot the action potential
time = np.arange(len(action_pot_tracker.output)) * model.dt
plt.plot(time, action_pot_tracker.output[:, 0], label="cell_30_30_5")
plt.plot(time, action_pot_tracker.output[:, 1], label="cell_40_40_5")
plt.legend(title='Model')
plt.show()