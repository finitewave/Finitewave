
"""
Courtemanche2D (Iso)
==========================

This example demonstrates how to use the Courtemanche model in 2D with
isotropic stencil.
"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw
import os

# create a tissue of size 400x400 with cardiomycytes:
n = 300
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, n//5))
stim_sequence.add_stim(fw.StimVoltageCoord2D(330, 1, 0, n//3, 0, n))


# create model object and set up parameters:
courtemanche = fw.Courtemanche2D()
courtemanche.dt = 0.01
courtemanche.dr = 0.25
courtemanche.t_max = 1000
# add the tissue and the stim parameters to the model object:
courtemanche.cardiac_tissue = tissue
courtemanche.stim_sequence = stim_sequence

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
# add the multi variable tracker:
multivariable_tracker = fw.MultiVariable2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
multivariable_tracker.cell_ind = [15, 15]
multivariable_tracker.var_list = ["u"]
tracker_sequence.add_tracker(multivariable_tracker)

animation_tracker = fw.Animation2DTracker()
animation_tracker.variable_name = "u"  # Specify the variable to track
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 10
animation_tracker.overwrite = True  # Remove existing files in dir_name
tracker_sequence.add_tracker(animation_tracker)

courtemanche.tracker_sequence = tracker_sequence

# run the model:
courtemanche.run()

animation_tracker.write(shape_scale=5, clear=True, fps=30, clim=[-85, 20])

# plot the action potential
time = np.arange(len(multivariable_tracker.output["u"])) * courtemanche.dt

# plt.imshow(courtemanche.u)
# plt.show()

# var = []
# with open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/var.txt", "r") as f:
#     var = f.readlines()
    
# file_path = "/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/var.txt"
# if os.path.exists(file_path):
#     os.remove(file_path)
# else:
#     print("The file does not exist")


# var = [float(v) for v in var]

# plt.figure()
# plt.plot(var)
# plt.legend(title='Courtemanche')
# plt.show()

plt.figure()
plt.plot(time, multivariable_tracker.output["u"], label="u")
plt.legend(title='Courtemanche')
plt.show()
