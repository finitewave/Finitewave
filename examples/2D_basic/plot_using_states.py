"""
StateKeeper example
====================

This example demonstrates how to use the StateKeeper class to save and load the
state of a simulation model. The model is run twice, and the state is saved
after the first run. The state is then loaded before the second run to continue
the simulation from the saved state.
"""

import matplotlib.pyplot as plt
import gc
import shutil

import finitewave as fw

# number of nodes on the side
# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                             n//2 - 3, n//2 + 3))

# set up state saver parameters:
# to save only one state you can use StateSaver directly
state_savers = fw.StateSaverCollection()
state_savers.savers.append(fw.StateSaver("state_0", time=10))
state_savers.savers.append(fw.StateSaver("state_1"))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 20
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.state_saver = state_savers

# run the model:
aliev_panfilov.run()

u_before = aliev_panfilov.u.copy()

# We delete model and use gc.collect() to ask the virtual machine remove
# objects from memory. Though it's not necessary to do this.
del aliev_panfilov
gc.collect()

# # # # # # # # #

# Here we create a new model and load states from the previous calculation to
# continue.


# recreate the model
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# add the tissue and the state_loader to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.state_loader = fw.StateLoader("state_0")

aliev_panfilov.run()
u_after = aliev_panfilov.u.copy()

# recreate the model
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# add the tissue and the state_loader to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.state_loader = fw.StateLoader("state_1")

aliev_panfilov.run()

# plot the results
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(u_before)
axs[0].set_title("First run from t=0")
axs[1].imshow(u_after)
axs[1].set_title("Second run from t=10")
axs[2].imshow(aliev_panfilov.u)
axs[2].set_title("Third run from t=20")
plt.show()

# remove the state directory
shutil.rmtree("state_0")
shutil.rmtree("state_1")
