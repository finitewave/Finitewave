
"""
Courtemanche2D (Iso)
==========================

This example demonstrates how to use the Courtemanche model in 2D with
isotropic stencil.
"""

import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue of size 400x400 with cardiomycytes:
n = 300
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                             n//2 - 3, n//2 + 3))

# create model object and set up parameters:
aliev_panfilov = fw.Courtemanche2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()
