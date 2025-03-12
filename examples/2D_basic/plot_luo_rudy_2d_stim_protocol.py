"""
S1-S3 stimulation protocol in 2D Luo-Rudy model
===============================================

This example demonstrates how to stimulate the tissue using the S1-S3
stimulation protocol in the 2D Luo-Rudy model. Stimulation is applied from
a point source (electrode with a radius of 5).
"""

import matplotlib.pyplot as plt
import finitewave as fw

n = 256
# create mesh
tissue = fw.CardiacTissue2D((n, n))
tissue.add_pattern(fw.Diffuse2DPattern(0.3))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
for t in [0, 150, 290]:
    stim = fw.StimCurrentArea2D(t, 30, 1)
    stim.add_stim_point([5, n//2], tissue.mesh, size=5)
    stim_sequence.add_stim(stim)


# create model object and set up parameters
luo_rudy = fw.LuoRudy912D()
luo_rudy.dt = 0.01
luo_rudy.dr = 0.3
luo_rudy.t_max = 700

# add the tissue and the stim parameters to the model object
luo_rudy.cardiac_tissue = tissue
luo_rudy.stim_sequence = stim_sequence

# run the model
luo_rudy.run(num_of_theads=16)

plt.imshow(luo_rudy.u)
plt.colorbar()
plt.show()
