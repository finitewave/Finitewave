import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# stimulate the corner of the tissue with a square pulse (10 nodes on the side)
# of 1.0 V at t=0.
# coordinates are always form a reactangular (slab in 3D) area of stimulation.
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, 
                                             volt_value=1.0, 
                                             x1=0, x2=10, 
                                             y1=0, y2=10))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 25
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