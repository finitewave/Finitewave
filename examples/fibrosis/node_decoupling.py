
import matplotlib.pyplot as plt

import finitewave as fw
import numpy as np

# create a tissue of size 400x400 with cardiomycytes:
n, m = 100, 160
tissue = fw.CardiacTissueGrid([n, m], dr=0.5)

y = np.arange(m//8, 3 * m//8)
x = np.full(len(y), n//2)
line = np.array([x, y]).T

tissue.add_pattern(fw.DecouplingPattern(line, axis=0))
tissue.add_pattern(fw.DecouplingPattern(density=0.4, axis=None, region=[[n//4, 3*n//4], [5*m//8, 7*m//8]]))

# set up stimulation parameters:
stim_sequence = fw.StimSequence()

for i in range(5):
    stim_time = 21 * i
    stim_sequence.add_stim(fw.StimVoltageCoord(time=stim_time, volt_value=1,
                                               x_min=0, x_max=5,
                                               y_min=0, y_max=m))
prepacing = fw.StimSingleCell(dt=0.01)
prepacing.add_stim(n_beats=5, cycle_length40, stim_duration=0.1, stim_amplitude=2.)
prepacing.add_stim(n_beats=5, cycle_length30, stim_duration=0.1, stim_amplitude=2.)
prepacing.add_stim(n_beats=5, cycle_length25, stim_duration=0.1, stim_amplitude=2.)

cardiac_model = fw.AlievPanfilov()
cardiac_model.prepacing(prepacing)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = stim_time + 10
simulation.cardiac_model = cardiac_model
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

u = simulation.cardiac_model.u

connectivity = tissue.connectivity

plt.figure()
plt.imshow(u, cmap="coolwarm", origin="lower")

jj = np.arange(m)
for i in range(n-1):
    mask = connectivity[i, jj, 0] == 0
    x_ = [i + 0.5, i + 0.5]
    y_ = jj[mask]
    y_ = [y_ - 0.5, y_ + 0.5]
    plt.plot(y_, x_, color="black", linewidth=1)

ii = np.arange(n)
for j in range(m-1):
    mask = connectivity[ii, j, 1] == 0
    x_ = ii[mask]
    x_ = [x_ - 0.5, x_ + 0.5]
    y_ = [j + 0.5, j + 0.5]
    plt.plot(y_, x_, color="black", linewidth=1)
plt.show()
