from pathlib import Path
import numpy as np
import pyvista as pv
import finitewave as fw


path = Path("/Users/arstanbek/Projects/fibrosis/ElementalWave/data")

coords = np.genfromtxt(path.joinpath("mesh.pts"),
                       skip_header=1,
                       usecols=[0, 1, 2])
coords /= 1000

# print(coords.min(axis=0), coords.max(axis=0))
elems = np.genfromtxt(path.joinpath("mesh.elem"),
                      skip_header=1,
                      usecols=[1, 2, 3],
                      dtype=int)

# create a tissue of size 400x400 with cardiomycytes:

tissue = fw.CardiacTissueElem(coords, elems)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoordElem(0, 1,
                                               -20, 20,
                                               -20, -19.9,
                                               -20, 20))
stim_sequence.add_stim(fw.StimVoltageCoordElem(45, 1,
                                               -20, 0,
                                               -20, 20,
                                               -20, 20))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilovElems()
aliev_panfilov.dt = 0.01
aliev_panfilov.t_max = 200
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run(num_of_threads=1)

u = aliev_panfilov.u

# u = np.zeros(coords.shape[0])
# u[coords[:, 1] <= -19.9] = 1
# show the potential map at the end of calculations:
faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
mesh.point_data["values"] = u
# plot
mesh.plot(cmap="RdBu_r")
