
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import finitewave as fw


def load_mesh(path):
    coords = np.genfromtxt(path.joinpath("mesh.pts"), skip_header=1,
                           usecols=[0, 1, 2])
    coords /= 1000

    # print(coords.min(axis=0), coords.max(axis=0))
    elems = np.genfromtxt(path.joinpath("mesh.elem"), skip_header=1,
                          usecols=[1, 2, 3], dtype=int)
    return coords, elems


# create a tissue of size 400x400 with cardiomycytes:
path = Path(__file__).parents[2].joinpath("data", "atrial_mesh")
coords, elems = load_mesh(path)

stim_coord = coords[len(coords) // 2]
stim_size = 1
stim = fw.StimVoltageElectrodesFEM(0, 1, stim_coord, stim_size)

tissue = fw.CardiacTissueFEM(coords, elems)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim)

# create model object and set up parameters:
model = fw.CourtemancheFEM()
model.dt = 0.01
model.t_max = 40
# add the tissue and the stim parameters to the model object:
model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence

# run the model:
model.run(num_of_threads=None)

u = model.u

# show the potential map at the end of calculations:
faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
mesh.point_data["values"] = u
# plot
mesh.plot(cmap="RdBu_r", show_edges=False)
