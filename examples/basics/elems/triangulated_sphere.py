from pathlib import Path
import numpy as np
import pyvista as pv
import finitewave.elementalwave as fw
import matplotlib.pyplot as plt


# path = Path("/Users/arstanbek/Projects/fibrosis/ElementalWave/data")
path = Path("C:/Users/aooke/Projects/ElementalWave/data/start")

coords = np.genfromtxt(path.joinpath("mesh.pts"),
                       skip_header=1,
                       usecols=[0, 1, 2])
coords /= 1000

# print(coords.min(axis=0), coords.max(axis=0))
elems = np.genfromtxt(path.joinpath("mesh.elem"),
                      skip_header=1,
                      usecols=[1, 2, 3],
                      dtype=int)

# print(coords.shape, elems.shape)

# # triangle edge length
# edge_length = [np.linalg.norm(coords[elems[:, 0]] - coords[elems[:, 1]], axis=1),
#                np.linalg.norm(coords[elems[:, 1]] - coords[elems[:, 2]], axis=1),
#                np.linalg.norm(coords[elems[:, 2]] - coords[elems[:, 0]], axis=1)]

# edge_length = np.concatenate(edge_length)

# plt.hist(edge_length)
# plt.show()

# stim_coords = coords[np.random.choice(coords.shape[0], 1, replace=False)]

# # create a tissue of size 400x400 with cardiomycytes:

# tissue = fw.CardiacTissueFEM(coords, elems)
# # set up stimulation parameters:
# stim_sequence = fw.StimSequence()
# stim_sequence.add_stim(fw.StimVoltageElectrodesFEM(0, 1, stim_coords, 1))

# # create model object and set up parameters:
# aliev_panfilov = fw.LuoRudy91FEM()
# aliev_panfilov.dt = 0.01
# aliev_panfilov.t_max = 2
# # add the tissue and the stim parameters to the model object:
# aliev_panfilov.cardiac_tissue = tissue
# aliev_panfilov.stim_sequence = stim_sequence

# # run the model:
# aliev_panfilov.run(num_of_threads=1)

# u = aliev_panfilov.u

# # u = np.zeros(coords.shape[0])
# # u[coords[:, 1] <= -19.9] = 1
# # show the potential map at the end of calculations:
# faces = np.hstack([[3, *tri] for tri in elems])
# mesh = pv.PolyData(coords, faces)
# mesh.point_data["values"] = u
# # plot
# mesh.plot(cmap="RdBu_r", show_edges=False)
