import numpy as np
import pyvista as pv
import finitewave as fw


# create a tissue of size 50x50 with 200x200 points:
n = 200
size = 50
coords, elems = fw.build_quadrilateral_mesh(n, n, (0, size), (0, size))

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type='Quadrilateral')
tissue.mesh_elems += (np.random.rand(*tissue.mesh_elems.shape) < 0.2).astype(int)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()

stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, size, 0, 1))
stim_sequence.add_stim(fw.StimVoltageCoord(45, 1, 0, size//2, 0, size))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(backend="jax")
simulation.dt = 0.01
simulation.t_max = 100
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
# set up the solver, default is Crank-Nicolson:
# ! Forward Euler is conditionally stable for quadrilateral meshes
# simulation.solver = fw.ForwardEulerSolver()

# run the model:
simulation.run()

# get the resulting potential at the element centers:
u = simulation.cardiac_model.u
u_elems = np.zeros(tissue.elems.shape[0]) * np.nan
u_elems[tissue.myo_elems_indexes] = u[tissue.myo_elements].mean(axis=1)

grid = fw.PyVistaSurfaceGrid(coords, elems)
grid["u"] = u_elems

pl = pv.Plotter()
pl.add_mesh(grid, cmap="RdBu_r", show_edges=False)
pl.camera_position = 'xy'
pl.show()
