import numpy as np
import pyvista as pv

import finitewave as fw


# Equal spacing in all directions makes every hexahedral element a cube.
nx, ny, nz = 100, 50, 10
x_range = (0, 50)
y_range = (0, 25)
z_range = (0, 5)

coords, elems = fw.build_hexahedral_slab(
    nx, ny, nz, x_range, y_range, z_range
)

tissue = fw.CardiacTissueElements(
    coords,
    elems,
    elem_type=fw.ElementType.HEXAHEDRON,
)

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(
    fw.StimVoltageCoord(
        0, 1,
        x_range[0], x_range[0] + 1,
        y_range[0], y_range[1],
        z_range[0], z_range[1],
    )
)

simulation = fw.CardiacSimulation(dt=0.01, t_max=10, backend="mlx")
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.FentonKarma()
simulation.stim_sequence = stim_sequence
simulation.time_integration = fw.BackwardEulerTimeIntegration(
    atol=1e-6,
    maxiter=100,
)
simulation.run()

# VTK uses the same node ordering as LinearHexahedralElement.
cells = np.column_stack([
    np.full(elems.shape[0], 8, dtype=np.int64),
    elems,
]).ravel()
cell_types = np.full(
    elems.shape[0],
    pv.CellType.HEXAHEDRON,
    dtype=np.uint8,
)
grid = pv.UnstructuredGrid(cells, cell_types, coords)
grid["u"] = simulation.cardiac_model.output("u")

plotter = pv.Plotter()
plotter.add_mesh(
    grid.extract_surface(),
    scalars="u",
    cmap="RdBu_r",
    show_edges=False,
)
plotter.show()
