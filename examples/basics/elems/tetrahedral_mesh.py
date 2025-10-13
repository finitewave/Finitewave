import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import finitewave as fw


def build_tet_mesh(nx, ny, nz, x_size, y_size, z_size):
    """
    Generate tetrahedral mesh of a box [0,L]x[0,H]x[0,W], fully vectorized.

    Parameters
    ----------
    L, H, W : float
        Dimensions of the box.
    nx, ny, nz : int
        Number of subdivisions in each direction.

    Returns
    -------
    coords : (N, 3) array
        Node coordinates.
    tets : (M, 4) array
        Tetrahedra connectivity.
    """
    # --- coordinates ---
    x = np.linspace(x_size[0], x_size[1], nx+1)
    y = np.linspace(y_size[0], y_size[1], ny+1)
    z = np.linspace(z_size[0], z_size[1], nz+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # --- cell indices ---
    I, J, K = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    I, J, K = I.ravel(), J.ravel(), K.ravel()

    # convert (i,j,k) to node index
    def idx(i, j, k):
        return i*(ny+1)*(nz+1) + j*(nz+1) + k

    # 8 corners of each cube
    n000 = idx(I,   J,   K  )
    n100 = idx(I+1, J,   K  )
    n010 = idx(I,   J+1, K  )
    n110 = idx(I+1, J+1, K  )
    n001 = idx(I,   J,   K+1)
    n101 = idx(I+1, J,   K+1)
    n011 = idx(I,   J+1, K+1)
    n111 = idx(I+1, J+1, K+1)

    # 5 tets per cube
    tets = np.vstack([
        np.column_stack([n000, n100, n010, n001]),
        np.column_stack([n100, n110, n010, n111]),
        np.column_stack([n100, n010, n001, n111]),
        np.column_stack([n010, n001, n011, n111]),
        np.column_stack([n100, n001, n101, n111]),
    ])

    return coords, tets


# create a tissue of size 400x400 with cardiomycytes:
nx = 100
ny = 100
nz = 20
size_x = (0, 50)
size_y = (0, 50)
size_z = (0, 10)

coords, elems = build_tet_mesh(nx, ny, nz, size_x, size_y, size_z)

print(coords.shape, elems.shape)

tissue = fw.CardiacTissueFEM(coords, elems)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                               0, size_x[1],
                                               0, 1,
                                               0, size_z[1]))
stim_sequence.add_stim(fw.StimVoltageCoord(45, 1,
                                               0, size_x[1]/2,
                                               0, size_y[1],
                                               0, size_z[1]))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilovFEM()
aliev_panfilov.dt = 0.01
aliev_panfilov.t_max = 200
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run(num_of_threads=1)

u = aliev_panfilov.u

# show the potential map at the end of calculations:
cells = np.hstack([np.full((elems.shape[0], 1), 4), elems]).ravel()
celltypes = np.full(elems.shape[0], pv.CellType.TETRA, dtype=np.uint8)

grid = pv.UnstructuredGrid(cells, celltypes, coords)
grid.point_data["u"] = u

# --- Visualization ---
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="RdBu_r")
plotter.show()
