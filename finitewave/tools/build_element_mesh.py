import numpy as np


def build_triangulated_mesh(n, m, x_range, y_range):
    """
    Build a 2D triangular mesh.

    Parameters:
    ----------
    n: int
        Number of divisions along the x-axis.
    m: int
        Number of divisions along the y-axis.
    x_range: tuple
        Range of x-coordinates (min, max).
    y_range: tuple
        Range of y-coordinates (min, max).

    Returns:
    -------
    coords: (N_nodes, 2) ndarray
        Coordinates of the mesh nodes.
    elems: (N_elems, 3) ndarray
        Element connectivity (node indices for each triangle).
    """
    x = np.linspace(x_range[0], x_range[1], n + 1)
    y = np.linspace(y_range[0], y_range[1], m + 1)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.ravel(), yv.ravel()])

    elems = np.empty((m * n * 2, 3), dtype=np.int64)

    for i in range(m):
        for j in range(n):
            n0 = i * (n + 1) + j
            n1 = n0 + 1
            n2 = n0 + (n + 1)
            n3 = n2 + 1
            elems[2 * (i * n + j)] = [n0, n1, n3]
            elems[2 * (i * n + j) + 1] = [n0, n3, n2]
    elems = np.array(elems)

    return coords, elems


def build_quadrilateral_mesh(n, m, x_range, y_range):
    """ Build a 2D quadrilateral mesh.

    Parameters:
    ----------
    n: int
        Number of divisions along the x-axis.
    m: int
        Number of divisions along the y-axis.
    x_range: tuple
        Range of x-coordinates (min, max).
    y_range: tuple
        Range of y-coordinates (min, max).

    Returns:
    -------
    coords: (N_nodes, 2) ndarray
        Coordinates of the mesh nodes.
    elems: (N_elems, 4) ndarray
        Element connectivity (node indices for each quadrilateral).
    """
    x = np.linspace(x_range[0], x_range[1], n+1)
    y = np.linspace(y_range[0], y_range[1], m+1)
    xv, yv = np.meshgrid(x, y)
    coords = np.vstack([xv.ravel(), yv.ravel()]).T

    elems = np.empty((m * n, 4), dtype=np.int64)

    for j in range(m):
        for i in range(n):
            v0 = j * (n + 1) + i
            v1 = v0 + 1
            v2 = v0 + n + 1
            v3 = v2 + 1
            elems[j * n + i] = [v0, v1, v3, v2]

    return coords, elems


def build_tetrahedral_mesh(nx, ny, nz, x_range, y_range, z_range):
    """
    Build a 3D tetrahedral mesh.

    Parameters
    ----------
    nx, ny, nz : int
        Number of subdivisions in each direction.
    x_range, y_range, z_range : tuple
        Size of the box in each direction (min, max).

    Returns
    -------
    coords : (N, 3) array
        Node coordinates.
    tets : (M, 4) array
        Tetrahedra connectivity.
    """
    # --- coordinates ---
    x = np.linspace(x_range[0], x_range[1], nx+1)
    y = np.linspace(y_range[0], y_range[1], ny+1)
    z = np.linspace(z_range[0], z_range[1], nz+1)
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
    n000 = idx(I,   J,   K)
    n100 = idx(I+1, J,   K)
    n010 = idx(I,   J+1, K)
    n110 = idx(I+1, J+1, K)
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
