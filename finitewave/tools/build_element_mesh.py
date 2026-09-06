import numpy as np
import pyvista as pv


def build_triangulated_plane(n, m, x_range, y_range):
    """
    Build a 

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


def build_quadrilateral_plane(n, m, x_range, y_range):
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


def build_hexahedral_slab(nx, ny, nz, x_range, y_range, z_range):
    """Build a structured slab of eight-node hexahedral elements.

    The elements are cubes when the spacing is the same along all three axes;
    otherwise, they are rectangular hexahedra. Nodes on each element are
    ordered around the lower face first and then around the upper face, as
    expected by :class:`LinearHexahedralElement`.

    Parameters
    ----------
    nx, ny, nz : int
        Number of elements along the x, y, and z axes.
    x_range, y_range, z_range : tuple
        Minimum and maximum coordinates along each axis.

    Returns
    -------
    coords : (N, 3) ndarray
        Node coordinates.
    elems : (M, 8) ndarray
        Hexahedral element connectivity.
    """
    x = np.linspace(x_range[0], x_range[1], nx + 1)
    y = np.linspace(y_range[0], y_range[1], ny + 1)
    z = np.linspace(z_range[0], z_range[1], nz + 1)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([
        x_grid.ravel(),
        y_grid.ravel(),
        z_grid.ravel(),
    ])

    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    i, j, k = i.ravel(), j.ravel(), k.ravel()

    def node_index(i_coord, j_coord, k_coord):
        return i_coord * (ny + 1) * (nz + 1) + j_coord * (nz + 1) + k_coord

    n000 = node_index(i, j, k)
    n100 = node_index(i + 1, j, k)
    n110 = node_index(i + 1, j + 1, k)
    n010 = node_index(i, j + 1, k)
    n001 = node_index(i, j, k + 1)
    n101 = node_index(i + 1, j, k + 1)
    n111 = node_index(i + 1, j + 1, k + 1)
    n011 = node_index(i, j + 1, k + 1)

    elems = np.column_stack([
        n000, n100, n110, n010,
        n001, n101, n111, n011,
    ])
    return coords, elems


def build_tetrahedral_slab(nx, ny, nz, x_range, y_range, z_range):
    """
    Build a slab of tetrahedral elements by subdividing a box into cubes and 
    then splitting each cube into 5 tetrahedra.

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


def build_triangulated_sphere(radius, nsub=7):
    mesh = pv.Icosphere(nsub=nsub, radius=radius)

    coords = mesh.points
    elems = mesh.faces.reshape((-1, 4))[:, 1:4]

    # find coordinates in spherecial coordinates
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    phi = np.arctan2(coords[:, 2], np.sqrt(coords[:, 0]**2 + coords[:, 1]**2))

    # make holes at (1) z = -radius, (2) theta = 0, phi = pi/4 (3) theta = pi/3, phi = pi/4
    center1 = np.array([0, 0, -radius])
    center2 = coords[(theta < 0.1) & (theta > -0.1) & 
                     (phi < np.pi/4 + 0.1) & 
                     (phi > np.pi/4 - 0.1)][0]
    center3 = coords[(theta < np.pi + 0.1) & 
                     (theta > np.pi - 0.1) & 
                     (phi < np.pi/4 + 0.1) & 
                     (phi > np.pi/4 - 0.1)][0]

    holes_center = [center1, center2, center3]
    holes_radius = [radius / 1.5, radius / 2, radius / 2]

    mask = np.ones(coords.shape[0], dtype=bool)
    for center, r in zip(holes_center, holes_radius):
        dist = np.linalg.norm(coords - center, axis=1)
        mask &= dist > r
        
    old_inds = - np.ones(coords.shape[0], dtype=int)
    coords = coords[mask, :]
    elems = elems[np.all(mask[elems], axis=1), :]

    old_inds[mask] = np.arange(coords.shape[0])
    elems = old_inds[elems]
    return coords, elems
