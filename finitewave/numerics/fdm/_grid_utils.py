import numpy as np
from numba import njit, prange


def nonzero_weights(mesh, ijk, ijk_list, w_list, index_map=None, direction=1):
    """Collect sparse entries with absolute weights at least 1e-12.

    Coordinates carrying retained weights must be valid for index_map.
    This helper does not check bounds or activity.

    Parameters
    ----------
    mesh : numpy.ndarray
        Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
        All mesh > 0 nodes are included in compact C-order matrix indexing.
    ijk : numpy.ndarray, shape (ndim, N_points)
        Full-grid coordinates of center nodes.
    ijk_list : list of numpy.ndarray
        Neighbor coordinate arrays, each of shape (ndim, N_points).
    w_list : list of numpy.ndarray
        Corresponding weight arrays, each of shape (N_points,).
    index_map : numpy.ndarray, optional
        Map full-grid coordinates to matrix indices. If None, construct the
        compact C-order map for mesh > 0. Default is None.
    direction : int, optional
        Multiplier for emitted weights, normally 1 or -1. Default is 1.

    Returns
    -------
    rows : numpy.ndarray
        Compact matrix row indices.
    cols : numpy.ndarray
        Compact matrix column indices.
    weights : numpy.ndarray
        Corresponding matrix coefficients.
    """
    if index_map is None:
        index_map = - np.ones_like(mesh, dtype=np.int64)
        index_map[mesh > 0] = np.arange(np.count_nonzero(mesh > 0))

    rows, cols, weights = nonzero_weight_numba(mesh, ijk, ijk_list, w_list, index_map, direction)
    return rows, cols, weights


def build_neighbor(ijk, shift, axis):
    """Copy coordinates and shift them along one axis.

    Parameters
    ----------
    ijk : numpy.ndarray, shape (ndim, N_points)
        Full-grid coordinates of center nodes.
    shift : int
        Signed number of grid cells to shift.
    axis : int
        Grid axis.

    Returns
    -------
    neighbor : numpy.ndarray, shape (ndim, N_points)
        Shifted coordinates; bounds are not checked.
    """

    ijk = ijk.copy()
    ijk[axis] += shift
    return ijk


def is_valid_index(index, mesh):
    """Check bounds and require mesh == 1 for each coordinate.

    Parameters
    ----------
    index : numpy.ndarray, shape (ndim, N_points)
        Full-grid coordinates to validate.
    mesh : numpy.ndarray
        Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
        All mesh > 0 nodes are included in compact C-order matrix indexing.

    Returns
    -------
    valid : numpy.ndarray, shape (N_points,)
        Boolean activity mask; fibrotic and empty nodes are invalid.
    """
    valid = is_valid_indexes_numba(index, mesh)
    return valid


def reindex_matrix(mesh, rows, cols, indexes):
    """Map full-grid matrix indices into the supplied node ordering.

    Parameters
    ----------
    mesh : numpy.ndarray
        Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
        All mesh > 0 nodes are included in compact C-order matrix indexing.
    rows : numpy.ndarray
        Flat full-grid row indices; every index must occur in indexes.
    cols : numpy.ndarray
        Flat full-grid column indices; every index must occur in indexes.
    indexes : numpy.ndarray
        Flat full-grid indices of center nodes. For operator builders,
        None selects mesh == 1; these are not compact matrix indices.

    Returns
    -------
    rows : numpy.ndarray
        Rows mapped to positions in indexes.
    cols : numpy.ndarray
        Columns mapped to positions in indexes.
    """
    c_indexes = np.zeros(mesh.size, dtype=np.int64)
    c_indexes[indexes] = np.arange(len(indexes))
    rows = c_indexes[rows]
    cols = c_indexes[cols]
    return rows, cols


@njit
def _is_valid_index_numba(multi_index, limits, mesh):
    for axis in range(mesh.ndim):
        coord = multi_index[axis]
        limit = limits[axis]
        if coord < 0 or coord >= limit:
            return False

    flat_index = ravel_multi_index_numba(multi_index, mesh.shape)
    return mesh.flat[flat_index] == 1


@njit(parallel=True)
def is_valid_indexes_numba(multi_indexes, mesh):
    n_points = multi_indexes.shape[1]
    limits = np.array(mesh.shape)
    mask = np.zeros(n_points, dtype=np.bool_)
    for i in prange(n_points):
        index = multi_indexes[:, i]
        mask[i] = _is_valid_index_numba(index, limits, mesh)
    return mask


@njit
def ravel_multi_index_numba(multi_index, shape):
    flat_index = 0
    for axis in range(len(shape)):
        flat_index = flat_index * shape[axis] + multi_index[axis]
    return flat_index


@njit(parallel=False)
def nonzero_weight_numba(mesh, ijk, ijk_list, w_list, index_map, direction=1):
    n_weights = len(w_list)
    n_points = ijk.shape[1]
    rows = np.empty(n_weights * n_points, dtype=np.int64)
    cols = np.empty(n_weights * n_points, dtype=np.int64)
    weights = np.empty(n_weights * n_points, dtype=w_list[0].dtype)

    count = 0
    for i in range(n_points):
        for j in range(n_weights):
            w = w_list[j][i]
            if abs(w) < 1e-12:
                continue

            ind = count
            flat_index = ravel_multi_index_numba(ijk[:, i], mesh.shape)
            neighbor_flat_index = ravel_multi_index_numba(ijk_list[j][:, i], mesh.shape)
            rows[ind] = index_map.flat[flat_index]
            cols[ind] = index_map.flat[neighbor_flat_index]
            weights[ind] = direction * w
            count += 1

    return rows[:count], cols[:count], weights[:count]
