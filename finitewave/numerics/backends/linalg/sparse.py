"""Conversions between sparse-matrix formats used by numerical backends."""

import numpy as np



def csr_to_ellpack(csr_matrix):
    """Convert a SciPy CSR matrix to Finitewave's ELLPACK representation.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        The input CSR matrix.

    Returns
    -------
    ellpack_indices : np.ndarray
        Column indexes with shape ``(n_rows, max_entries_per_row)``.
    ellpack_data : np.ndarray
        Matrix values with the same shape as ``ellpack_indices``. Unused
        entries are padded with zeros.

    Notes
    -----
    Padded column indexes default to the row index. Their corresponding values
    are zero, so they do not affect matrix-vector products.
    """
    indptr = csr_matrix.indptr
    indices = csr_matrix.indices
    data = csr_matrix.data

    rows_len = np.diff(indptr)
    n_cols = np.max(rows_len)
    n_rows = csr_matrix.shape[0]

    ellpack_indices = np.repeat(np.arange(n_rows), n_cols).reshape(n_rows, n_cols)
    ellpack_data = np.zeros((n_rows, n_cols), dtype=data.dtype)

    inds = np.repeat([np.arange(n_cols)], n_rows, axis=0)
    mask = inds < rows_len[:, None]
    ellpack_indices[mask] = indices
    ellpack_data[mask] = data

    return ellpack_indices, ellpack_data
