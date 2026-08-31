import numpy as np



def csr_to_ellpack(csr_matrix):
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
    