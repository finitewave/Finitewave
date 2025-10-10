from numba import njit, prange


class ExplicitEuler:
    def __init__(self):
        pass

    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        a_matrix = mass_matrix + dt * stiffness_matrix
        return [a_matrix, mass_matrix]

    def solve(self, u_new, u, rhs, matrices, indexes):
        A, M = matrices
        return diffusion_kernel(u_new, u, rhs, indexes, A.indptr, A.indices,
                                A.data)


@njit(parallel=True, fastmath=True)
def diffusion_kernel(u_new, u, rhs, indexes, indptr, indices, data):
    n_rows = indptr.size - 1

    for i in prange(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        if start == end:
            continue
        acc = 0.0
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            acc += data[j] * u.flat[jj]

        ii = indexes[i]
        u_new.flat[ii] = acc + rhs.flat[ii]

    return u_new
