class BackendLinalg:
    def __init__(self):
        pass

    def wrap_matrix(self, csr_matrix, dtype, indexes=None):
        """Converts a sparse matrix in CSR format to the appropriate format for the backend.

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CSR format.
        dtype : np.dtype
            The data type for the backend-specific format arrays.
        indexes : 1D array of int, optional
            Array of indexes where the solution is defined.

        Returns
        -------
        backend-specific representation of the matrix.
        """
        raise NotImplementedError("wrap_matrix must be implemented by subclasses.")
    
    def evaluate(self, u, step):
        """Evaluates the solution vector, performing any necessary backend-specific operations.

        Parameters
        ----------
        u : backend-specific array
            The solution vector to evaluate.
        step : int
            The current time step of the simulation.

        Returns
        -------
        Evaluated solution vector.
        """
        raise NotImplementedError("evaluate must be implemented by subclasses.")
    
    def explicit_step(self, indices, data, u_old, rhs, dt, indexes, u):
        """Performs the explicit step for the Forward Euler method.

        Parameters
        ----------
        indices : backend-specific array
            The column indices of the non-zero elements in the system matrix.
        data : backend-specific array
            The non-zero values of the system matrix.
        u_old : backend-specific array
            The solution vector at the previous time step.
        rhs : backend-specific array
            The right-hand side vector from the cardiac model.
        dt : float
            Time step for the simulation.
        indexes : 1D array of int
            Array of indexes where the solution is defined.
        u : backend-specific array
            The solution vector to update with the new values.

        Returns
        -------
        Updated solution vector after the explicit step.
        """        
        raise NotImplementedError("explicit_step must be implemented by subclasses.")
    
    def axmy(self, a, x, y, indexes, out):
        """Performs the operation out = a * x + y for the specified indexes.

        Parameters
        ----------
        a : float
            Scalar multiplier for x.
        x : backend-specific array
            The input vector to be scaled and added.
        y : backend-specific array
            The input vector to be added.
        indexes : 1D array of int
            Array of indexes where the operation is performed.
        out : backend-specific array
            The output vector to store the result.

        Returns
        -------
        Updated output vector after performing the axmy operation.
        """
        raise NotImplementedError("axmy must be implemented by subclasses.")
    
    def axpy(self, a, x, y, indexes, out):
        """Performs the operation out = a * x + y for the specified indexes.

        Parameters
        ----------
        a : float
            Scalar multiplier for x.
        x : backend-specific array
            The input vector to be scaled and added.
        y : backend-specific array
            The input vector to be added.
        indexes : 1D array of int
            Array of indexes where the operation is performed.
        out : backend-specific array
            The output vector to store the result.

        Returns
        -------
        Updated output vector after performing the axpy operation.
        """
        raise NotImplementedError("axpy must be implemented by subclasses.")
    
    def matvec(self, indices, data, x, indexes, out):
        """Performs the matrix-vector multiplication out = A @ x where A is represented in a backend-specific format.

        Parameters
        ----------
        indices : backend-specific array
            The column indices of the non-zero elements in the system matrix.
        data : backend-specific array
            The non-zero values of the system matrix.
        x : backend-specific array
            The input vector to be multiplied by the matrix.
        indexes : 1D array of int
            Array of indexes where the operation is performed.
        out : backend-specific array
            The output vector to store the result.

        Returns
        -------
        Updated output vector after performing the matrix-vector multiplication.
        """
        raise NotImplementedError("matvec must be implemented by subclasses.")
    
    def cg_solve(self, indices, data, b, x0, indexes, atol=1e-8, maxiter=1):
        """Solves the linear system Ax = b using the Conjugate Gradient method, where A is represented in a backend-specific format.

        Parameters
        ----------
        indices : backend-specific array
            The column indices of the non-zero elements in the system matrix.
        data : backend-specific array
            The non-zero values of the system matrix.
        b : backend-specific array
            The right-hand side vector.
        x0 : backend-specific array
            The initial guess for the solution.
        indexes : 1D array of int
            Array of indexes where the solution is defined.
        atol : float, optional
            Absolute tolerance for convergence (default is 1e-8).
        maxiter : int, optional
            Maximum number of iterations (default is 1).

        Returns
        -------
        Tuple containing:
            - Updated solution vector after solving Ax = b.
            - Number of iterations taken to converge.
        """
        raise NotImplementedError("cg_solve must be implemented by subclasses.")
    
    