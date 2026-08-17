import warnings


class Backend:
    def __init__(self):
        import numpy as np
        self.name = "numpy"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True
        self.gpu_support = False
        self.linalg_operator = None
        self.linalg_method = None

    def config(self, device=None, float_dtype=None, num_of_threads=None):
        """
        Configures the backend with specific settings. 
        This method should be overridden by subclasses to implement backend-specific configuration options.
        """
        pass

    def device_info(self):
        """
        Returns information about the computational device being used.
        """
        pass

    def wrap_array(self, arr):
        """
        Wraps an array into the backend's array type if it is not already.
        """
        # if hasattr(arr, "__array_namespace__") and arr.size > 1: - do lists have this?
        #     return self.lib.array(arr, dtype=self.float_dtype)
        if isinstance(arr, (int, float)):
            return arr
        return self.lib.array(arr, dtype=self.float_dtype)
    
    def wrap_indexes(self, arr):
        """
        Wraps an array of indexes into the backend's array type if it is not already.
        """
        return self.lib.array(arr, dtype=self.int_dtype)

    def wrap_sparse(self, csr_matrix, indexes=None, local_indexing=False):
        """
        Converts a sparse matrix in CSR format to the backend's compatible format.
        """
        pass

    def sync(self, arr):
        """
        Synchronizes the array across different backends if necessary.
        For backends that do not require synchronization, this method does nothing.

        Parameters
        ----------
        arr : array-like
            The array(s) to synchronize.
        """
        pass
    
    def select_values(self, arr, inds):
        """
        Selects values from an array based on the provided indexes.
        """
        return arr.flat[inds]
    
    def set_values(self, arr, inds, values):
        """
        Sets values in an array at the specified indexes.
        """
        arr[inds] = values
        return arr
    
    def set_flat_values(self, arr, inds, values):
        """
        Sets values in a flattened array at the specified indexes.
        """
        inds = self.lib.atleast_1d(inds)
        arr.flat[inds] = values
        return arr

    def add_flat_values(self, arr, inds, values):
        """
        Adds values to a flattened array at the specified indexes.
        """
        arr.flat[inds] += values
        return arr

    def copy(self, arr):
        """
        Creates a copy of the array.
        """
        return arr.copy()
