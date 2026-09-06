

class Backend:
    def __init__(self):
        import numpy as np
        self.name = "numpy"
        self.lib = np
        self._float_dtype = np.float64
        self._int_dtype = np.int64
        self.sparse_support = True
        self.gpu_support = False
        self.linalg = None
        self.model_generator = None
        self.sync_step = 1

    @property
    def float_dtype(self):
        return self._float_dtype

    @float_dtype.setter
    def float_dtype(self, value):
        self._float_dtype = value

    @property
    def int_dtype(self):
        return self._int_dtype

    @int_dtype.setter
    def int_dtype(self, value):
        self._int_dtype = value

    def config(self, **kwargs):
        """
        Configures the backend with specific settings. 
        This method should be overridden by subclasses to implement backend-specific configuration options.
        """
        pass

    def sync(self, *args):
        """
        Synchronizes the backend if necessary. 
        This method should be overridden by subclasses to implement backend-specific synchronization.
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
        # If the array is boolean wrap is as boolean
        wraped_arr = self.lib.asarray(arr, dtype=self.float_dtype)
        if wraped_arr.size == 1:
            return arr
        return self.lib.asarray(arr, dtype=self.float_dtype)
    
    def wrap_indexes(self, arr):
        """
        Wraps an array of indexes into the backend's array type if it is not already.
        """
        return self.lib.asarray(arr, dtype=self.int_dtype)

    def wrap_mask(self, arr):
        """
        Wraps an array of boolean values into the backend's array type if it is not already.
        """
        return self.lib.asarray(arr, dtype=self.lib.bool_)

    def wrap_sparse(self, csr_matrix, indexes=None, row_reduced=False, local_indexing=False):
        """
        Converts a sparse matrix in CSR format to the backend's compatible format.
        """
        raise NotImplementedError("wrap_sparse must be implemented.")

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
