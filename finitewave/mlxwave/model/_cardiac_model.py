import numpy as np
import mlx.core as mx
from warnings import warn

from finitewave.mlxwave.model.kernel.ionic_mlx_kernel import IonicMlxGenerator
from finitewave.cpuwave.model.kernel.prepacing_numba_kernel import PrepacingNumbaGenerator
from finitewave.cpuwave.model._cardiac_model import (
    CardiacModel as CardiacModelCPU
)


class CardiacModel(CardiacModelCPU):
    """
    Base class for cardiac grid models.

    Attributes
    ----------
    memory_save : bool
        Whether to save memory by only storing the state variables at the
        tissue indexes (``mesh > 0``).
    myo_indexes : np.ndarray
        Array of indices corresponding to the myocytes in the mesh.
        If `memory_saving` is True, the indexes correspond to `mesh.flat[tissue_indexes[myo_indexes]]`.
        Otherwise, they correspond to `mesh.flat[myo_indexes]`.
    tissue_indexes : np.ndarray
        Array of indices corresponding to the tissue points. For consistency, when `memory_save` is False,
        this will be an array of all indexes in the mesh.
    ionic_kernel_generator : KernelGenerator
        Object that generates the multithreaded `ionic_kernel` function for the model.
    prepacing_generator : KernelGenerator
        Object that generates the signle-cell `prepacing_kernel` function for the model.
    """

    def __init__(self):
        """
        Initializes the CardiacModel instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Whether to save memory by only storing the state variables at the
            tissue indexes (``mesh > 0``).
        """
        super().__init__(memory_save=True)
        self.myo_indexes = None
        self.tissue_indexes = None
        self.ionic_kernel_generator = IonicMlxGenerator()
        self.prepacing_generator = PrepacingNumbaGenerator()
        
    def compute_indexes(self, cardiac_tissue):
        """
        Computes the myocyte and tissue indexes based on the cardiac tissue mesh.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The cardiac tissue object.
        """
        super().compute_indexes(cardiac_tissue)
        self.myo_indexes = mx.array(self.myo_indexes, dtype=mx.int32)
        self.tissue_indexes = mx.array(self.tissue_indexes, dtype=mx.int32)

    def run(self):
        """
        Executes the ionic kernel for the current time step.

        Parameters
        ----------
        dt : float
            Time step size for the simulation.
        """
        if (self.simulation.step - 1) % self.step != 0:
            return
        
        res = self.ionic_kernel(self.simulation.dt, self.u, *self.ionic_kernel_args)

        self.rhs = res[0]

        # if mx.any(mx.isnan(self.rhs)) or mx.any(mx.isinf(self.rhs)):  # --- DEBUG ---
        #     print("NaN or Inf found in rhs")
        #     print(self.simulation.t)
        #     print(np.array(self.u)[np.where(np.isnan(self.rhs))])
        #     breakpoint()
        
        i = 0
        for name in self.state_vars:
            if name == "u":
                continue
            self.__dict__[name] = res[i+1]
            i += 1
        
        self.ionic_kernel_args[:len(res) - 1] = res[1:]

    def _allocate_arrays(self, simulation):
        """
        Allocates the model's state variable arrays based on the simulation's cardiac tissue mesh.
        
        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac tissue mesh.
        """
        shape = (len(simulation.cardiac_tissue.tissue_indexes), )
        
        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            array_val = init_val * np.ones(shape, dtype=np.float32)
            setattr(self, name, mx.array(array_val))

        # validate parameter fields shapes if they are arrays
        for name in self.default_parameters.keys():
            par = getattr(self, name)
            if isinstance(par, mx.array) and par.size > 1:
                if par.shape != shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape {shape}"
                    ) 
