import re
import warnings
import textwrap
import ast
import inspect

from finitewave.core.model.kernel.vectorized_kernel_generator import VectorizedKernelGenerator


class LoopKernelGenerator(VectorizedKernelGenerator):
    """
    Base generator for model ionic kernels. 

    Attributes
    ----------
    kernel_func_name : str
        Name of the generated kernel function.
    common_args : list
        Names of common arguments passed to all kernels (e.g., rhs, indexes, dt, step)
    arrays : list
        Names passed as array arguments (e.g., u, gating variables, current fields)
    scalars : list
        Names passed as scalar arguments (e.g., parameters)
    model_args : list
        Names of all model-specific arguments (`arrays` + `scalars`): `step` function signature.
    state_vars : list
        Names of state variables that are updated in the step function (e.g., u, v).
    output_args : list
        Names of arguments that should be returned.
    observers : list
        List of dicts: {"name": <arg_name>, "expr": <code>}
        where expr is injected at the end of the per-cell loop body.
    body : str
        The body of the kernel function, executed for each cell/index.

    Notes
    -----
    - Observers are advanced instrumentation;
    - `expr` must be numba-friendly and race-safe.
    - Better no dynamic append / allocation in parallel kernels.
    """

    def __init__(self):
        VectorizedKernelGenerator.__init__(self)
    
    def generate_loop(self) -> str:
        """
        The header for the loop that iterates over the indexes.
        """
        raise NotImplementedError

    def _update_indexing(self, name, arrays):
        """
        Updates the indexing in the body of the kernel function to use the loop index.
        """
        raise NotImplementedError

    def _assign_indexing(self, name, arrays):
        """
        Assigns the indexing in the body of the kernel function to use the loop index.
        """
        raise NotImplementedError
