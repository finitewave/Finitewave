from functools import lru_cache
from numba import njit, prange
from finitewave.core.model.kernel_generator import KernelGenerator


@lru_cache(maxsize=64)
def _build_cached(name, src, glb_key):
    loc = {}
    glb = { # dict of injected globals (calc_*, etc.)
        "njit": njit,
        "prange": prange,
        **dict(glb_key),        
    }
    exec(src, glb, loc)

    return loc[name]

def build_kernel(kernel_name, kernel_str, step_func_name, step_func_body, model_func):
    """
    Builds a Numba JIT-compiled kernel function from the provided kernel string and step function string.

    Parameters
    ----------
    kernel_str : str
        The source code of the kernel function as a string.
    step_func_name : str
        The name of the step function.
    step_func_body : str
        The body of the step function.
    model_func : dict
        A dictionary of model-specific functions (e.g., calc_dv, calc_rhs) that
    """

    # make globals hashable for caching
    sorted_model_func = tuple(sorted(model_func.items(), key=lambda kv: kv[0]))

    step_func = _build_cached(
        step_func_name,
        step_func_body,
        sorted_model_func,
    )
    
    # add step_func to globals for the kernel, so it can call it
    sorted_model_func += ((step_func_name, step_func),)

    kernel_func = _build_cached(
        kernel_name,
        kernel_str,
        sorted_model_func,
    )

    return kernel_func
