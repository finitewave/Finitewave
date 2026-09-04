import numpy as np

from finitewave.core.model.kernel.tools.wrapping_functions import inject_globals


def wrap_numba_func(ops, start_with="calc_", exclude_funcs=["calc_where"]):
    """
    Identify all model-specific functions in the ops module that start
    with "calc_", inject numba compatible global functions into their namespaces,
    and apply Numba's JIT compilation to them.

    Parameters
    ----------
    ops : module
        The operations module containing model-specific functions.
    start_with : str
        The prefix for identifying model-specific functions.
    exclude_funcs : list
        A list of function names to exclude from JIT compilation.

    Returns
    -------
    dict
        A dictionary mapping function names to their JIT-compiled versions.
    """

    from numba import njit, prange

    def calc_where(cond, x, y):
        return x if cond else y
    
    glb_funcs = {"njit": njit,
                 "prange": prange,
                 "np": np,
                 "log": np.log,
                 "exp": np.exp,
                 "sqrt": np.sqrt}
    
    glb_funcs["calc_where"] = njit(cache=True)(calc_where)

    model_funcs = {}
    for name in dir(ops):
        if name.startswith(start_with):
            if name in exclude_funcs:
                continue

            func = getattr(ops, name)
            if callable(func):
                func = inject_globals(func, glb_funcs)
                model_funcs[name] = njit(cache=True)(func)

    return model_funcs, glb_funcs