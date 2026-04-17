from functools import lru_cache
import warnings
import numpy as np
import mlx.core as mx
from numba import njit, prange


def wrap_mlx_func(ops, start_with="calc_", exclude_funcs=["calc_where"]):
    """
    Identify all model-specific functions in the ops module that start
    with "calc_", inject mlx global functions into their namespaces

    Parameters
    ----------
    ops : module
        The operations module containing model-specific functions.
    start_with : str
        The prefix for identifying model-specific functions.
    exclude_funcs : list
        A list of function names to exclude from processing.
    
    Returns
    -------
    dict
        A dictionary mapping function names to their versions with injected globals.
    """
    glb_funcs = {"mx": mx,
                 "log": mx.log,
                 "exp": mx.exp,
                 "sqrt": mx.sqrt,
                 "abs": mx.abs,
                 "tanh": mx.tanh,
                 "calc_where": mx.where}
    
    model_funcs = {}
    for name in dir(ops):
        if name.startswith(start_with):
            func = getattr(ops, name)
            if name in exclude_funcs:
                continue

            if callable(func):
                func = _inject_globals(func, glb_funcs)
                model_funcs[name] = func

    return model_funcs, glb_funcs


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
                func = _inject_globals(func, glb_funcs)
                model_funcs[name] = njit(cache=True)(func)

    return model_funcs, glb_funcs


def _inject_globals(func, glb_funcs):
    g = getattr(func, "__globals__", None)

    if g is None:
        return func

    for k, v in glb_funcs.items():
        if k in g:
            func.__globals__[k] = v
    return func


@lru_cache(maxsize=64)
def build_func(func_name, func_code, glb_funcs, model_funcs):
    """Builds a function from its source code, injecting necessary globals.
    
    Parameters
    ----------
    func_name : str
        The name of the function to build.
    func_code : str
        The source code of the function to build.
    glb_funcs : dict
        A dictionary of global functions to inject into the function's namespace.
    model_funcs : dict
        A dictionary of model-specific functions to inject into the function's namespace.
    
    Returns
    -------
    function
        The built function with injected globals.
    """
    glb_all = {**dict(glb_funcs), **dict(model_funcs)}
    loc = {}
    exec(func_code, glb_all, loc)
    return loc[func_name]