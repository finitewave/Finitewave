from functools import lru_cache
from importlib.metadata import entry_points
from numba import njit

REQS = ("get_variables", "get_parameters")


@lru_cache(maxsize=1)
def discover() -> dict:
    eps = entry_points()
    group = "finitewave.models"
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:
        selected = eps.get(group, [])
    return {ep.name: ep for ep in selected}


def load_ops(model_id: str):
    eps = discover()
    if model_id not in eps:
        raise KeyError(f"Model '{model_id}' not found via entry point group 'finitewave.models'.")
    mod = eps[model_id].load()   # ops package
    ops = getattr(mod, "ops", mod)
    for name in REQS:
        if not hasattr(ops, name):
            raise ValueError(f"Model '{model_id}' missing '{name}' in ops.")
    return ops


def wrap_calc(ops):
    """
    Here we identify all functions in the ops module that start with "calc_" or "rhs_", and we apply Numba's JIT compilation to them. 
    """
    py_funcs = {}
    for name in dir(ops):
        if name.startswith(("calc_", "rhs_")):
            fn = getattr(ops, name)
            if callable(fn):
                py_funcs[name] = fn

    jitted = {name: njit(cache=True)(fn) for name, fn in py_funcs.items()}

    for name, fn in py_funcs.items():
        g = getattr(fn, "__globals__", None)
        if not g:
            continue
        for dep_name, dep_jit in jitted.items():
            if dep_name in g:
                g[dep_name] = dep_jit

    jitted2 = {name: njit(cache=True)(fn) for name, fn in py_funcs.items()}

    return jitted2


@lru_cache(maxsize=64)
def get_ops_and_jit(model_id: str):
    try:
        ops = load_ops(model_id)
    except KeyError as e:
        raise ImportError(
            f"Finitewave model '{model_id}' not found.\n"
            f"Make sure the corresponding model package is installed "
            f"and exposes entry points under 'finitewave.models'."
        ) from e

    return ops, wrap_calc(ops)