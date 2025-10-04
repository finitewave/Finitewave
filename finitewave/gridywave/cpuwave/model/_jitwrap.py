# finitewave/models/_jitwrap.py
from numba import njit


def wrap_calc(ops):
    jitted = {}
    fns = getattr(ops, "__all__", [])
    fns = [fn for fn in fns if fn.startswith("calc_")]
    for name in fns:
        fn = getattr(ops, name)
        if callable(fn):
            ops.__dict__[name] = njit(cache=True)(fn)
            jitted[name] = njit(cache=True)(fn)
    return jitted
