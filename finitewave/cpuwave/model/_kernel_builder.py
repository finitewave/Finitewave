from functools import lru_cache
from numba import njit, prange
import math
import numpy as np


@lru_cache(maxsize=64)
def _build_cached(name, src, glb_key):
    loc = {}
    glb = { # dict of injected globals (calc_*, etc.)
        "njit": njit,
        "prange": prange,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "np": np,
        **dict(glb_key),        
    }
    exec(src, glb, loc)

    return loc[name]
