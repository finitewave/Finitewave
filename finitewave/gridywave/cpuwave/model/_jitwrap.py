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

# import jax
# from functools import partial
# import inspect


# def wrap_calc(ops):
#     jitted = {}

#     # get list of functions to wrap
#     fns = getattr(ops, "__all__", [])
#     fns = [fn for fn in fns if fn.startswith("calc_")]

#     for name in fns:
#         fn = getattr(ops, name)
#         if callable(fn):
#             # inspect signature
#             sig = inspect.signature(fn)
#             kwargs = [
#                 k for k, v in sig.parameters.items()
#                 if v.default is not inspect.Parameter.empty
#             ]

#             # create jitted version
#             if kwargs:
#                 jitted_fn = partial(jax.jit, static_argnames=kwargs)(fn)
#             else:
#                 jitted_fn = jax.jit(fn)

#             # update module and store in dict
#             ops.__dict__[name] = jitted_fn
#             jitted[name] = jitted_fn

#     return jitted
