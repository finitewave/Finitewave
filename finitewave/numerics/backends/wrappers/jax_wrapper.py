from finitewave.core.model.kernel.tools.wrapping_functions import inject_globals


def wrap_jax_func(ops, start_with="calc_", exclude_funcs=["calc_where"]):
    """
    Identify all model-specific functions in the ops module that start
    with "calc_", inject jax global functions into their namespaces,

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
    import jax
    import jax.numpy as jnp

    glb_funcs = {"jax": jax,
                 "log": jnp.log,
                 "exp": jnp.exp,
                 "sqrt": jnp.sqrt,
                 "abs": jnp.abs,
                 "tanh": jnp.tanh,
                 "calc_where": jnp.where}
    
    model_funcs = {}
    for name in dir(ops):
        if name.startswith(start_with):
            func = getattr(ops, name)
            if name in exclude_funcs:
                continue

            if callable(func):
                func = inject_globals(func, glb_funcs)
                model_funcs[name] = func

    return model_funcs, glb_funcs