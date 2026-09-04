from finitewave.core.model.kernel.tools.wrapping_functions import inject_globals


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
    import mlx.core as mx
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
                func = inject_globals(func, glb_funcs)
                model_funcs[name] = func

    return model_funcs, glb_funcs