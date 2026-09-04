from functools import lru_cache


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


def inject_globals(func, glb_funcs):
    g = getattr(func, "__globals__", None)

    if g is None:
        return func

    for k, v in glb_funcs.items():
        if k in g:
            func.__globals__[k] = v
    return func
