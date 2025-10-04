# finitewave/models/_registry.py
from importlib.metadata import entry_points

REQS = ("get_variables", "get_parameters")


def discover() -> dict:
    eps = entry_points(group="finitewave.models")
    return {ep.name: ep for ep in eps}


def load_ops(model_id: str):
    eps = discover()
    if model_id not in eps:
        raise KeyError(f"Model '{model_id}' not found via entry point 'finitewave.models'.")
    mod = eps[model_id].load()   # ops package
    ops = getattr(mod, "ops", mod)
    for name in REQS:
        if not hasattr(ops, name):
            raise ValueError(f"Model '{model_id}' missing '{name}' in ops.")
    return ops