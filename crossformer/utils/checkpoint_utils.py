# crossformer/utils/checkpoint_utils.py

from typing import Mapping
from flax.core import FrozenDict


def canonicalize_restored_params(restored: Mapping):
    """
    Canonicalize parameters restored via the new Flax / Orbax API.

    Supported restore formats:

    1) TrainState-style restore (primary):
       {
         "model": {
           "params": <PyTree>
         },
         ...
       }

    2) Bare params restore (secondary / forward-compatible):
       {
         "params": <PyTree>
       }

    Any other format raises an explicit error.
    """
    if not isinstance(restored, Mapping):
        raise TypeError(
            "Restored object must be a mapping (dict-like), "
            f"got {type(restored)}"
        )

    # Primary: TrainState-style restore
    model = restored.get("model", None)
    if isinstance(model, Mapping) and "params" in model:
        params = model["params"]
        _validate_param_tree(params)
        return params

    # Secondary: bare params
    if "params" in restored:
        params = restored["params"]
        _validate_param_tree(params)
        return params

    raise KeyError(
        "Unsupported checkpoint restore format. Expected either "
        "TrainState-style restore with 'model.params' or a bare "
        "'params' PyTree."
    )


def _validate_param_tree(params):
    """
    Lightweight validation to ensure the extracted object
    looks like a Flax parameter PyTree.
    """
    if not isinstance(params, (dict, FrozenDict)):
        raise TypeError(
            "Extracted params must be a dict or FrozenDict, "
            f"got {type(params)}"
        )
