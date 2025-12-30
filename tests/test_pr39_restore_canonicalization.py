import numpy as np
import pytest
from flax.core import FrozenDict
from jax.tree_util import tree_leaves


def canonicalize_restored_params(restored):
    """
    Canonical parameter extraction for new Flax / Orbax API.

    Supported formats:
    1) TrainState-style:
       { "model": { "params": <PyTree> }, ... }

    2) Bare params:
       { "params": <PyTree> }

    Any other format is an error.
    """
    if not isinstance(restored, dict):
        raise TypeError("Restored object must be a dict")

    # Primary: TrainState-style restore
    model = restored.get("model", None)
    if isinstance(model, dict) and "params" in model:
        return model["params"]

    # Secondary: bare params
    if "params" in restored:
        return restored["params"]

    raise KeyError(
        "Unsupported restore format: expected TrainState-style "
        "('model.params') or bare 'params'"
    )


def _assert_valid_param_tree(params):
    assert isinstance(params, (dict, FrozenDict))
    leaves = tree_leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert isinstance(leaf, np.ndarray)


def test_pr38_trainstate_style_restore():
    restored = {
        "model": {
            "params": FrozenDict(
                {
                    "encoder": {
                        "layer1": {
                            "kernel": np.zeros((2, 2)),
                            "bias": np.zeros((2,))
                        }
                    }
                }
            )
        },
        "opt_state": {},
        "rng": None,
        "step": 100,
    }

    params = canonicalize_restored_params(restored)

    assert "encoder" in params
    assert "layer1" in params["encoder"]
    assert params["encoder"]["layer1"]["kernel"].shape == (2, 2)

    _assert_valid_param_tree(params)


def test_pr38_bare_params_restore():
    restored = {
        "params": {
            "decoder": {
                "layer2": {
                    "kernel": np.ones((4, 4)),
                    "bias": np.ones((4,))
                }
            }
        }
    }

    params = canonicalize_restored_params(restored)

    assert "decoder" in params
    assert "layer2" in params["decoder"]
    assert params["decoder"]["layer2"]["kernel"].shape == (4, 4)

    _assert_valid_param_tree(params)


def test_pr38_unsupported_restore_format_errors():
    restored = {
        "weights": {
            "layer": np.zeros((1,))
        }
    }

    with pytest.raises(KeyError):
        canonicalize_restored_params(restored)
