import numpy as np
from flax.core import FrozenDict
from jax.tree_util import tree_leaves


def _canonicalize_restored_params(restored):
    """
    Minimal local canonicalization helper.

    This mirrors the logic used in crossformer_model.load_pretrained():
    - Orbax restore returns a TrainState-style dict
    - Model parameters live under restored['model']['params']
    """
    assert isinstance(restored, dict)
    assert "model" in restored
    assert "params" in restored["model"]
    return restored["model"]["params"]


def test_orbax_restore_canonicalization_trainstate_format():
    # Simulated Orbax TrainState-style restore output
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
        "step": 123,
    }

    params = _canonicalize_restored_params(restored)

    # Structural assertions
    assert isinstance(params, (dict, FrozenDict))
    assert "encoder" in params
    assert "layer1" in params["encoder"]
    assert "kernel" in params["encoder"]["layer1"]

    # Shape sanity check
    assert params["encoder"]["layer1"]["kernel"].shape == (2, 2)

    # Non-empty PyTree guarantee
    assert len(tree_leaves(params)) > 0
