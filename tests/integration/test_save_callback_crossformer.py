from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from crossformer.model.crossformer_model import _download_from_huggingface, CrossFormerModel
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.train_utils import TrainState

pytestmark = pytest.mark.integration
pytestmark = pytest.mark.slow

HF_REPO = "hf://rail-berkeley/crossformer"


@pytest.fixture(scope="module")
def hf_model() -> CrossFormerModel:
    ckpt_path = _download_from_huggingface(HF_REPO.removeprefix("hf://"))
    return CrossFormerModel.load_pretrained(ckpt_path)


@pytest.fixture(scope="module")
def hf_state(hf_model) -> TrainState:
    return TrainState.create(
        rng=jax.random.PRNGKey(0),
        model=hf_model,
        tx=optax.sgd(0.01),
    )


# ---------------------------------------------------------------------------
# SaveCallback.load round-trip with CrossFormerModel
# ---------------------------------------------------------------------------


class TestSaveCallbackWithCrossFormer:
    def test_params_round_trip_legacy(self, tmp_path, hf_state):
        """Save and reload params via legacy API; values must be identical."""
        cb = SaveCallback(save_dir=tmp_path, new_api=False)
        cb(hf_state, step=0)
        cb.wait()
        loaded = cb.load(hf_state, step=0)
        jax.tree.map(
            np.testing.assert_array_equal,
            hf_state.model.params,
            loaded.model.params,
        )

    def test_params_round_trip_new_api(self, tmp_path, hf_state):
        """Save and reload params via new API; values must be identical."""
        cb = SaveCallback(save_dir=tmp_path, new_api=True)
        cb(hf_state, step=0)
        cb.wait()
        loaded = cb.load(hf_state, step=0)
        jax.tree.map(
            np.testing.assert_array_equal,
            hf_state.model.params,
            loaded.model.params,
        )

    def test_load_pretrained_compat(self, tmp_path, hf_state):
        """CrossFormerModel.load_pretrained can read a SaveCallback params dir."""
        cb = SaveCallback(save_dir=tmp_path, new_api=False)
        cb(hf_state, step=0)
        cb.wait()
        model = CrossFormerModel.load_pretrained(str(tmp_path / "params"))
        jax.tree.map(
            np.testing.assert_array_equal,
            hf_state.model.params,
            model.params,
        )

    def test_metadata_written_to_params_dir(self, tmp_path, hf_state):
        """config.json / example_batch / dataset_statistics land in params/."""
        cb = SaveCallback(save_dir=tmp_path)
        cb(hf_state, step=0)
        cb.wait()
        params_dir = tmp_path / "params"
        assert (params_dir / "config.json").exists()
        assert (params_dir / "example_batch.msgpack").exists()
        assert (params_dir / "dataset_statistics.json").exists()

    def test_state_not_required_for_load(self, tmp_path, hf_state):
        """load() returns correct type without touching state_mngr."""
        cb = SaveCallback(save_dir=tmp_path, new_api=False)
        cb(hf_state, step=0)
        cb.wait()
        loaded = cb.load(hf_state, step=0)
        assert type(loaded) is type(hf_state)

    def test_multi_step_latest(self, tmp_path, hf_state):
        """Default step resolves to latest params checkpoint."""
        ones = hf_state.replace(model=hf_state.model.replace(params=jax.tree.map(jnp.ones_like, hf_state.model.params)))
        cb = SaveCallback(save_dir=tmp_path, new_api=False)
        cb(hf_state, step=0)
        cb(ones, step=1)
        cb.wait()
        loaded = cb.load(hf_state)  # no step → latest
        jax.tree.map(
            np.testing.assert_array_equal,
            ones.model.params,
            loaded.model.params,
        )
