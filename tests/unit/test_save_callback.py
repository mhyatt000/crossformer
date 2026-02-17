from __future__ import annotations

import json
from pathlib import Path

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest

from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.train_utils import TrainState

pytestmark = pytest.mark.unit


class _SimpleModel(nn.Module):
    """One dense layer — mirrors SimpleModel in test_jax_flax.py."""

    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)


@struct.dataclass
class _MinimalModel:
    """Lightweight pytree with the attrs SaveCallback needs."""

    params: dict
    example_batch: dict
    dataset_statistics: dict
    config: dict = struct.field(pytree_node=False)


def _minimal_state() -> TrainState:
    """TrainState backed by a single-layer model — fast, no CrossFormerModel."""
    nn_model = _SimpleModel(features=4)
    params = nn_model.init(jax.random.PRNGKey(0), jnp.ones((1, 8)))["params"]
    model = _MinimalModel(
        params=params,
        config={"type": "test"},
        example_batch={},
        dataset_statistics={},
    )
    tx = optax.sgd(0.01)
    return TrainState.create(rng=jax.random.PRNGKey(0), model=model, tx=tx)


# Both API modes are exercised via this fixture.
@pytest.fixture(params=[False, True], ids=["old_api", "new_api"])
def new_api(request):
    return request.param


# ---------------------------------------------------------------------------
# None save_dir — everything should be a no-op
# ---------------------------------------------------------------------------


class TestNoneSaveDir:
    def test_no_managers_created(self):
        cb = SaveCallback(save_dir=None)
        assert cb.save_dir is None
        assert not hasattr(cb, "state_mngr")
        assert not hasattr(cb, "params_mngr")

    def test_call_is_noop(self):
        cb = SaveCallback(save_dir=None)
        cb(_minimal_state(), step=0)  # must not raise

    def test_wait_is_noop(self):
        cb = SaveCallback(save_dir=None)
        cb.wait()  # must not raise


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


class TestDirCreation:
    def test_creates_state_and_params_dirs(self, tmp_path, new_api):
        SaveCallback(save_dir=tmp_path, new_api=new_api)
        assert (tmp_path / "state").is_dir()
        assert (tmp_path / "params").is_dir()

    def test_resolves_path(self, tmp_path, new_api):
        cb = SaveCallback(save_dir=str(tmp_path), new_api=new_api)
        assert isinstance(cb.save_dir, Path)
        assert cb.save_dir == tmp_path.resolve()

    def test_idempotent_when_dirs_exist(self, tmp_path, new_api):
        (tmp_path / "state").mkdir()
        (tmp_path / "params").mkdir()
        SaveCallback(save_dir=tmp_path, new_api=new_api)  # must not raise

    def test_sets_ckpt_and_params_paths(self, tmp_path, new_api):
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        assert cb.ckpt_path == tmp_path / "state"
        assert cb.params_path == tmp_path / "params"


# ---------------------------------------------------------------------------
# __call__ / checkpointing
# ---------------------------------------------------------------------------


class TestCall:
    def test_params_checkpoint_written(self, tmp_path, new_api):
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(_minimal_state(), step=0)
        cb.wait()
        assert cb.params_mngr.latest_step() == 0

    def test_state_checkpoint_written(self, tmp_path, new_api):
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(_minimal_state(), step=0)
        cb.wait()
        assert cb.state_mngr.latest_step() == 0

    def test_params_all_steps_retained(self, tmp_path, new_api):
        """params_mngr has max_to_keep=None — all steps survive."""
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        state = _minimal_state()
        cb(state, step=0)
        cb(state, step=1)
        cb.wait()
        assert set(cb.params_mngr.all_steps()) == {0, 1}

    def test_state_max_to_keep_one(self, tmp_path, new_api):
        """state_mngr has max_to_keep=1 — only latest step survives."""
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        state = _minimal_state()
        cb(state, step=0)
        cb(state, step=1)
        cb.wait()
        assert cb.state_mngr.latest_step() == 1
        assert len(cb.state_mngr.all_steps()) <= 1


# ---------------------------------------------------------------------------
# wait()
# ---------------------------------------------------------------------------


class TestWait:
    def test_wait_flushes_async_writes(self, tmp_path, new_api):
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(_minimal_state(), step=0)
        cb.wait()
        assert cb.params_mngr.latest_step() == 0


# ---------------------------------------------------------------------------
# save_extra  (API-agnostic — only writes plain files)
# ---------------------------------------------------------------------------


class TestSaveExtra:
    def test_writes_config_json(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())
        config_path = tmp_path / "state" / "config.json"
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == {"type": "test"}

    def test_writes_example_batch_msgpack(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())
        assert (tmp_path / "state" / "example_batch.msgpack").exists()

    def test_writes_dataset_statistics_json(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())
        assert (tmp_path / "state" / "dataset_statistics.json").exists()

    def test_does_not_overwrite_existing_config(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())

        config_path = tmp_path / "state" / "config.json"
        config_path.write_text('{"v": 99}')

        cb.save_extra(_minimal_state())  # second call must not overwrite
        assert json.loads(config_path.read_text()) == {"v": 99}


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_returns_train_state(self, tmp_path, new_api):
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        state = _minimal_state()
        cb(state, step=0)
        cb.wait()
        loaded = cb.load(state, step=0)
        assert type(loaded) is type(state)

    def test_params_round_trip(self, tmp_path, new_api):
        import numpy as np

        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        state = _minimal_state()
        cb(state, step=0)
        cb.wait()
        loaded = cb.load(state, step=0)
        jax.tree.map(np.testing.assert_array_equal, state.model.params, loaded.model.params)

    def test_default_step_loads_latest(self, tmp_path, new_api):
        import numpy as np

        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        state = _minimal_state()
        ones = state.replace(model=state.model.replace(params=jax.tree.map(jnp.ones_like, state.model.params)))
        cb(state, step=0)
        cb(ones, step=1)
        cb.wait()
        loaded = cb.load(state)  # no step → latest (1)
        jax.tree.map(np.testing.assert_array_equal, ones.model.params, loaded.model.params)

    def test_specific_step_loaded(self, tmp_path, new_api):
        import numpy as np

        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        state = _minimal_state()
        zeros = state.replace(model=state.model.replace(params=jax.tree.map(jnp.zeros_like, state.model.params)))
        ones = state.replace(model=state.model.replace(params=jax.tree.map(jnp.ones_like, state.model.params)))
        cb(zeros, step=0)
        cb(ones, step=1)
        cb.wait()
        loaded0 = cb.load(state, step=0)
        jax.tree.map(np.testing.assert_array_equal, zeros.model.params, loaded0.model.params)
        loaded1 = cb.load(state, step=1)
        jax.tree.map(np.testing.assert_array_equal, ones.model.params, loaded1.model.params)

    def test_none_save_dir_raises(self):
        cb = SaveCallback(save_dir=None)
        with pytest.raises(ValueError, match="save_dir is None"):
            cb.load(_minimal_state())
