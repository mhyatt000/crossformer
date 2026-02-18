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
        config_path = tmp_path / "params" / "config.json"
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == {"type": "test"}

    def test_writes_example_batch_msgpack(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())
        assert (tmp_path / "params" / "example_batch.msgpack").exists()

    def test_writes_dataset_statistics_json(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())
        assert (tmp_path / "params" / "dataset_statistics.json").exists()

    def test_does_not_overwrite_existing_config(self, tmp_path):
        cb = SaveCallback(save_dir=tmp_path)
        cb.save_extra(_minimal_state())

        config_path = tmp_path / "params" / "config.json"
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


# ---------------------------------------------------------------------------
# Sharding helpers & fixture
# ---------------------------------------------------------------------------

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np


@pytest.fixture
def mesh2():
    """2-device mesh; skips the test if fewer than 2 GPUs are visible."""
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip("requires 2+ GPUs")
    return Mesh(devs, ("model",))


def _state_with_sharding(sharding=None) -> TrainState:
    """Minimal TrainState with params placed on *sharding* (or unsharded if None)."""
    nn_model = _SimpleModel(features=4)
    params = nn_model.init(jax.random.PRNGKey(0), jnp.ones((1, 8)))["params"]
    if sharding is not None:
        params = jax.device_put(params, sharding)
    model = _MinimalModel(
        params=params,
        config={"type": "test"},
        example_batch={},
        dataset_statistics={},
    )
    return TrainState.create(rng=jax.random.PRNGKey(0), model=model, tx=optax.sgd(0.01))


def _assert_params_equal(a, b):
    jax.tree.map(np.testing.assert_array_equal, a, b)


# ---------------------------------------------------------------------------
# Sharded round-trips and cross-topology loads
# ---------------------------------------------------------------------------


class TestShardedLoad:
    """Cover every save-sharding x load-sharding combination.

    "1 GPU"  == SingleDeviceSharding on devices[0]
    "2 GPU replicated" == NamedSharding(mesh2, P())       -- both devices hold full copy
    "2 GPU sharded"    == NamedSharding(mesh2, P('model'))-- data split across devices
    """

    # --- same-topology round-trips ---

    def test_replicated_round_trip(self, tmp_path, new_api, mesh2):
        rep = NamedSharding(mesh2, P())
        state = _state_with_sharding(rep)
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(state, step=0)
        cb.wait()
        _assert_params_equal(state.model.params, cb.load(state, step=0).model.params)

    def test_sharded_round_trip(self, tmp_path, new_api, mesh2):
        shard = NamedSharding(mesh2, P("model"))
        state = _state_with_sharding(shard)
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(state, step=0)
        cb.wait()
        _assert_params_equal(state.model.params, cb.load(state, step=0).model.params)

    # --- cross-topology: 1 GPU → 2 GPU ---

    def test_save_single_load_replicated(self, tmp_path, new_api, mesh2):
        """Save on 1 GPU, load replicated onto 2 GPUs."""
        s_single = _state_with_sharding(jax.devices()[0])
        s_rep = _state_with_sharding(NamedSharding(mesh2, P()))
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(s_single, step=0)
        cb.wait()
        _assert_params_equal(s_single.model.params, cb.load(s_rep, step=0).model.params)

    def test_save_single_load_sharded(self, tmp_path, new_api, mesh2):
        """Save on 1 GPU, load data-parallel sharded onto 2 GPUs."""
        s_single = _state_with_sharding(jax.devices()[0])
        s_shard = _state_with_sharding(NamedSharding(mesh2, P("model")))
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(s_single, step=0)
        cb.wait()
        _assert_params_equal(s_single.model.params, cb.load(s_shard, step=0).model.params)

    # --- cross-topology: 2 GPU → 1 GPU ---

    def test_save_sharded_load_single(self, tmp_path, new_api, mesh2):
        """Save sharded on 2 GPUs, load onto 1 GPU."""
        s_shard = _state_with_sharding(NamedSharding(mesh2, P("model")))
        s_single = _state_with_sharding(jax.devices()[0])
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(s_shard, step=0)
        cb.wait()
        _assert_params_equal(s_shard.model.params, cb.load(s_single, step=0).model.params)

    def test_save_replicated_load_single(self, tmp_path, new_api, mesh2):
        """Save replicated on 2 GPUs, load onto 1 GPU."""
        s_rep = _state_with_sharding(NamedSharding(mesh2, P()))
        s_single = _state_with_sharding(jax.devices()[0])
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(s_rep, step=0)
        cb.wait()
        _assert_params_equal(s_rep.model.params, cb.load(s_single, step=0).model.params)

    # --- cross-topology: 2 GPU sharding change ---

    def test_save_replicated_load_sharded(self, tmp_path, new_api, mesh2):
        """Save replicated, load data-parallel (different 2-GPU layout)."""
        s_rep = _state_with_sharding(NamedSharding(mesh2, P()))
        s_shard = _state_with_sharding(NamedSharding(mesh2, P("model")))
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(s_rep, step=0)
        cb.wait()
        _assert_params_equal(s_rep.model.params, cb.load(s_shard, step=0).model.params)

    def test_save_sharded_load_replicated(self, tmp_path, new_api, mesh2):
        """Save data-parallel, load replicated (different 2-GPU layout)."""
        s_shard = _state_with_sharding(NamedSharding(mesh2, P("model")))
        s_rep = _state_with_sharding(NamedSharding(mesh2, P()))
        cb = SaveCallback(save_dir=tmp_path, new_api=new_api)
        cb(s_shard, step=0)
        cb.wait()
        _assert_params_equal(s_shard.model.params, cb.load(s_rep, step=0).model.params)
