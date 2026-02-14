"""Integration: Config-driven model creation and initialization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.cn import ModelFactory, Size, Train

from .conftest import make_fake_batch, requires_gpu
from .helpers import assert_finite, init_train_state


@pytest.mark.integration
class TestConfigLoading:
    """Test loading and using tyro-based config objects."""

    def test_cn_train_config_loads_and_initializes(self):
        """Load a cn.Train config and initialize model from it."""
        # Create a minimal Train config using tyro dataclass
        train_cfg = Train(
            debug=True,
            steps=100,
            seed=42,
        )

        # Verify config has nested objects
        assert train_cfg.model is not None
        assert train_cfg.optimizer is not None
        assert train_cfg.data is not None

        # Verify ModelFactory can create model config
        model_config = train_cfg.model.create()
        assert "model" in model_config
        assert "observation_tokenizers" in model_config["model"]
        assert "heads" in model_config["model"]
        assert len(model_config["model"]["heads"]) > 0

    @requires_gpu
    def test_cn_model_factory_with_dummy_size(self):
        """ModelFactory with DUMMY size for fast testing."""
        # Create config with dummy transformer size
        model_factory = ModelFactory(size=Size.DUMMY, heads=["single"])
        model_config = model_factory.create()

        # Verify the config structure is correct
        assert model_config["model"]["token_embedding_size"] == 256  # dummy size
        assert "single" in model_config["model"]["heads"]
        assert model_config["model"]["transformer_kwargs"]["num_layers"] == 1

    @requires_gpu
    def test_cn_train_config_full_initialization(self):
        """Full initialization chain: cn.Train -> model -> TrainState."""
        # Create Train config
        train_cfg = Train(
            debug=True,
            steps=10,
            seed=42,
        )
        train_cfg.model.size = Size.DUMMY
        train_cfg.model.heads = ["single"]

        # Extract model config
        model_config = train_cfg.model.create()

        # Create fake batch
        batch = jax.tree.map(jnp.asarray, make_fake_batch(rng=np.random.default_rng(42)))

        # Initialize TrainState using config
        state = init_train_state(
            jax.random.PRNGKey(train_cfg.seed),
            batch,
            model_config,
        )

        # Verify state is valid
        assert state.step == 0
        assert state.model.params is not None
        assert state.opt_state is not None
        assert_finite(state.model.params)

    @requires_gpu
    def test_cn_config_multihead_setup(self):
        """cn.Train config with multiple action heads."""
        train_cfg = Train(debug=True, seed=42)
        train_cfg.model.size = Size.DUMMY
        train_cfg.model.heads = ["single", "bimanual"]

        model_config = train_cfg.model.create()

        # Verify both heads are present
        assert "single" in model_config["model"]["heads"]
        assert "bimanual" in model_config["model"]["heads"]

        # Verify readouts match heads
        assert len(model_config["model"]["readouts"]) == 2
