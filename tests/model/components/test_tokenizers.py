import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.tokenizers import (
    BinTokenizer,
    ImageTokenizer,
    LanguageTokenizer,
    LowdimObsTokenizer,
    TokenLearner,
    generate_proper_pad_mask,
    regex_filter,
    regex_match,
)
from crossformer.model.components.vit_encoders import PatchEncoder
from crossformer.utils.spec import ModuleSpec


def test_generate_proper_pad_mask_combines_sources():
    tokens = jnp.zeros((2, 3, 4, 2))
    pad_mask_dict = {
        "a": jnp.array([[True, False, True], [False, True, True]]),
        "b": jnp.array([[False, False, False], [False, False, False]]),
    }
    mask = generate_proper_pad_mask(tokens, pad_mask_dict, ("a", "b"))
    expected = jnp.array(
        [
            [[True], [False], [True]],
            [[False], [True], [True]],
        ]
    )
    expected = jnp.broadcast_to(expected, tokens.shape[:-1])
    np.testing.assert_array_equal(mask, expected)


def test_token_learner_reduces_token_count():
    module = TokenLearner(num_tokens=2)
    inputs = jnp.ones((1, 4, 8))
    variables = module.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        inputs,
        train=True,
    )
    outputs = module.apply(variables, inputs, train=False)
    assert outputs.shape == (1, module.num_tokens, inputs.shape[-1])


@pytest.mark.parametrize("pattern, value", [("image_.*", True), ("depth_.*", False)])
def test_regex_helpers(pattern, value):
    keys = ["image_primary", "language_instruction"]
    assert regex_match((pattern,), keys[0]) is value
    filtered = regex_filter((pattern,), keys)
    if value:
        assert filtered == [keys[0]]
    else:
        assert filtered == []


def _make_image_tokenizer(**overrides):
    spec = ModuleSpec.create(PatchEncoder, patch_size=1, num_features=8)
    return ImageTokenizer(encoder=spec, **overrides)


def test_image_tokenizer_produces_token_group():
    tokenizer = _make_image_tokenizer()
    observations = {
        "image_primary": jnp.ones((2, 2, 2, 2, 3)),
        "pad_mask_dict": {"image_primary": jnp.array([[True, False], [True, True]])},
    }
    tasks = {}

    variables = tokenizer.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        observations,
        tasks,
        train=False,
    )
    group = tokenizer.apply(variables, observations, tasks, train=False)
    assert isinstance(group, TokenGroup)
    assert group.tokens.shape[0:2] == (2, 2)
    assert group.mask.shape == (2, 2, group.tokens.shape[2])


def test_image_tokenizer_with_token_learner():
    tokenizer = _make_image_tokenizer(use_token_learner=True, num_tokens=3)
    observations = {
        "image_primary": jnp.ones((1, 2, 2, 2, 3)),
        "pad_mask_dict": {"image_primary": jnp.array([[True, True]])},
    }
    tasks = {}
    variables = tokenizer.init(
        {"params": jax.random.PRNGKey(2), "dropout": jax.random.PRNGKey(3)},
        observations,
        tasks,
        train=True,
    )
    group = tokenizer.apply(
        variables,
        observations,
        tasks,
        train=True,
        rngs={"dropout": jax.random.PRNGKey(4)},
    )
    assert group.tokens.shape == (1, 2, 3, group.tokens.shape[-1])


def test_language_tokenizer_uses_pad_mask():
    tokenizer = LanguageTokenizer(proper_pad_mask=True)
    tasks = {
        "language_instruction": jnp.array([[[[1]], [[2]], [[3]]]]),
        "pad_mask_dict": {"language_instruction": jnp.array([[True, True, False]])},
    }
    observations = {}
    group = tokenizer(observations, tasks, train=False)
    assert isinstance(group, TokenGroup)
    assert group.tokens.shape == (1, 3, 1, 1)
    np.testing.assert_array_equal(group.mask, jnp.array([[[True], [True], [False]]]))


def test_bin_tokenizer_encode_and_decode():
    tokenizer = BinTokenizer(n_bins=4, bin_type="uniform", low=0.0, high=1.0)
    inputs = jnp.array([[0.1, 0.5, 0.9]])
    variables = tokenizer.init(jax.random.PRNGKey(5), inputs)
    tokens = tokenizer.apply(variables, inputs)
    assert jnp.issubdtype(tokens.dtype, jnp.integer)
    decoded = tokenizer.apply(variables, tokens, method=tokenizer.decode)
    assert decoded.shape == inputs.shape


def test_lowdim_obs_tokenizer_continuous_and_discrete():
    observations = {"state": jnp.ones((1, 2, 3))}
    tasks = {}
    tokenizer = LowdimObsTokenizer(obs_keys=("state",), discretize=False)
    variables = tokenizer.init({"params": jax.random.PRNGKey(6)}, observations, tasks, train=False)
    group = tokenizer.apply(variables, observations, tasks, train=False)
    assert group.tokens.shape == (1, 2, 3, 1)

    tokenizer_discrete = LowdimObsTokenizer(
        obs_keys=("state",), discretize=True, n_bins=8
    )
    variables_disc = tokenizer_discrete.init(
        {"params": jax.random.PRNGKey(7)}, observations, tasks, train=False
    )
    group_disc = tokenizer_discrete.apply(variables_disc, observations, tasks, train=False)
    assert group_disc.tokens.shape == (1, 2, 3, 8)
    assert jnp.all((group_disc.tokens.sum(axis=-1) == 1))
