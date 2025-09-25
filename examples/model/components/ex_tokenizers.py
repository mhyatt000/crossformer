"""Tokenizer usage notes referencing :mod:`tests.model.components.test_tokenizers`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


def pad_mask_demo() -> jnp.ndarray:
    """Combine per-modality pad masks into a single broadcastable array."""
    tokens = jnp.zeros((2, 3, 4, 2))
    pad_mask_dict = {
        "a": jnp.array([[True, False, True], [False, True, True]]),
        "b": jnp.zeros((2, 3), dtype=bool),
    }
    return generate_proper_pad_mask(tokens, pad_mask_dict, ("a", "b"))


def token_learner_demo() -> jnp.ndarray:
    """Learn a compact set of visual tokens with dropout disabled."""
    module = TokenLearner(num_tokens=2)
    inputs = jnp.ones((1, 4, 8))
    variables = module.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        inputs,
        train=True,
    )
    return module.apply(variables, inputs, train=False)


def regex_helper_demo(pattern: str) -> tuple[bool, list[str]]:
    """Check whether token keys match a regex and filter accordingly."""
    keys = ["image_primary", "language_instruction"]
    return regex_match((pattern,), keys[0]), regex_filter((pattern,), keys)


def _make_image_tokenizer(**overrides) -> ImageTokenizer:
    spec = ModuleSpec.create(PatchEncoder, patch_size=1, num_features=8)
    return ImageTokenizer(encoder=spec, **overrides)


def image_tokenizer_demo() -> TokenGroup:
    """Encode image observations into spatial tokens."""
    tokenizer = _make_image_tokenizer()
    observations = {
        "image_primary": jnp.ones((2, 2, 2, 2, 3)),
        "pad_mask_dict": {"image_primary": jnp.array([[True, False], [True, True]])},
    }
    tasks: dict[str, jnp.ndarray] = {}
    variables = tokenizer.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        observations,
        tasks,
        train=False,
    )
    return tokenizer.apply(variables, observations, tasks, train=False)


def image_tokenizer_with_token_learner_demo() -> TokenGroup:
    """Chain token learner pooling after convolutional encoding."""
    tokenizer = _make_image_tokenizer(use_token_learner=True, num_tokens=3)
    observations = {
        "image_primary": jnp.ones((1, 2, 2, 2, 3)),
        "pad_mask_dict": {"image_primary": jnp.array([[True, True]])},
    }
    tasks: dict[str, jnp.ndarray] = {}
    variables = tokenizer.init(
        {"params": jax.random.PRNGKey(2), "dropout": jax.random.PRNGKey(3)},
        observations,
        tasks,
        train=True,
    )
    return tokenizer.apply(
        variables,
        observations,
        tasks,
        train=True,
        rngs={"dropout": jax.random.PRNGKey(4)},
    )


def language_tokenizer_demo() -> TokenGroup:
    """Turn tokenized language instructions into padded token groups."""
    tokenizer = LanguageTokenizer(proper_pad_mask=True)
    tasks = {
        "language_instruction": jnp.array([[[[1]], [[2]], [[3]]]]),
        "pad_mask_dict": {"language_instruction": jnp.array([[True, True, False]])},
    }
    observations: dict[str, jnp.ndarray] = {}
    return tokenizer(observations, tasks, train=False)


def bin_tokenizer_demo() -> jnp.ndarray:
    """Discretize scalar inputs and decode them back to the original scale."""
    tokenizer = BinTokenizer(n_bins=4, bin_type="uniform", low=0.0, high=1.0)
    inputs = jnp.array([[0.1, 0.5, 0.9]])
    variables = tokenizer.init(jax.random.PRNGKey(5), inputs)
    tokens = tokenizer.apply(variables, inputs)
    decoded = tokenizer.apply(variables, tokens, method=tokenizer.decode)
    return decoded


def lowdim_tokenizer_demo() -> tuple[TokenGroup, TokenGroup]:
    """Compare continuous and discretized encodings of low-dimensional observations."""
    observations = {"state": jnp.ones((1, 2, 3))}
    tasks: dict[str, jnp.ndarray] = {}
    tokenizer_cont = LowdimObsTokenizer(obs_keys=("state",), discretize=False)
    vars_cont = tokenizer_cont.init(
        {"params": jax.random.PRNGKey(6)}, observations, tasks, train=False
    )
    group_cont = tokenizer_cont.apply(vars_cont, observations, tasks, train=False)

    tokenizer_disc = LowdimObsTokenizer(obs_keys=("state",), discretize=True, n_bins=8)
    vars_disc = tokenizer_disc.init(
        {"params": jax.random.PRNGKey(7)}, observations, tasks, train=False
    )
    group_disc = tokenizer_disc.apply(vars_disc, observations, tasks, train=False)
    return group_cont, group_disc


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("pad_mask", pad_mask_demo().shape)
    print("token_learner", token_learner_demo().shape)
    print("regex", regex_helper_demo("image_.*"))
    print("image", image_tokenizer_demo().tokens.shape)
    print("image+learner", image_tokenizer_with_token_learner_demo().tokens.shape)
    print("language", language_tokenizer_demo().tokens.shape)
    print("bin", bin_tokenizer_demo().shape)
    cont, disc = lowdim_tokenizer_demo()
    print("lowdim cont", cont.tokens.shape, "disc", disc.tokens.shape)
