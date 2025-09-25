"""Vision encoder cookbook echoing :mod:`tests.model.components.test_vit_encoders`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from crossformer.model.components.vit_encoders import (
    PatchEncoder,
    ResNet26,
    ResNet26FILM,
    ResNetStage,
    ResidualUnit,
    SmallStem,
    SmallStem16,
    SmallStem32,
    StdConv,
    ViTResnet,
    normalize_images,
    vit_encoder_configs,
    weight_standardize,
)


def normalization_demo() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compare default and ImageNet normalization strategies."""
    default = normalize_images(jnp.array([[[[0], [255]]]], dtype=jnp.uint8))
    imagenet_input = jnp.ones((1, 2, 2, 6), dtype=jnp.uint8)
    imagenet = normalize_images(imagenet_input, img_norm_type="imagenet")
    return default, imagenet


def weight_standardize_demo() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Standardize convolution kernels and report mean/std."""
    kernel = jnp.arange(3 * 3 * 2 * 4, dtype=jnp.float32).reshape(3, 3, 2, 4)
    standardized = weight_standardize(kernel, axis=[0, 1, 2], eps=1e-5)
    mean = standardized.mean(axis=(0, 1, 2))
    std = standardized.std(axis=(0, 1, 2))
    return mean, std


def stdconv_demo() -> jnp.ndarray:
    """Run :class:`StdConv` and compare with a manual reference convolution."""
    conv = StdConv(features=2, kernel_size=(3, 3))
    inputs = jnp.ones((1, 5, 5, 2))
    variables = conv.init(jax.random.PRNGKey(0), inputs)
    outputs = conv.apply(variables, inputs)
    kernel = variables["params"]["kernel"]
    standardized_kernel = weight_standardize(kernel, axis=[0, 1, 2], eps=1e-5)
    ref_conv = nn.Conv(features=2, kernel_size=(3, 3))
    ref_params = {"params": {"kernel": standardized_kernel, "bias": variables["params"].get("bias", jnp.zeros((2,)))}}
    ref_outputs = ref_conv.apply(ref_params, inputs)
    return outputs - ref_outputs


def film_patch_and_stem_demo() -> tuple[int, int]:
    """Encode images with conditional FiLM modules and report channel counts."""
    patch = PatchEncoder(use_film=True, patch_size=2, num_features=4)
    obs = jnp.ones((1, 16, 16, 3))
    cond = jnp.ones((1, 2))
    patch_vars = patch.init(jax.random.PRNGKey(1), obs, train=False, cond_var=cond)
    patch_out = patch.apply(patch_vars, obs, train=False, cond_var=cond)

    stem = SmallStem(
        use_film=True,
        patch_size=16,
        features=(32, 64),
        strides=(2, 2),
        kernel_sizes=(3, 3),
        padding=(1, 1),
    )
    stem_vars = stem.init(jax.random.PRNGKey(2), obs, train=False, cond_var=cond)
    stem_out = stem.apply(stem_vars, obs, train=False, cond_var=cond)
    return patch_out.shape[-1], stem_out.shape[-1]


def residual_stack_demo() -> tuple[int, int]:
    """Pass data through a residual unit and a short ResNet stage."""
    unit = ResidualUnit(features=32)
    x = jnp.ones((1, 4, 4, 128))
    vars_unit = unit.init(jax.random.PRNGKey(3), x)
    out_unit = unit.apply(vars_unit, x)

    stage = ResNetStage(block_size=1, nout=32, first_stride=(1, 1))
    vars_stage = stage.init(jax.random.PRNGKey(4), out_unit)
    out_stage = stage.apply(vars_stage, out_unit)
    return out_unit.shape[-1], out_stage.shape[-1]


def vit_resnet_variants_demo() -> tuple[int, int, int]:
    """Compare ViT-ResNet hybrids with and without FiLM conditioning."""
    obs = jnp.ones((1, 8, 8, 3))
    vit = ViTResnet(use_film=False, num_layers=(1, 1))
    vit_vars = vit.init(jax.random.PRNGKey(5), obs, train=False)
    vit_out = vit.apply(vit_vars, obs, train=False)

    cond = jnp.ones((1, 4))
    film = ResNet26FILM()
    film_vars = film.init(jax.random.PRNGKey(6), obs, train=False, cond_var=cond)
    film_out = film.apply(film_vars, obs, train=False, cond_var=cond)

    resnet26 = ResNet26()
    resnet_vars = resnet26.init(jax.random.PRNGKey(7), obs, train=False)
    resnet_out = resnet26.apply(resnet_vars, obs, train=False)
    return vit_out.ndim, film_out.shape[0], resnet_out.shape[0]


def small_stem_variants_demo() -> tuple[int, int]:
    """Show the default patch sizes baked into SmallStem16/32 variants."""
    obs = jnp.ones((1, 8, 8, 3))
    stem16 = SmallStem16()
    vars16 = stem16.init(jax.random.PRNGKey(8), obs, train=False)
    _ = stem16.apply(vars16, obs, train=False)
    stem32 = SmallStem32()
    vars32 = stem32.init(jax.random.PRNGKey(9), obs, train=False)
    _ = stem32.apply(vars32, obs, train=False)
    return stem16.patch_size, stem32.patch_size


def config_catalog_demo() -> dict[str, type[nn.Module]]:
    """Materialize the preset factory to emphasize extensibility."""
    catalog = {}
    inputs = jnp.ones((1, 4, 4, 3))
    cond = jnp.ones((1, 4))
    for name, factory in vit_encoder_configs.items():
        module = factory()
        kwargs = dict(train=False)
        if getattr(module, "use_film", False):
            kwargs["cond_var"] = cond
        variables = module.init(jax.random.PRNGKey(hash(name) & 0xFFFF), inputs, **kwargs)
        _ = module.apply(variables, inputs, **kwargs)
        catalog[name] = type(module)
    return catalog


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("normalize", [arr.shape for arr in normalization_demo()])
    print("weight standardize", weight_standardize_demo())
    print("stdconv diff", jnp.abs(stdconv_demo()).max())
    print("film patch/stem", film_patch_and_stem_demo())
    print("residual stage", residual_stack_demo())
    print("variants", vit_resnet_variants_demo())
    print("stem variants", small_stem_variants_demo())
    print("catalog", list(config_catalog_demo().keys())[:3], "...")
