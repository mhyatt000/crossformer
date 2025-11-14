from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from crossformer.model.components.vit_encoders import (
    normalize_images,
    PatchEncoder,
    ResidualUnit,
    ResNet26,
    ResNet26FILM,
    ResNetStage,
    SmallStem,
    SmallStem16,
    SmallStem32,
    StdConv,
    vit_encoder_configs,
    ViTResnet,
    weight_standardize,
)


def test_normalize_images_default_and_imagenet():
    default = normalize_images(jnp.array([[[[0], [255]]]], dtype=jnp.uint8))
    assert default.min() >= -1.0 and default.max() <= 1.0

    imagenet_input = jnp.ones((1, 2, 2, 6), dtype=jnp.uint8)
    imagenet = normalize_images(imagenet_input, img_norm_type="imagenet")
    assert imagenet.shape == imagenet_input.shape


def test_weight_standardize_normalizes_kernel():
    kernel = jnp.arange(3 * 3 * 2 * 4, dtype=jnp.float32).reshape(3, 3, 2, 4)
    standardized = weight_standardize(kernel, axis=[0, 1, 2], eps=1e-5)
    mean = standardized.mean(axis=(0, 1, 2))
    std = standardized.std(axis=(0, 1, 2))
    np.testing.assert_allclose(mean, jnp.zeros_like(mean), atol=1e-6)
    np.testing.assert_allclose(std, jnp.ones_like(std), atol=1e-6)


def test_stdconv_applies_weight_standardization():
    conv = StdConv(features=2, kernel_size=(3, 3))
    inputs = jnp.ones((1, 5, 5, 2))
    variables = conv.init(jax.random.PRNGKey(0), inputs)
    outputs = conv.apply(variables, inputs)

    kernel = variables["params"]["kernel"]
    standardized_kernel = weight_standardize(kernel, axis=[0, 1, 2], eps=1e-5)
    ref_conv = nn.Conv(features=2, kernel_size=(3, 3))
    params = dict(variables["params"])
    ref_params = {
        "params": {
            "kernel": standardized_kernel,
            "bias": params["bias"] if "bias" in params else jnp.zeros((kernel.shape[-1],), kernel.dtype),
        }
    }
    ref_outputs = ref_conv.apply(ref_params, inputs)
    np.testing.assert_allclose(outputs, ref_outputs, atol=1e-5)


def test_patch_encoder_and_small_stem_with_film():
    patch = PatchEncoder(use_film=True, patch_size=2, num_features=4)
    obs = jnp.ones((1, 16, 16, 3))
    cond = jnp.ones((1, 2))
    variables = patch.init(jax.random.PRNGKey(1), obs, train=False, cond_var=cond)
    out = patch.apply(variables, obs, train=False, cond_var=cond)
    assert out.shape[-1] == 4

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
    assert stem_out.shape[-1] == stem.num_features


def test_residual_unit_and_stage():
    unit = ResidualUnit(features=32)
    x = jnp.ones((1, 4, 4, 128))
    vars_unit = unit.init(jax.random.PRNGKey(3), x)
    out_unit = unit.apply(vars_unit, x)
    assert out_unit.shape[-1] == 128

    stage = ResNetStage(block_size=1, nout=32, first_stride=(1, 1))
    vars_stage = stage.init(jax.random.PRNGKey(4), out_unit)
    out_stage = stage.apply(vars_stage, out_unit)
    assert out_stage.shape[-1] == 128


def test_vit_resnet_variants():
    model = ViTResnet(use_film=False, num_layers=(1, 1))
    obs = jnp.ones((1, 8, 8, 3))
    vars_model = model.init(jax.random.PRNGKey(5), obs, train=False)
    out = model.apply(vars_model, obs, train=False)
    assert out.ndim == 4

    model_film = ResNet26FILM()
    cond = jnp.ones((1, 4))
    vars_film = model_film.init(jax.random.PRNGKey(6), obs, train=False, cond_var=cond)
    out_film = model_film.apply(vars_film, obs, train=False, cond_var=cond)
    assert out_film.shape[0] == obs.shape[0]

    resnet26 = ResNet26()
    vars_resnet26 = resnet26.init(jax.random.PRNGKey(7), obs, train=False)
    out_resnet26 = resnet26.apply(vars_resnet26, obs, train=False)
    assert out_resnet26.shape[0] == obs.shape[0]


def test_small_stem_variants_inherit_patch_size():
    obs = jnp.ones((1, 8, 8, 3))
    stem16 = SmallStem16()
    vars16 = stem16.init(jax.random.PRNGKey(8), obs, train=False)
    out16 = stem16.apply(vars16, obs, train=False)
    assert stem16.patch_size == 16
    assert out16.ndim == 4

    stem32 = SmallStem32()
    vars32 = stem32.init(jax.random.PRNGKey(9), obs, train=False)
    out32 = stem32.apply(vars32, obs, train=False)
    assert stem32.patch_size == 32
    assert out32.ndim == 4


def test_vit_encoder_configs_return_modules():
    inputs = jnp.ones((1, 4, 4, 3))
    cond = jnp.ones((1, 4))
    for name, factory in vit_encoder_configs.items():
        module = factory()
        assert isinstance(module, nn.Module)
        kwargs = {"train": False}
        if getattr(module, "use_film", False):
            kwargs["cond_var"] = cond
        variables = module.init(jax.random.PRNGKey(hash(name) & 0xFFFF), inputs, **kwargs)
        out = module.apply(variables, inputs, **kwargs)
        assert out.ndim >= 3
