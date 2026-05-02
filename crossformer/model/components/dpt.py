from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp


@dataclass
class DPTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    out_channels: int = 1
    readout_layers: tuple[int, ...] = (2, 5, 8, 11)
    feature_dim: int = 256


class PatchEmbed(nn.Module):
    cfg: DPTConfig

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, int, int]:
        # x: [B, H, W, C]
        x = nn.Conv(
            features=self.cfg.embed_dim,
            kernel_size=(self.cfg.patch_size, self.cfg.patch_size),
            strides=(self.cfg.patch_size, self.cfg.patch_size),
            padding="VALID",
            name="patch_proj",
        )(x)

        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        return x, h, w


class TransformerBlock(nn.Module):
    cfg: DPTConfig

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.cfg.num_heads,
            qkv_features=self.cfg.embed_dim,
            out_features=self.cfg.embed_dim,
            deterministic=not train,
        )(y)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(int(self.cfg.embed_dim * self.cfg.mlp_ratio))(y)
        y = nn.gelu(y)
        y = nn.Dense(self.cfg.embed_dim)(y)
        x = x + y
        return x


class ViTBackbone(nn.Module):
    cfg: DPTConfig

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> tuple[list[Array], int, int]:
        x, h, w = PatchEmbed(self.cfg)(x)

        n = h * w
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, n, self.cfg.embed_dim),
        )
        x = x + pos

        outs = []
        for i in range(self.cfg.depth):
            x = TransformerBlock(self.cfg, name=f"block_{i}")(x, train=train)
            if i in self.cfg.readout_layers:
                outs.append(x)

        return outs, h, w


class SpatialFeatureEncoder(nn.Module):
    """Adapt an encoder that returns one NHWC spatial feature map."""

    encoder: nn.Module
    num_readouts: int = 4
    freeze: bool = False

    @nn.compact
    def __call__(
        self,
        x: Array,
        train: bool = False,
    ) -> tuple[tuple[Array, ...], int, int]:
        spatial = self.encoder(x, train=train)
        if isinstance(spatial, tuple):
            spatial = spatial[0]
        if self.freeze:
            spatial = jax.lax.stop_gradient(spatial)
        if spatial.ndim != 4:
            raise ValueError(f"expected NHWC spatial features, got {spatial.shape}")

        b, h, w, c = spatial.shape
        tokens = spatial.reshape(b, h * w, c)
        return (tokens,) * self.num_readouts, h, w


class ReassembleBlock(nn.Module):
    """Convert ViT token sequence back into image-like feature map."""

    out_dim: int
    scale: float

    @nn.compact
    def __call__(self, x: Array, h: int, w: int) -> Array:
        # x: [B, N, C] -> [B, H, W, C]
        b, _, c = x.shape
        x = x.reshape(b, h, w, c)

        x = nn.Conv(self.out_dim, kernel_size=(1, 1), name="proj")(x)

        if self.scale != 1.0:
            new_h = int(h * self.scale)
            new_w = int(w * self.scale)
            x = jax_image_resize(x, new_h, new_w)

        x = nn.Conv(self.out_dim, kernel_size=(3, 3), padding="SAME", name="post_conv")(x)
        return x


def jax_image_resize(x: Array, new_h: int, new_w: int) -> Array:
    # NHWC resize
    return jnp.asarray(
        jax.image.resize(
            x,
            shape=(x.shape[0], new_h, new_w, x.shape[-1]),
            method="bilinear",
        )
    )


class FusionBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: Array, skip: Array | None = None) -> Array:
        if skip is not None:
            if x.shape[1:3] != skip.shape[1:3]:
                x = jax_image_resize(x, skip.shape[1], skip.shape[2])
            x = x + skip

        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Upsample by 2 after fusion, like DPT-style refinement.
        x = jax_image_resize(x, x.shape[1] * 2, x.shape[2] * 2)
        return x


class DPTHead(nn.Module):
    cfg: DPTConfig

    @nn.compact
    def __call__(self, feats: Sequence[Array]) -> Array:
        # feats expected from coarse-to-fine or same-grid manufactured scales.
        x = feats[-1]

        for i, skip in enumerate(reversed(feats[:-1])):
            x = FusionBlock(self.cfg.feature_dim, name=f"fusion_{i}")(x, skip)

        x = nn.Conv(self.cfg.feature_dim // 2, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.cfg.out_channels, (1, 1), padding="SAME")(x)
        return x


class DPT(nn.Module):
    cfg: DPTConfig
    encoder: nn.Module | None = None
    scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        # x: [B, H, W, C]
        input_h, input_w = x.shape[1], x.shape[2]

        encoder = self.encoder
        if encoder is None:
            encoder = ViTBackbone(self.cfg, name="vit")

        tokens, h, w = encoder(x, train=train)
        if not tokens:
            raise ValueError("DPT encoder must return at least one readout")
        if len(tokens) > len(self.scales):
            raise ValueError(f"got {len(tokens)} readouts but only {len(self.scales)} DPT scales")

        feats = [
            ReassembleBlock(
                out_dim=self.cfg.feature_dim,
                scale=self.scales[i],
                name=f"reassemble_{i}",
            )(tok, h, w)
            for i, tok in enumerate(tokens)
        ]

        y = DPTHead(self.cfg, name="head")(feats)

        # Restore to input resolution.
        y = jax_image_resize(y, input_h, input_w)
        return y


if __name__ == "__main__":
    cfg = DPTConfig(
        image_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        out_channels=1,  # depth map
    )
    model = DPT(cfg)
    x = jnp.ones((2, 224, 224, 3))
    variables = model.init(jax.random.PRNGKey(0), x, train=True)
    y = model.apply(variables, x, train=True)
    print(y.shape)
    # (2, 224, 224, 1)
