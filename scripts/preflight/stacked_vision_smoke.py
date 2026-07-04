"""Golden smoke test for the stacked multiview vision path.

Pulls one real batch from the grain loader and runs it through the stacked
tokenizer (TIPS default) and the CrossFormerTransformer, verifying the
loader -> model contract: stacked image (B, T, V, H, W, C), per-token view ids
1..V that survive the train-time view permutation, view_mask handling, and the
view_embed table in the transformer.

Usage:
    uv run scripts/preflight/stacked_vision_smoke.py --data.mix xgym_lift_single --mp 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from rich import print
import tyro

from crossformer.cn.dataset import Dataset
from crossformer.data.grain.loader import GrainDataFactory
from crossformer.model.components.multiview import (
    StackedDinoTokenizer,
    StackedTipsTokenizer,
    TIPS_VARIANT_DEFAULT,
)
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.crossformer_module import CrossFormerTransformer
from crossformer.utils.spec import spec


@dataclass
class Config:
    data: Dataset = Dataset().field()
    mp: int = 0
    seed: int = 42
    window_size: int = 1

    encoder: Literal["tips", "dino"] = "tips"
    tips_variant: str = TIPS_VARIANT_DEFAULT
    print_spec: bool = False


def make_tokenizer(cfg: Config) -> StackedTipsTokenizer | StackedDinoTokenizer:
    if cfg.encoder == "tips":
        return StackedTipsTokenizer(variant=cfg.tips_variant)
    return StackedDinoTokenizer()


def main(cfg: Config) -> None:
    factory = GrainDataFactory(cfg.mp)
    loader = factory.make(cfg, shard_fn=lambda b: b, train=True)
    batch = next(iter(loader.dataset))

    if cfg.print_spec:
        print(spec(batch, simple=True))

    img = np.asarray(batch["observation"]["image"])
    assert img.ndim == 6, f"expected stacked image (B, T, V, H, W, C), got {img.shape}"
    B, T, V = img.shape[:3]
    print(f"image: {img.shape} {img.dtype}  (B={B} T={T} V={V})")

    obs = {"image": jnp.asarray(img)}
    view_mask = batch.get("mask", {}).get("view")
    if view_mask is not None:
        obs["view_mask"] = jnp.asarray(np.asarray(view_mask)).reshape(B, 1, V)
        print(f"view_mask: {obs['view_mask'].shape} real views per sample: {np.asarray(view_mask).sum(-1)}")
    pad_mask = jnp.ones((B, T), dtype=bool)

    #
    # 1. Tokenizer alone
    #

    tok = make_tokenizer(cfg)
    rngs = {"params": jax.random.PRNGKey(cfg.seed), "dropout": jax.random.PRNGKey(cfg.seed + 1)}
    variables = tok.init(rngs, obs, train=False)
    out = tok.apply(variables, obs, train=True, rngs={"dropout": jax.random.PRNGKey(cfg.seed + 2)})

    tokens, mask, view = out.tokens, out.mask, out.view
    print(f"tokens: {tokens.shape}  mask: {mask.shape}  view: {view.shape}")
    assert view is not None, "tokenizer must emit per-token view ids"
    assert tokens.shape[:2] == (B, T) and mask.shape == tokens.shape[:-1] == view.shape
    assert jnp.all(jnp.isfinite(tokens)), "non-finite tokens"

    views = np.asarray(view)
    N = tokens.shape[-2] // V
    assert set(np.unique(views).tolist()) <= set(range(1, V + 1)), np.unique(views)
    for v in range(1, V + 1):
        assert int((views == v).sum()) == B * T * N, f"view {v} token count mismatch"

    # padded view slots (mask.view False) must be fully masked out
    if view_mask is not None:
        vm = np.asarray(view_mask).reshape(B, V).astype(bool)
        m = np.asarray(mask)
        for b in range(B):
            for v in range(1, V + 1):
                if not vm[b, v - 1]:
                    assert not m[b][views[b] == v].any(), f"sample {b}: padded view {v} not masked"

    #
    # 2. Through the transformer (view_embed + view ids on outputs)
    #

    token_embedding_size, transformer_kwargs = common_transformer_sizes("dummy")
    model = CrossFormerTransformer(
        observation_tokenizers={"image": make_tokenizer(cfg)},
        task_tokenizers={},
        readouts={"action": 4},
        transformer_kwargs=transformer_kwargs,
        token_embedding_size=token_embedding_size,
        max_horizon=max(T, 2),
        repeat_task_tokens=False,
    )
    params = model.init(rngs, obs, {}, pad_mask, train=False)["params"]
    assert "view_embed" in params, list(params.keys())
    outputs = model.apply(
        {"params": params}, obs, {}, pad_mask, train=True, rngs={"dropout": jax.random.PRNGKey(cfg.seed + 3)}
    )

    obs_group = outputs["obs"]
    assert obs_group.view is not None and obs_group.view.shape == obs_group.mask.shape
    assert outputs["readout_action"].view is None  # readouts carry no view identity
    assert jnp.all(jnp.isfinite(outputs["obs"].tokens)), "non-finite transformer outputs"
    print(f"view_embed: {params['view_embed']['embedding'].shape}")
    print(f"obs group: tokens {obs_group.tokens.shape}  views {np.unique(np.asarray(obs_group.view))}")

    print("[green]stacked vision smoke ok[/green]")


if __name__ == "__main__":
    main(tyro.cli(Config))
