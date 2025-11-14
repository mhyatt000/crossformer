from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp
from rich import print

# Import your model
from crossformer.model.act_vae import ACTVAEModel
from crossformer.utils.spec import spec


def build_dummy_batch(
    batch_size=2,
    Tctx=4,
    H=64,
    W=64,
    action_dim=8,
    chunk_len=10,
):
    images = jnp.ones((batch_size, Tctx, H, W, 9), dtype=jnp.float32)
    joints = jnp.zeros((batch_size, Tctx, 7), dtype=jnp.float32)
    gripper = jnp.zeros((batch_size, Tctx, 1), dtype=jnp.float32)
    actions_chunk = jnp.zeros((batch_size, chunk_len, action_dim), dtype=jnp.float32)
    return images, joints, gripper, actions_chunk


def main():
    action_dim = 8
    chunk_len = 10
    d_model = 8 * 4 * 256  # must be compatible with ResNet26FILM output

    model = ACTVAEModel(
        action_dim=action_dim,
        chunk_len=chunk_len,
        d_model=d_model,
        depth=2,  # smaller for debugging
        n_heads=2,
        dropout=0.1,
    )

    print(model)
    images, joints, gripper, actions_chunk = build_dummy_batch(
        batch_size=1,
        Tctx=4,
        H=64,
        W=64,
        action_dim=action_dim,
        chunk_len=chunk_len,
    )

    print(spec(images), spec(joints), spec(gripper), spec(actions_chunk))
    rng = jax.random.PRNGKey(0)
    rng_params, rng_latent_init, rng_drop = jax.random.split(rng, 3)

    # INIT
    rngs = {"params": rng_params, "latent": rng_latent_init, "dropout": rng_drop}
    params = model.init(
        rngs,
        images,
        joints,
        gripper,
        actions_chunk=actions_chunk,
        train=True,
    )
    print("Param tree:")
    # set to builtin print
    print(
        nn.tabulate(model, rngs, depth=2)(
            images,
            joints,
            gripper,
            actions_chunk=actions_chunk,
            train=True,
        )
    )

    # APPLY (train mode)
    bound = model.bind(params, rngs=rngs)
    actions_pred, kl = bound(
        images,
        joints,
        gripper,
        actions_chunk=actions_chunk,
        train=True,
    )

    print("actions_pred shape:", actions_pred.shape)  # expect [B, H, action_dim]
    print("kl shape:", kl.shape)  # expect [B]
    print("kl:", kl)

    # APPLY (eval mode: no actions_chunk -> prior)
    actions_pred_prior, kl_prior = bound(
        images,
        joints,
        gripper,
        actions_chunk=None,
        train=False,
    )
    print("prior actions_pred shape:", actions_pred_prior.shape)
    print("prior kl shape:", kl_prior.shape)
    print("prior kl:", kl_prior)


if __name__ == "__main__":
    main()
