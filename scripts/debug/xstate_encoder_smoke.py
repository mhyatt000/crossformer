from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.model.components.xstate import XStateEncoder


def make_obs() -> dict:
    bsz, win, slots = 2, 1, 5
    base = jnp.arange(bsz * win * slots, dtype=jnp.float32).reshape(bsz, win, slots) / 10.0
    return {
        "state": {
            "base": base,
            "id": jnp.array(
                [
                    [1, 2, 3, 4, 0],
                    [5, 6, 0, 0, 0],
                ],
                dtype=jnp.int32,
            ),
            "view": jnp.array(
                [
                    [0, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0],
                ],
                dtype=jnp.int32,
            ),
        },
        "mask": {
            "state": {
                "base": jnp.array(
                    [
                        [True, True, True, True, False],
                        [True, True, False, False, False],
                    ],
                    dtype=bool,
                )
            }
        },
    }


def main() -> None:
    obs = make_obs()
    enc = XStateEncoder(num_latents=4, num_channels=512, num_heads=16, num_blocks=8)
    params = enc.init(jax.random.PRNGKey(0), obs, train=False)
    out = enc.apply(params, obs, train=False)

    print(f"tokens: {out.tokens.shape} {out.tokens.dtype}")
    print(f"mask:   {out.mask.shape} all={bool(jnp.all(out.mask))}")
    print(f"view:   {None if out.view is None else out.view.shape}")
    print(f"finite: {bool(jnp.all(jnp.isfinite(out.tokens)))}")


if __name__ == "__main__":
    main()
