"""Golden smoke test for the XFlow bundled-action path."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from rich import print
import tyro

from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.config import LowdimTokenizerCfg, ModelCfg, TransformerCfg, XStateTokenizerCfg
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.dummy import make_chunk_steps, make_fake_batch
from crossformer.run.train_step import make_train_step
from crossformer.utils.callbacks.base import flatten_obs
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.train_utils import TrainState


@dataclass
class Config:
    batch_size: int = 2
    obs_horizon: int = 2
    action_horizon: int = 3
    embodiment: tuple[str, ...] = ("cart_gripper", "nav")
    lr: float = 1e-3
    print_spec: bool = False


def _model_config(max_a: int, obs_horizon: int, action_horizon: int) -> dict:
    transformer = TransformerCfg.from_size("dummy", max_horizon=obs_horizon)
    cfg = ModelCfg(
        observation_tokenizers=[
            LowdimTokenizerCfg(
                name="proprio",
                obs_keys=("proprio",),
                dropout_rate=0.0,
                token_drop=0.0,
            ),
            XStateTokenizerCfg(
                num_latents=2,
                num_channels=32,
                num_heads=2,
                num_blocks=1,
                input_drop_prob=0.0,
                latent_drop_prob=0.0,
            ),
        ],
        heads={
            "action": ModuleSpec.create(
                XFlowHead,
                readout_key="readout_action",
                max_dofs=max_a,
                max_horizon=action_horizon,
                num_query_channels=32,
                num_heads=2,
                num_blocks=1,
                num_self_attend_layers=1,
                dropout_prob=0.0,
                flow_steps=2,
                max_action=5.0,
            )
        },
        readouts={"action": 4},
        transformer=transformer,
    )
    return {"model": cfg.create()}


def _model_obs(batch: dict) -> dict:
    obs = {
        "proprio": batch["observation"]["proprio"],
        "timestep_pad_mask": batch["observation"]["timestep_pad_mask"],
        "pad_mask_dict": {"proprio": batch["observation"]["pad_mask_dict"]["proprio"]},
    }
    return flatten_obs(obs, ("proprio",), state=batch["state"], mask=batch["mask"])


def main(cfg: Config) -> None:
    batch = make_fake_batch(
        batch_size=cfg.batch_size,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        embodiment=cfg.embodiment,
    )
    obs = _model_obs(batch)
    task = batch["task"]
    actions = batch["act"]["base"]
    dof_ids = batch["act"]["id"]
    chunk_steps = make_chunk_steps(batch)

    if cfg.print_spec:
        print(spec({"observation": obs, "task": task, "act": batch["act"]}))

    model = CrossFormerModel.from_config(
        _model_config(dof_ids.shape[-1], cfg.obs_horizon, cfg.action_horizon),
        {"observation": obs, "task": task},
        text_processor=None,
        verbose=False,
        rng=jax.random.PRNGKey(0),
        dataset_statistics=None,
    )

    outputs = model.run_transformer(obs, task, obs["timestep_pad_mask"], train=False)
    readout = outputs["readout_action"].tokens
    assert readout.shape[:2] == (cfg.batch_size, cfg.obs_horizon), readout.shape
    assert jnp.all(jnp.isfinite(readout)), "non-finite transformer readout"

    pred = model.sample_actions(
        obs,
        task,
        timestep_pad_mask=obs["timestep_pad_mask"],
        rng=jax.random.PRNGKey(1),
        train=False,
        head_name="action",
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
    )
    assert pred.shape == actions.shape, f"pred={pred.shape} actions={actions.shape}"
    assert jnp.all(jnp.isfinite(pred)), "non-finite sampled actions"

    state = TrainState.create(model=model, tx=optax.sgd(cfg.lr), rng=jax.random.PRNGKey(2))
    train_step = make_train_step(model.module, cfg.lr)
    state, metrics = train_step(
        state,
        obs,
        task,
        obs["timestep_pad_mask"],
        actions,
        dof_ids,
        chunk_steps,
    )
    assert int(state.step) == 1, state.step
    assert jnp.isfinite(metrics["loss"]), metrics
    assert jnp.isfinite(metrics["grad_norm"]), metrics

    print(
        {
            "ok": True,
            "loss": float(metrics["loss"]),
            "pred_shape": tuple(pred.shape),
            "dof_ids": dof_ids.tolist(),
        }
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
