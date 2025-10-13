#
# SET UP MODEL ROLLOUTS (from improve & SIMPLER)
#
from __future__ import annotations


def _model_step(params, batch, rng, train=False):
    """for evaluation in env"""

    # modified for crossformer from octo
    print(spec(batch))

    """
        actions = model.sample_actions(
            batch,
            task,
            model.dataset_statistics['bridge_dataset'],
            head_name='single_arm',
            rng=rng,
        )[0, :, : 7]
        """

    bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
    transformer_embeddings = bound_module.crossformer_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )
    actions = bound_module.heads["single_arm"](
        transformer_embeddings,  # doesnt need rng since its not diffusion
        train=train,
    )

    return actions


@partial(
    jax.jit,
    # state is replicated, batch is data-parallel
    in_shardings=(dp_sharding),
    out_shardings=(replicated_sharding),
    # allows jax to modify `state` in-place, saving a lot of memory
    # donate_argnums=0,
)
def model_step(batch):
    actions = _model_step(train_state.model.params, batch, train_state.rng)
    # act_horizon is 4 for single_arm
    # we act on the last obs horizon
    actions = actions[: cfg.rollout_kwargs.num_envs, -1, :4, :]
    return actions

    use_rollout = cfg.rollout_kwargs.use_rollout
    if use_rollout:
        import simpler_env as simpler
        from simpler_utils import mk_envs

        tasks = [e for e in simpler.ENVIRONMENTS if "widowx" in e]
        # replicates a few times
        tasks = tasks
        venv = mk_envs(tasks, cfg.rollout_kwargs.num_envs)
        instructions = venv.env_method("get_language_instruction")


def transform(batch):
    # zeros = jax.tree.map(lambda arr: jnp.zeros(arr), gapspec)
    batch["observation"]["timestep_pad_mask"] = batch["observation"].pop("pad_mask")

    zeros = jax.tree.map(
        lambda arr: jnp.zeros(
            (
                cfg.data.batch_size - cfg.rollout_kwargs.num_envs,
                *arr.shape[1:],
            )
        ),
        batch,
    )
    batch = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), batch, zeros)

    _instruct = instructions + ["" for _ in range(cfg.data.batch_size - cfg.rollout_kwargs.num_envs)]
    batch["task"] = {"language_instruction": [i.encode("utf-8") for i in _instruct]}
    batch["dataset_name"] = "bridge_dataset"  # dummy variable

    batch = shard(process_batch(batch))
    return batch

    if use_rollout:
        from improve.fm.oxes import OXESimplerInference, PolicyStepper

        stepper = PolicyStepper(
            model_type="func",
            dataset_id="bridge_dataset",  # or google dataset
            func=model_step,
            transform=transform,
        )

        oxes = OXESimplerInference(
            stepper,
            batch_size=cfg.rollout_kwargs.num_envs,
            image_size=224,
        )
        oxes.reset(instructions)

        def og_step(obs):
            _raw, act = oxes.step(obs)
            return act

        eval_callback = EvalCallback(venv, og_step)
