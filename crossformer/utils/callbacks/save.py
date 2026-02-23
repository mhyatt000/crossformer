from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import flax
import jax
import orbax.checkpoint as ocp

from crossformer.utils.train_utils import TrainState


def _make_manager(path: Path, max_to_keep: int | None, new_api: bool) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        enable_async_checkpointing=True,
        enable_background_delete=True,
    )
    if new_api:
        return ocp.CheckpointManager(path, options=options)
    return ocp.CheckpointManager(path, ocp.PyTreeCheckpointer(), options=options)


@dataclass
class SaveCallback:
    """Callback that saves checkpoints under ``save_dir``.

    Args:
        save_dir: Root directory for checkpoints. ``None`` disables all saving.
        new_api: Use the new orbax API (no deprecation warnings). Defaults to
            ``False`` to preserve compatibility with existing checkpoints.
    """

    save_dir: Path | str | None
    new_api: bool = field(default=False)

    def __post_init__(self):
        if self.save_dir is None:
            return

        self.save_dir = Path(self.save_dir).expanduser().resolve()

        if jax.process_index() == 0:
            (self.save_dir / "state").mkdir(parents=True, exist_ok=True)
            (self.save_dir / "params").mkdir(parents=True, exist_ok=True)
            logging.info(f"Created checkpoint dirs under {self.save_dir}")

        self.ckpt_path = self.save_dir / "state"
        self.params_path = self.save_dir / "params"

        # Keep only the latest full TrainState
        self.state_mngr = _make_manager(self.ckpt_path, max_to_keep=1, new_api=self.new_api)
        # Keep all params-only checkpoints
        self.params_mngr = _make_manager(self.params_path, max_to_keep=None, new_api=self.new_api)

    def __call__(self, train_state: TrainState, step: int):
        if self.save_dir is None:
            return

        self.save(train_state, step)
        self.save_extra(train_state)

    def save(self, train_state: TrainState, step: int):
        """Save a checkpoint of the given TrainState at the given step."""

        if self.new_api:
            self.params_mngr.save(step, args=ocp.args.StandardSave(item=train_state.model.params))
        else:
            self.params_mngr.save(step, train_state.model.params)

        if self.state_mngr.should_save(step):
            if self.new_api:
                self.state_mngr.save(step, args=ocp.args.StandardSave(item=train_state))
            else:
                self.state_mngr.save(step, train_state)

    def wait(self):
        """Wait until all async checkpointing is done."""
        if self.save_dir is None:
            return
        self.params_mngr.wait_until_finished()
        self.state_mngr.wait_until_finished()

    def load(self, target: TrainState, step: int | None = None) -> TrainState:
        """Restore model params from a params checkpoint into ``target``.

        Args:
            target: TrainState whose structure is used as the restore template.
                All non-params fields (optimizer state, rng, etc.) are kept from
                ``target``; only ``model.params`` is replaced with checkpoint values.
            step: Checkpoint step to load. Defaults to ``params_mngr.latest_step()``.

        Returns:
            A new TrainState identical to ``target`` except with loaded params.
        """
        if self.save_dir is None:
            raise ValueError("save_dir is None — nothing to load")
        step = step if step is not None else self.params_mngr.latest_step()
        if step is None:
            raise ValueError("No checkpoint found in params directory")

        if self.new_api:
            # Include sharding so orbax places restored arrays on the right devices.
            # This makes cross-topology loads (e.g. save 1-GPU → load 2-GPU) work
            # without any extra steps from the caller.
            abstract = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
                target.model.params,
            )
            params = self.params_mngr.restore(step, args=ocp.args.StandardRestore(abstract))
        else:
            # PyTreeCheckpointer uses items= as a sharding template.
            params = self.params_mngr.restore(step, items=target.model.params)

        return target.replace(model=target.model.replace(params=params))

    def save_extra(self, train_state: TrainState):
        if jax.process_index() != 0:
            return
        model = train_state.model

        # Write alongside params so CrossFormerModel.load_pretrained(params_path) works
        config_path = self.params_path / "config.json"
        if not config_path.exists():
            with config_path.open("w") as f:
                json.dump(model.config, f)

        example_batch_path = self.params_path / "example_batch.msgpack"
        if not example_batch_path.exists():
            with example_batch_path.open("wb") as f:
                f.write(flax.serialization.msgpack_serialize(model.example_batch))

        dataset_statistics_path = self.params_path / "dataset_statistics.json"
        if not dataset_statistics_path.exists():
            with dataset_statistics_path.open("w") as f:
                json.dump(
                    jax.tree_map(lambda x: x.tolist(), model.dataset_statistics),
                    f,
                )
