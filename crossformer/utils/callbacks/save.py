from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import flax
import jax
import orbax.checkpoint as ocp

from crossformer.utils.train_utils import TrainState


@dataclass
class SaveCallback:
    """Callback that saves checkpoints under ``save_dir``."""

    save_dir: Path | str | None

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
        self.state_mngr = ocp.CheckpointManager(
            self.ckpt_path,
            ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                enable_async_checkpointing=True,
                enable_background_delete=True,
            ),
        )

        # Keep all params-only checkpoints
        self.params_mngr = ocp.CheckpointManager(
            self.save_dir / "params",
            ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=None,
                enable_async_checkpointing=True,
                enable_background_delete=True,
            ),
        )

    def __call__(self, train_state: TrainState, step: int):
        if self.save_dir is None:
            return

        self.params_mngr.save(
            step,
            args=ocp.args.StandardSave(item=train_state.model.params),
        )
        self.save_extra(train_state)

        if self.state_mngr.should_save(step):
            self.state_mngr.save(
                step,
                args=ocp.args.StandardSave(item=train_state),
            )

    def wait(self):
        """Wait until all async checkpointing is done."""
        if self.save_dir is None:
            return
        self.params_mngr.wait_until_finished()
        self.state_mngr.wait_until_finished()

    def save_extra(self, train_state: TrainState):
        if jax.process_index() != 0:
            return
        model = train_state.model

        config_path = self.ckpt_path / "config.json"
        if not config_path.exists():
            with config_path.open("w") as f:
                json.dump(model.config, f)

        example_batch_path = self.ckpt_path / "example_batch.msgpack"
        if not example_batch_path.exists():
            with example_batch_path.open("wb") as f:
                f.write(flax.serialization.msgpack_serialize(model.example_batch))

        dataset_statistics_path = self.ckpt_path / "dataset_statistics.json"
        if not dataset_statistics_path.exists():
            with dataset_statistics_path.open("w") as f:
                json.dump(
                    jax.tree_map(lambda x: x.tolist(), model.dataset_statistics),
                    f,
                )
