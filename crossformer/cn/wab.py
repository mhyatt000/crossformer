from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import crossformer
from crossformer.cn.base import CN
import wandb

StrPath: TypeAlias = str | Path


class WandbMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


@dataclass()
class Wandb(CN):
    project: str = "bafl"
    group: str = "ssl-luc"
    entity: str | None = None

    # id of run to resume... optionally use f'{id}?_step={step}'
    resume_from: str | None = None
    use: bool = True  # to use or not to use wandb
    dir: StrPath = crossformer.ROOT

    # Specifies how run data is managed, with the following options:
    # -- online (default): Enables sync with W&B in real-time
    # -- offline: Suitable for air-gapped or offline environments; data
    # is saved locally and can be synced later.
    # -- disabled: Disables all W&B functionality, making the run methods no-ops
    # mode: WandbMode = WandbMode.ONLINE

    def mode(self, use: bool):
        """Returns the mode for wandb based on the provided settings."""
        m = WandbMode.DISABLED if not use else WandbMode.ONLINE
        return m.value

    def initialize(self, cfg, name=None):
        run = wandb.init(
            project=self.project,
            group=self.group,
            entity=self.entity,
            dir=str(self.dir),
            mode=cfg.wandb.mode(self.use),
            config=cfg,
            **{"name": name} if name is not None else {},
        )
        return run
