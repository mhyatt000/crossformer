from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from crossformer.cn.base import CN


class WandbMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


@dataclass()
class Wandb(CN):

    project: str = "bafl"
    group: str = "ssl-luc"
    entity: Optional[str] = None

    # id of run to resume... optionally use f'{id}?_step={step}'
    resume_from: Optional[str] = None

    # Specifies how run data is managed, with the following options:
    # -- online (default): Enables sync with W&B in real-time
    # -- offline: Suitable for air-gapped or offline environments; data
    # is saved locally and can be synced later. 
    # -- disabled: Disables all W&B functionality, making the run’s methods no-ops
    # mode: WandbMode = WandbMode.ONLINE

    def mode(self, debug):
        """ Returns the mode for wandb based on the provided settings. """
        m = WandbMode.DISABLED if debug else WandbMode.ONLINE
        return m.value
