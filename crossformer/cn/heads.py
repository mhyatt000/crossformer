from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crossformer.cn.base import CN
from crossformer.cn.dataset import Head
from crossformer.cn.dataset.action import HEAD2SPACE
from crossformer.model.components.heads import AdjFlowHead, DiffusionActionHead, FlowMatchingActionHead, L1ActionHead
from crossformer.utils.spec import ModuleSpec


class ModuleE(Enum):
    L1 = L1ActionHead
    DIFFUSION = DiffusionActionHead
    FLOW = FlowMatchingActionHead
    ADJFLOW = AdjFlowHead


@dataclass
class HeadFactory(CN):
    horizon: int = 20
    module: ModuleE = ModuleE.FLOW
    steps: int = 10  # diffusion/flow steps

    def __post_init__(self):
        if self.module == ModuleE.DIFFUSION:
            assert self.steps, "Diffusion steps must be 1+"
        if self.module == ModuleE.L1:
            pass
        if isinstance(self.module.value, FlowMatchingActionHead | AdjFlowHead):
            assert self.steps, "Flow steps must be 1+"

        h = Head[self.name.upper()]
        self.dim = HEAD2SPACE[h]

    def kwargs(self):
        d = {
            "pool_strategy": "use_map",
            "clip_pred": False,
            "loss_weight": 1.0,
            "constrain_loss_dims": True,
            "readout_key": f"readout_{self.name}",
            "num_preds": 0,
        }
        if self.module == ModuleE.DIFFUSION:
            d["diffusion_steps"] = self.steps
        if isinstance(self.module.value, FlowMatchingActionHead | AdjFlowHead):
            d["flow_steps"] = self.steps

        return d

    def create(self):
        return ModuleSpec.create(
            self.module.value,
            action_horizon=self.horizon,
            action_dim=self.dim.value,
            **self.kwargs(),
        )


# TODO deprecate single_arm and we should remove this eventually
_SINGLE = "single_arm"
_SINGLE = "single"
