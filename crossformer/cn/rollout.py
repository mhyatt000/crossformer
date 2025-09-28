from __future__ import annotations

from dataclasses import dataclass

from crossformer.cn.base import CN


@dataclass()
class Rollout(CN):
    num_envs: int = 4
    use: bool = False
