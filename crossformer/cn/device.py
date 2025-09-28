from __future__ import annotations

from dataclasses import dataclass
import os

from crossformer.cn.base import CN


class Machine(CN):
    pass


class Cluster(CN):
    pass


@dataclass
class Slurm(Cluster):
    """only specific params for now (there are alot)"""

    num_nodes: int = 1
    nodelist: list[str] | str | None = None

    def from_env(self):
        env = os.environ
        slurm = {k: v for k, v in env.items() if k.startswith("SLURM_JOB_")}
        slurm = {k.replace("SLURM_JOB_", "").lower(): v for k, v in slurm.items()}

        num_nodes = int(slurm.get("num_nodes", 1))
        nodelist = slurm.get("nodelist")

        return Slurm(num_nodes=num_nodes, nodelist=nodelist)

    def parse_nodelist(self, nodelist: str):
        """parse node string
        gpu[1-2,4-5] -> [gpu1, gpu2, gpu4, gpu5]
        """
        nodes = []
        for node in nodelist.split(","):
            if "[" not in node:
                nodes.append(node)
            else:
                prefix, suffix = node.split("[")
                suffix = suffix.replace("]", "")
                start, end = suffix.split("-")
                nodes += [f"{prefix}{i}" for i in range(int(start), int(end) + 1)]
        return nodes

    def __post_init__(self):
        if self.nodelist:
            self.nodelist = self.parse_nodelist(self.nodelist)
            assert len(self.nodelist) == self.num_nodes, "num_nodes != len(nodelist)"
