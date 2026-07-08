"""Serve a BELA/PIO (or crossformer/xflow) checkpoint for IRL rollouts.

Thin entry point over run.policy_factory.load_policy — the same
ModelPolicy + ActionDenormWrapper + GrainlikeWrapper stack that
scripts/debug/server_compare.py validated against the training pipeline.
Trunk dispatch (BELAModel vs CrossFormerModel) is automatic.

Example:
    uv run scripts/serve/bela.py --path ~/bafl/luc-ssl/<run>/params \
        --dataset-name xgym_sweep_single --port 8001
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich import print
import tyro
from webpolicy.server import Server

from crossformer.run.policy_factory import load_policy, Trunk


@dataclass
class Config:
    path: Path = tyro.MISSING
    dataset_name: str = tyro.MISSING  # stats source for action denorm / proprio norm (e.g. xgym_sweep_single)
    step: int | None = None
    trunk: Trunk = "auto"
    head_name: str = "action"
    flow_steps: int | None = None  # None keeps the checkpoint's value
    horizon: int | None = None  # None keeps the checkpoint's value
    use_guidance: bool = False
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")
    resize_to: int | None = None  # None derives from the checkpoint's trained image size
    host: str = "0.0.0.0"
    port: int = 8001
    warmup: bool = True
    viser: bool = True  # plot step inputs/outputs (images, joints, kp3dc) to a viser viewer
    viser_port: int = 8080


def main(cfg: Config) -> None:
    print(cfg)
    policy = load_policy(
        cfg.path,
        dataset_name=cfg.dataset_name,
        step=cfg.step,
        trunk=cfg.trunk,
        head_name=cfg.head_name,
        flow_steps=cfg.flow_steps,
        horizon=cfg.horizon,
        use_guidance=cfg.use_guidance,
        guide_keys=cfg.guide_keys,
        resize_to=cfg.resize_to,
    )
    if cfg.viser:
        from crossformer.run.viewer import PolicyViewer, ViserWrappedPolicy

        policy = ViserWrappedPolicy(policy, PolicyViewer(host=cfg.host, port=cfg.viser_port))
        print(f"viser viewer on {cfg.host}:{cfg.viser_port}")
    if cfg.warmup:
        # forwarded to ModelPolicy.warmup via PolicyWrapper.__getattr__;
        # compiles the serve path (accumulate=False) on example_batch
        policy.warmup(accumulate=False)
    print(f"serving on {cfg.host}:{cfg.port}")
    Server(policy, host=cfg.host, port=cfg.port).serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
