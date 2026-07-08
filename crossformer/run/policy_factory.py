"""Checkpoint-aware policy loading: trunk dispatch + the IRL wrapper stack.

``load_policy`` is the one place that knows BELAModel exists. It resolves
which model class a checkpoint needs (recorded ``trunk`` key for new
checkpoints, orbax param-tree sniff for old ones) and composes the stack
validated by scripts/debug/server_compare.py:

    ModelPolicy -> ActionDenormWrapper -> GrainlikeWrapper

so raw robot observations go through the real grain pipeline transforms and
predictions come back denormalized.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
from typing import Literal

from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain import metadata
from crossformer.model.bela import BELAModel
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.base_policy import ActionDenormWrapper, ModelPolicy
from crossformer.run.wrappers import PolicyWrapper
from crossformer.run.wrappers.grainlike import GrainlikeWrapper

log = logging.getLogger(__name__)

Trunk = Literal["auto", "bela", "crossformer"]


def _step_metadata_path(path: Path, step: int | None) -> Path | None:
    """Locate ``<step>/default/_METADATA`` for local (params/) or HF-style layouts."""
    root = path / "params" if (path / "params").exists() else path
    if not root.is_dir():
        return None
    if step is not None:
        candidates = [root / str(step)]
    else:
        candidates = sorted(
            (d for d in root.iterdir() if d.name.isdigit()),
            key=lambda d: int(d.name),
            reverse=True,
        )
    for d in candidates:
        meta = d / "default" / "_METADATA"
        if meta.exists():
            return meta
    return None


def resolve_trunk(path: str | Path, step: int | None = None) -> Literal["bela", "crossformer"]:
    """Decide which model class a checkpoint needs.

    Priority: the ``trunk`` key recorded in config.json (checkpoints saved
    after ModelFactory.create started emitting it), else the orbax param-tree
    metadata — a BELA trunk owns ``{readout}_xattn`` params under
    ``crossformer_transformer`` that the block transformer never has — else
    assume the block-transformer trunk.
    """
    path = Path(path).expanduser().resolve()
    with (path / "config.json").open() as f:
        config = json.load(f)
    trunk = config.get("trunk")
    if trunk in ("bela", "crossformer"):
        return trunk

    meta_path = _step_metadata_path(path, step)
    if meta_path is not None:
        with meta_path.open() as f:
            meta = json.load(f)
        keys = meta.get("tree_metadata", meta)
        if any("crossformer_transformer" in k and "_xattn" in k for k in keys):
            return "bela"
        return "crossformer"

    log.warning("could not determine trunk for %s; assuming crossformer", path)
    return "crossformer"


def _trained_image_size(model: CrossFormerModel) -> int:
    # per-camera keys are image_{name}; the stacked-view tokenizer uses a
    # single "image" key with shape (B, W, V, H, W, C) — H is at -3 either way
    sizes = {v.shape[-3] for k, v in model.example_batch["observation"].items() if k.startswith("image")}
    if len(sizes) != 1:
        raise ValueError(f"Ambiguous trained image sizes {sorted(sizes)}; pass resize_to explicitly")
    return sizes.pop()


def load_policy(
    path: str | Path,
    *,
    dataset_name: str,
    step: int | None = None,
    trunk: Trunk = "auto",
    head_name: str = "action",
    flow_steps: int | None = None,
    horizon: int | None = None,
    use_guidance: bool = False,
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation"),
    resize_to: int | None = None,
    shard_fn: Callable | None = None,
) -> PolicyWrapper:
    """Load a checkpoint into the full IRL policy stack.

    Dataset statistics come from the checkpoint (model.dataset_statistics),
    not a data loader, so no mix/grain config is needed. ``resize_to``
    defaults to the image resolution the checkpoint was trained at.
    """
    resolved = resolve_trunk(path, step) if trunk == "auto" else trunk
    model_cls = BELAModel if resolved == "bela" else CrossFormerModel
    log.info("loading %s with %s (trunk=%s)", path, model_cls.__name__, resolved)

    policy = ModelPolicy(
        str(path),
        step=step,
        head_name=head_name,
        guide_keys=guide_keys,
        use_guidance=use_guidance,
        flow_steps=flow_steps,
        horizon=horizon,
        model_cls=model_cls,
    )

    raw_stats = policy.model.dataset_statistics[dataset_name]
    stats = (
        raw_stats
        if isinstance(raw_stats, metadata.DatasetStatistics)
        else metadata.DatasetStatistics.from_json(raw_stats)
    )
    embodiment = Arec.from_name(dataset_name).embodiment

    policy = ActionDenormWrapper(policy, stats, embodiment=embodiment)
    return GrainlikeWrapper(
        policy,
        dataset_name=dataset_name,
        embodiment=embodiment,
        max_a=embodiment.action_dim,
        stats=stats,
        proprio_keys=list(stats.proprio.keys()),
        resize_to=resize_to if resize_to is not None else _trained_image_size(policy.unwrapped().model),
        shard_fn=shard_fn,
        norm_action=False,
    )
