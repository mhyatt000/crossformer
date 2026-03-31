from __future__ import annotations

import abc
from collections import deque
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from webpolicy.base_policy import BasePolicy

from crossformer.embody import MASK_ID
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer
from crossformer.utils.tree.core import drop_fn


class PolicyWrapper(BasePolicy, abc.ABC):
    """ABC for wrappers that pre/post-process around an inner policy."""

    def __init__(self, inner: BasePolicy):
        self.inner = inner

    def reset(self, payload: dict | None = None) -> dict | None:
        return self.inner.reset(payload) if payload else self.inner.reset()

    def step(self, payload: dict) -> dict:
        payload = self.preprocess(payload)
        result = self.inner.step(payload)
        return self.postprocess(payload, result)

    def preprocess(self, payload: dict) -> dict:
        return payload

    def postprocess(self, payload: dict, result: dict) -> dict:
        return result


class DtypeGuardWrapper(PolicyWrapper):
    """Reject non-numeric leaves before they reach the model's jnp.asarray.

    Walks the observation (and task) trees, printing a spec of any leaf
    whose dtype is non-numeric (e.g. strings, objects) and raising
    ``TypeError`` with the offending paths.

    Toggle via the ``enabled`` flag so it can be cheaply disabled in prod.
    """

    _NUMERIC_KINDS: ClassVar[set[str]] = set("biufcBIUFC")  # bool, int, uint, float, complex

    def __init__(self, inner: BasePolicy, *, enabled: bool = True):
        super().__init__(inner)
        self.enabled = enabled

    def preprocess(self, payload: dict) -> dict:
        if not self.enabled:
            return payload
        bad: list[str] = []
        for top_key in ("observation", "task"):
            tree = payload.get(top_key)
            if tree is None:
                continue
            self._check(tree, prefix=top_key, bad=bad)
        if bad:
            bad = "\n  ".join(bad)
            raise TypeError(f"Non-numeric leaves found in payload — cannot convert to JAX array:\n  {bad}")
        return payload

    def _check(self, tree, *, prefix: str, bad: list[str]):
        if isinstance(tree, dict):
            for k, v in tree.items():
                self._check(v, prefix=f"{prefix}.{k}", bad=bad)
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                self._check(v, prefix=f"{prefix}[{i}]", bad=bad)
        else:
            arr = np.asarray(tree)
            if arr.dtype.kind not in self._NUMERIC_KINDS:
                bad.append(f"{prefix}: dtype={arr.dtype}, shape={arr.shape}, value={arr.flat[0]!r}")


class ObsPaddingWrapper(PolicyWrapper):
    """Zero-fill missing observation keys so the payload matches the checkpoint.

    Missing keys get zeros with the same shape/dtype as ``example_obs``,
    and their ``pad_mask_dict`` entry is set to all-False (masked out).
    Extra keys in the payload that aren't in the checkpoint raise ValueError.
    """

    def __init__(self, inner: BasePolicy, example_obs: dict):
        super().__init__(inner)
        self.example_obs = example_obs

    def preprocess(self, payload: dict) -> dict:
        obs = payload.get("observation")
        if obs is None:
            return payload

        expected = set(self.example_obs)
        got = set(obs)
        extra = got - expected - {"pad_mask_dict"}
        if extra:
            raise ValueError(f"Unexpected observation keys not in checkpoint: {extra}")

        missing = expected - got - {"pad_mask_dict", "timestep_pad_mask"}

        obs = {**obs}

        # timestep_pad_mask: ones (all valid) when absent
        if "timestep_pad_mask" not in obs and "timestep_pad_mask" in self.example_obs:
            obs["timestep_pad_mask"] = np.ones_like(np.asarray(self.example_obs["timestep_pad_mask"]))

        # zero-fill genuinely missing observation keys
        for key in missing:
            if key == "pad_mask_dict":
                continue
            obs[key] = np.zeros_like(np.asarray(self.example_obs[key]))

        # pad_mask_dict: ones for keys the caller provided, zeros for missing
        example_pad = self.example_obs.get("pad_mask_dict", {})
        pad = dict(obs.get("pad_mask_dict", {}))
        for pk in example_pad:
            if pk not in pad:
                fill = np.ones_like if pk in got else np.zeros_like
                pad[pk] = fill(np.asarray(example_pad[pk]))
        obs["pad_mask_dict"] = pad
        return {**payload, "observation": obs}


class LegacyDenormWrapper(PolicyWrapper):
    """Unnormalize actions using per-head dataset statistics (legacy heads)."""

    def __init__(self, inner: BasePolicy, stats: dict, head_name: str, dataset_name: str):
        super().__init__(inner)
        raw = stats[dataset_name]["action"]
        raw = drop_fn(raw, lambda x: x is None or x.dtype == "O")
        self.unnorm_stats = {k: jnp.array(v) for k, v in raw[head_name].items()}

    def postprocess(self, payload: dict, result: dict) -> dict:
        actions = result["actions"]
        mean = self.unnorm_stats["mean"]
        std = self.unnorm_stats["std"]
        result["actions"] = np.asarray(actions * std + mean)
        return result


class XFlowDenormWrapper(PolicyWrapper):
    """Denormalize xflow actions per-DOF using ActionBatchDenormalizer."""

    def __init__(self, inner: BasePolicy, stats: dict, dataset_name: str):
        super().__init__(inner)
        self.denorm = ActionBatchDenormalizer(stats)
        self.dataset_name = dataset_name

    def postprocess(self, payload: dict, result: dict) -> dict:
        actions = result["actions"]
        # NOTE: must read from result, not payload. CorePolicy pads dof_ids
        # to max_dofs and puts it in result. After denorm we strip MASK columns
        # and write the active-only dof_ids back to result for downstream
        # wrappers (BodyPartGroupWrapper) that also read from result.
        dof_ids = result.get("dof_ids")
        if dof_ids is None:
            raise ValueError("XFlowDenormWrapper requires 'dof_ids' in result (set by CorePolicy)")
        dof_ids = np.asarray(dof_ids).reshape(-1)
        active = dof_ids != MASK_ID

        if actions.ndim == 1:
            denormed = self.denorm.denormalize_slot(actions, dof_ids, self.dataset_name)
            result["actions"] = denormed[active]
        else:
            denormed = np.stack(
                [self.denorm.denormalize_slot(actions[t], dof_ids, self.dataset_name) for t in range(len(actions))]
            )
            result["actions"] = denormed[:, active]

        result["dof_ids"] = dof_ids[active]  # strip padding for downstream
        return result


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image(s) to (H, W) using lanczos3."""
    x = jnp.asarray(img)
    *lead, h, w, c = x.shape
    x = x.reshape((-1, h, w, c)) if lead else x[None]
    x = jax.image.resize(x, (x.shape[0], size[0], size[1], c), method="lanczos3", antialias=True)
    x = jnp.clip(jnp.rint(x), 0, 255).astype(jnp.uint8)
    x = x.reshape((*lead, size[0], size[1], c)) if lead else x[0]
    return np.asarray(x)


class ImageResizeWrapper(PolicyWrapper):
    """Resize image_* keys in observations to expected (H, W)."""

    def __init__(self, inner: BasePolicy, img_hw: dict[str, tuple[int, int]]):
        super().__init__(inner)
        self.img_hw = img_hw  # e.g. {"image_primary": (224, 224)}

    def preprocess(self, payload: dict) -> dict:
        obs = payload.get("observation")
        if obs is None:
            return payload
        for key, expect_hw in self.img_hw.items():
            if key not in obs:
                continue
            got_hw = tuple(obs[key].shape[-3:-1])
            if got_hw != expect_hw:
                obs = {**obs} if obs is payload.get("observation") else obs
                obs[key] = _resize(obs[key], expect_hw)
        if obs is not payload.get("observation"):
            payload = {**payload, "observation": obs}
        return payload


class ProprioNormWrapper(PolicyWrapper):
    """Normalize proprio_* keys in observations using dataset statistics."""

    def __init__(self, inner: BasePolicy, norm_stats: dict[str, dict]):
        super().__init__(inner)
        self.norm_stats = norm_stats  # e.g. {"single": {"mean": ..., "std": ...}}

    def preprocess(self, payload: dict) -> dict:
        obs = payload.get("observation")
        if obs is None:
            return payload
        changed = False
        for key in list(obs):
            if "proprio" not in key or key == "proprio_bimanual":
                continue
            k = key.replace("proprio_", "")
            n = self.norm_stats.get(k)
            if n is None:
                continue
            if not changed:
                obs = {**obs}
                changed = True
            obs[key] = (obs[key] - n["mean"]) / n["std"]
        if changed:
            payload = {**payload, "observation": obs}
        return payload


class EnsemblerWrapper(PolicyWrapper):
    """Exponentially-weighted action chunk ensembling."""

    def __init__(self, inner: BasePolicy, exp_weight: float, pred_horizon: int, chunk: int):
        super().__init__(inner)
        self.exp_weight = exp_weight
        self.chunk = chunk
        self.history: deque[np.ndarray] = deque(maxlen=pred_horizon)

    def reset(self, payload: dict | None = None) -> dict | None:
        self.history.clear()
        return super().reset(payload)

    def postprocess(self, payload: dict, result: dict) -> dict:
        if "actions" not in result:
            return result
        actions = result["actions"][: self.chunk]
        self.history.append(actions)
        n = len(self.history)
        curr = np.stack([pred[i] for i, pred in zip(range(n - 1, -1, -1), self.history)])
        weights = np.exp(-self.exp_weight * np.arange(n))
        weights = weights / weights.sum()
        result["actions"] = np.sum(weights[:, None] * curr, axis=0)
        return result


def _stack_and_pad(history: deque, num_obs: int) -> dict:
    """Stack observation dicts and add a timestep padding mask."""
    horizon = len(history)
    full_obs = jax.tree.map(lambda *xs: np.stack(xs), *history)
    pad_length = horizon - min(num_obs, horizon)
    mask = np.ones(horizon)
    mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = mask
    return full_obs


class HistoryWrapper(PolicyWrapper):
    """Accumulate observations into a sliding window with padding mask."""

    def __init__(self, inner: BasePolicy, horizon: int):
        super().__init__(inner)
        self.horizon = horizon
        self.history: deque[dict] = deque(maxlen=horizon)
        self.num_obs = 0

    def reset(self, payload: dict | None = None) -> dict | None:
        self.history.clear()
        self.num_obs = 0
        return super().reset(payload)

    def preprocess(self, payload: dict) -> dict:
        obs = payload.get("observation")
        if obs is None:
            return payload
        self.history.append(obs)
        self.num_obs += 1
        if self.horizon > 1:
            obs = _stack_and_pad(self.history, self.num_obs)
        payload = {**payload, "observation": obs}
        return payload
