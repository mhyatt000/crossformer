"""Policy wrapper that replicates the grain data pipeline transforms.

Transforms raw ArrayRecord data (grain_raw format) into the same structure
produced by GrainDataFactory (grain_full format), reusing actual pipeline
functions wherever possible to maintain parity.
"""

from __future__ import annotations

import fnmatch

import augmax
import jax
import jax.numpy as jnp
import numpy as np
from webpolicy.base_policy import BasePolicy

from crossformer.data.grain import metadata, transforms
from crossformer.data.grain.embody import build_action_norm_mask, embody_transform
from crossformer.data.grain.loader import mix_precompatibility
from crossformer.data.grain.pipelines import add_mask, compatibility, drop_str
from crossformer.data.grain.util.remap import rekey
from crossformer.embody import Embodiment
from crossformer.run.wrappers import PolicyWrapper
from crossformer.utils.jax_utils import str2jax
from crossformer.utils.tree import drop, flat


class GrainlikeWrapper(PolicyWrapper):
    """Wraps a policy to preprocess raw ArrayRecord samples into grain_full format.

    Applies the same transforms as GrainDataFactory.make in order:
    1. init lang defaults
    2. drop RL keys
    3. rekey language
    4. drop strings
    5. restructure trajectory (add task, info, timestep, dataset_name, embodiment)
    6. build action_norm_mask
    7. normalize action & proprio
    8. add_pad_mask_dict
    9. add_head_action_mask
    10. mix_precompatibility (crop, resize 224, rename worm->low etc)
    11. unsqueeze proprio horizon
    12. embody_transform
    13. add_mask
    14. compatibility (rename image/lang/proprio keys)
    15. frame resize (224 -> 64)
    """

    def __init__(
        self,
        inner: BasePolicy,
        *,
        dataset_name: str,
        embodiment: Embodiment,
        max_a: int,
        stats: metadata.DatasetStatistics,
        proprio_keys: list[str],
        skip_norm_keys: tuple[str, ...] = (),
        resize_to: int = 64,
        mask_prob: float = 0.0,
        shuffle_slot: bool = False,
        shard_fn=None,
    ):
        super().__init__(inner)
        self.dataset_name = dataset_name
        self.embodiment = embodiment
        self.max_a = max_a
        self.stats = stats
        self.proprio_keys = proprio_keys
        self.skip_norm_keys = skip_norm_keys
        self.resize_to = resize_to
        self.mask_prob = mask_prob
        self.shuffle_slot = shuffle_slot
        self.shard_fn = shard_fn if shard_fn is not None else (lambda x: x)

    def preprocess_batch(self, payload: dict) -> dict:
        """Unbatch, preprocess each sample, rebatch (no inference)."""
        leaves = jax.tree.leaves(payload)
        batch_size = leaves[0].shape[0] if leaves else 1
        samples = [jax.tree.map(lambda x, i=i: x[i], payload) for i in range(batch_size)]
        preprocessed = [self.preprocess(s) for s in samples]
        return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *preprocessed)

    def step(self, payload: dict, **kwargs) -> dict:
        preprocessed = self.preprocess_batch(payload)
        preprocessed = self.shard_fn(jax.tree.map(jnp.array, preprocessed))
        result = self.inner.step(preprocessed, **kwargs)
        extra = {"info": preprocessed["info"]}
        if "act" in preprocessed:
            extra["act"] = preprocessed["act"]
        return jax.tree.map(np.asarray, result) | jax.tree.map(np.asarray, extra)

    def predict_flow(self, payload: dict) -> np.ndarray:
        return self.inner.predict_flow(self.preprocess_batch(payload))

    def preprocess(self, payload: dict) -> dict:
        x = dict(payload)

        # 1. init lang defaults
        if "language_instruction" not in x:
            x["language_instruction"] = ""
        if "language_embedding" not in x:
            x["language_embedding"] = np.zeros((512,), dtype=np.float32)

        # 2. drop RL keys
        x = drop(x, keys=["discount", "is_terminal", "reward", "is_first", "is_last"])

        # 3. rekey language
        x = rekey(
            x,
            inp=["language_instruction", "language_embedding"],
            out=["language.instruction", "language.embedding"],
        )

        # 4. drop strings
        x = drop_str(x)

        # 5. restructure trajectory
        x = self._restructure(x)

        has_action = "action" in x

        # 6. note embodiment (action keys only)
        if has_action:
            x["embodiment"] = {k: np.array(1, dtype=np.bool_).reshape(-1) for k in x["action"]}

        # 7. build action_norm_mask
        if has_action:
            x["action_norm_mask"] = build_action_norm_mask(x["action"], self.embodiment)

        # 8. normalize action when present; debug server payloads may omit GT actions.
        if has_action:
            x = metadata.normalize_action_and_proprio(
                x,
                metadata=self.stats,
                normalization_type=metadata.NormalizationType.NORMAL,
                proprio_keys=self.proprio_keys,
                skip_norm_keys=self.skip_norm_keys,
            )
        else:
            x = metadata.normalize_proprio(
                x,
                metadata=self.stats,
                normalization_type=metadata.NormalizationType.NORMAL,
                proprio_keys=self.proprio_keys,
                skip_norm_keys=self.skip_norm_keys,
            )

        # 9. add_pad_mask_dict
        x = transforms.add_pad_mask_dict(x)

        # 10. add_head_action_mask (doesn't read action — only needs obs.timestep + dataset name)
        x = transforms.add_head_action_mask(x, name=self.dataset_name)

        # 11. mix_precompatibility (crop, resize 224, rename worm->low, overhead->over)
        x = mix_precompatibility(x)

        # 12. unsqueeze proprio horizon
        x = self._unsqueeze_proprio_horizon(x)

        # 13. embody_transform (skipped when no GT actions — ModelPolicy hardcodes dof_ids)
        if has_action:
            x = embody_transform(
                x,
                embodiment=self.embodiment,
                max_a=self.max_a,
                mask_prob=self.mask_prob,
                shuffle_slot=self.shuffle_slot,
            )

        # 14. add_mask
        x = add_mask(x)

        # 15. convert to numpy float32
        x = jax.tree.map(
            lambda y: np.array(y, dtype=np.float32) if np.asarray(y).dtype == np.float64 else np.array(y), x
        )

        # 16. compatibility (rename image/lang/proprio keys to final form)
        x = compatibility(x)

        # 17. drop top-level dataset_name (grain pipeline drops it via drop_str after batching)
        x.pop("dataset_name", None)

        # 18. frame resize (224 -> resize_to) and add horizon dim
        x = self._resize_images(x)

        # 19. task from checkpoint example_batch (strip leading batch dim — preprocess is per-sample)
        x["task"] = jax.tree.map(lambda v: np.asarray(v[0]), self.unwrapped().model.example_batch["task"])

        return x

    def _restructure(self, x: dict) -> dict:
        """Mirrors builders._restructure_trajectory."""
        info = x.get("info", {})
        sid = np.array(info.get("step", 0)).reshape(-1)
        eid = np.array(info.get("episode", 0)).reshape(-1)
        info["id"] = {"step": sid, "episode": eid}
        x["info"] = info

        x["observation"]["timestep"] = sid

        # build task dict from language key
        lang_key = "language.embedding"
        task = {}
        if lang_key in x:
            task[lang_key] = x[lang_key]
        x["task"] = task

        # add dataset_name
        x["dataset_name"] = str2jax(self.dataset_name)
        x["info"]["dataset_name"] = str2jax(self.dataset_name)

        out = {
            "observation": x["observation"],
            "task": x["task"],
            "dataset_name": x["dataset_name"],
            "info": x["info"],
        }
        if "action" in x:
            out["action"] = x["action"]
        return out

    @staticmethod
    def _unsqueeze_proprio_horizon(x: dict) -> dict:
        """Add leading horizon dim of 1 to proprio keys."""
        if not fnmatch.filter(flat(x).keys(), "observation.proprio.*"):
            return x
        obs = x["observation"]
        if "proprio" in obs:
            obs["proprio"] = jax.tree.map(lambda y: y[None], obs["proprio"])
        return x

    def _resize_images(self, x: dict) -> dict:
        """Resize observation images to self.resize_to and add horizon dim.

        Uses augmax.Resize (same as grain pipeline's get_frame_transform)
        to ensure identical interpolation.
        """
        obs = x.get("observation", {})
        sz = self.resize_to
        resize_fn = augmax.Chain(augmax.Resize(sz))
        rng = jax.random.PRNGKey(0)
        for k in list(obs):
            if not k.startswith("image"):
                continue
            img = obs[k]
            if img.ndim == 3:
                # augmax expects HWC, returns HWC
                resized = resize_fn(rng, jnp.asarray(img))
                obs[k] = np.asarray(resized)[None]  # add horizon dim
            elif img.ndim == 4:
                frames = [np.asarray(resize_fn(rng, jnp.asarray(f))) for f in img]
                obs[k] = np.stack(frames)
        x["observation"] = obs
        return x
