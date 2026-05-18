from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from einops import rearrange
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from rich import print
import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

from crossformer.embody import KP2D_NAMES
from crossformer.utils.spec import spec
from scripts.train.dream import (
    _count_params,
    _denormalize_kp2d,
    _image_to_float,
    extract_keypoints,
    final_pred_heatmaps,
    make_model,
    net_out_size,
    prepare_pred_heatmaps,
    prepare_pred_mask,
)


@dataclass
class DreamModelConfig:
    """DREAM model shape/config fields used for random init."""

    seed: int = 0
    net_in_size: tuple[int, int] = (200, 200)
    image_c: int = 3
    num_keypoints: int = 0  # 0 = xArm DREAM landmarks
    encoder: Literal["vgg", "tips"] = "tips"
    variant: Literal["quarter", "half", "full"] = "full"
    decoder: Literal["auto", "upsample", "deconv", "dpt"] = "dpt"
    tips_variant: str = "tips_v2_b14"
    tips_checkpoint: Path | None = None
    tips_trainable: bool = False
    deconv_decoder: bool | None = None
    full_output: bool | None = None
    skip_connections: bool = False
    n_stages: int = 1
    internalize_spatial_softmax: bool = False
    learned_beta: bool = True
    initial_beta: float = 1.0


@dataclass
class ReturnConfig:
    heatmaps: bool = False  # heatmaps are large; keep off for interactive serving
    mask: bool = True  # return mask when the decoder produces one


@dataclass
class Config:
    """Serve a DREAM model."""

    host: str = "0.0.0.0"
    port: int = 8002
    path: Path | None = None  # params checkpoint dir; None keeps random init
    step: int | None = None  # checkpoint step; None loads latest
    warmup: bool = True
    ret: ReturnConfig = field(default_factory=ReturnConfig)
    dream: DreamModelConfig = field(default_factory=DreamModelConfig)


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if tuple(image.shape[1:3]) == (h, w):
        return image

    x = jnp.asarray(image)
    x = jax.image.resize(x, (x.shape[0], h, w, x.shape[-1]), method="lanczos3", antialias=True)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        x = jnp.clip(jnp.rint(x), info.min, info.max).astype(image.dtype)
    return np.asarray(x)


def _params_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "params":
        return path
    if (path / "params").exists():
        return path / "params"
    return path


def _load_params(path: Path, target_params, step: int | None):
    path = _params_path(path)
    mngr = ocp.CheckpointManager(path)
    step = step if step is not None else mngr.latest_step()
    if step is None:
        raise ValueError(f"no checkpoints found under {path}")
    abstract = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
        target_params,
    )
    print(f"loading DREAM params: path={path} step={step}")
    return mngr.restore(step, args=ocp.args.StandardRestore(abstract))


class DreamPolicy(BasePolicy):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_cfg = cfg.dream
        self.net_h, self.net_w = self.model_cfg.net_in_size
        self.out_h, self.out_w = net_out_size(self.model_cfg)
        self.num_keypoints = self.model_cfg.num_keypoints or len(KP2D_NAMES)
        self.model = make_model(self.model_cfg, self.num_keypoints)

        rng = jax.random.PRNGKey(self.model_cfg.seed)
        image = jnp.zeros((1, self.net_h, self.net_w, self.model_cfg.image_c), dtype=jnp.float32)
        self.params = self.model.init(rng, image)["params"]
        if self.cfg.path is not None:
            self.params = _load_params(self.cfg.path, self.params, self.cfg.step)

        @jax.jit
        def _predict(params, image):
            model_out, _ = self.model.apply({"params": params}, _image_to_float(image))
            heatmaps = final_pred_heatmaps(prepare_pred_heatmaps(model_out, self.out_h, self.out_w))
            uv, conf = extract_keypoints(heatmaps)
            uv_norm = uv / jnp.array([self.out_w, self.out_h], dtype=jnp.float32)
            out = {
                "keypoints": _denormalize_kp2d(uv_norm, self.net_h, self.net_w),
                "keypoints_norm": uv_norm,
                "confidence": conf,
            }
            mask = prepare_pred_mask(model_out, self.out_h, self.out_w)
            if self.cfg.ret.mask and mask is not None:
                out["mask"] = mask
            if self.cfg.ret.heatmaps:
                out["heatmaps"] = heatmaps
            return out

        self._predict = _predict
        source = "checkpoint" if self.cfg.path is not None else "random"
        print(f"loaded {source} DREAM params: {_count_params(self.params):,}")
        print(
            f"input_size={(self.net_h, self.net_w)} output_size={(self.out_h, self.out_w)} keypoints={self.num_keypoints}"
        )
        if self.cfg.warmup:
            self.warmup()

    def warmup(self):
        image = np.zeros((1, self.net_h, self.net_w, self.model_cfg.image_c), dtype=np.uint8)
        print("warmup input spec")
        print(spec({"image": image}))
        out = self.step({"image": image})
        print("warmup output spec")
        print(spec(out))

    def reset(self, payload: dict | None = None) -> dict:
        return {"reset": True}

    def step(self, payload: dict) -> dict:
        if payload.get("reset", False):
            return self.reset(payload)
        image = rearrange(np.asarray(payload["image"]), "... h w c -> (...) h w c")
        image = _resize_image(image, self.model_cfg.net_in_size)
        out = self._predict(self.params, jnp.asarray(image))
        return jax.device_get(out)


def main(cfg: Config):
    policy = DreamPolicy(cfg)
    server = Server(policy, host=cfg.host, port=cfg.port, metadata=None)
    print("serving DREAM on", cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
