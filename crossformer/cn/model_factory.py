from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import flax
from rich import print

from crossformer.cn.base import CN
from crossformer.cn.heads import _SINGLE, HeadFactory
from crossformer.model.components.dino_encoder import DinoV3Encoder, MODEL_ID_DEFAULT
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.vit_encoders import vit_encoder_configs
from crossformer.model.config import ImageTokenizerCfg, LowdimTokenizerCfg, ModelCfg, TransformerCfg
from crossformer.utils.spec import ModuleSpec

_DEFAULT_IMAGE_KEYS = ("primary", "side", "left_wrist")
_DEFAULT_PROPRIO_KEYS = (_SINGLE,)
_HEAD_TEMPLATES = {
    _SINGLE: HeadFactory(name=_SINGLE),
    "action": HeadFactory(name=_SINGLE),
    "bimanual": HeadFactory(name="bimanual"),
    "mano": HeadFactory(name="mano"),
    "k3ds": HeadFactory(name="k3ds"),
}


class Size(Enum):
    DUMMY = "dummy"
    VANILLA = "vanilla"
    DETR = "detr"
    VIT_T = "vit_t"
    VIT_S = "vit_s"
    VIT_B = "vit_b"
    VIT_L = "vit_l"
    VIT_H = "vit_h"
    VINT = "vint"
    VIT_T_REPEAT = "vit_t_repeat"
    VIT_S_REPEAT = "vit_s_repeat"
    DETR_BIG = "detr_big"


@dataclass
class Vision(CN):
    use_film: bool = True
    encoder: Literal[*vit_encoder_configs] = "resnetv2-26-film"
    use_dino: bool = False
    dino_model_id: str = MODEL_ID_DEFAULT
    dino_target_size: tuple[int, int] = (240, 320)
    dino_patch_only: bool = False


@dataclass
class XFlow(CN):
    readout_name: str = "action"
    readout_tokens: int = 4  # usually this is related to horizon size, but doesnt have to be
    max_dofs: int = 8
    max_horizon: int = 20
    head_channels: int = 256
    head_depth: int = 2
    head_heads: int = 8
    head_blocks: int = 1
    flow_steps: int = 50
    use_guidance: bool = False
    guidance_input_dim: int | None = None
    compress_guidance: bool = False
    num_guidance_latents: int = 4

    def create(self, *, token_dim: int) -> ModuleSpec:
        return ModuleSpec.create(
            XFlowHead,
            readout_key=f"readout_{self.readout_name}",
            max_dofs=self.max_dofs,
            max_horizon=self.max_horizon,
            num_query_channels=self.head_channels,
            num_heads=self.head_heads,
            num_blocks=self.head_blocks,
            num_self_attend_layers=self.head_depth,
            flow_steps=self.flow_steps,
            use_guidance=self.use_guidance,
            guidance_embed_dim=token_dim,
            guidance_input_dim=self.guidance_input_dim,
            compress_guidance=self.compress_guidance,
            num_guidance_latents=self.num_guidance_latents,
        )


@dataclass
class ModelFactory(CN):
    size: Size = Size.DETR
    window: int = 20
    image_keys: tuple[str, ...] = _DEFAULT_IMAGE_KEYS
    proprio_keys: tuple[str, ...] = _DEFAULT_PROPRIO_KEYS
    vision: Vision = Vision().field()
    xflow: XFlow = XFlow().field()
    debug: bool = False
    proprio_token_drop_prob: float = 0.0

    @property
    def heads(self) -> list[str]:
        return [self.xflow.readout_name]

    def _obs_tokenizers(self):
        toks = []
        if self.image_keys:
            encoder = self.make_obs_im_encoder()
            toks.extend(self.make_obs_im(key, encoder=encoder) for key in self.image_keys)
        toks.extend(self.make_obs_proprio(key) for key in self.proprio_keys)
        return toks

    def _head_specs(self, *, token_dim: int) -> dict[str, ModuleSpec]:
        return {self.xflow.readout_name: self.xflow.create(token_dim=token_dim)}

    def to_model_cfg(self) -> ModelCfg:
        transformer = TransformerCfg.from_size(self.size.value, max_horizon=self.window)
        return ModelCfg(
            observation_tokenizers=self._obs_tokenizers(),
            readouts={self.xflow.readout_name: self.xflow.readout_tokens},
            heads=self._head_specs(token_dim=transformer.token_embedding_size),
            transformer=transformer,
        )

    def create(self) -> dict[str, Any]:
        return {"model": self.to_model_cfg().create()}

    def build(self):
        return self.to_model_cfg().build()

    def spec(self) -> dict[str, Any]:
        model = self.create()["model"]
        return {
            "model": {
                "observation_tokenizers": {k: v["module"] for k, v in model["observation_tokenizers"].items()},
                "heads": {k: v["module"] for k, v in model["heads"].items()},
                "readouts": dict(model["readouts"].items()),
            }
        }

    def flatten(self) -> list[str]:
        flattened = flax.traverse_util.flatten_dict(self.spec(), keep_empty_nodes=True)
        return list(flattened.keys())

    def delete(self, flat, verbose=False) -> dict[str, Any]:
        _print = print if verbose else lambda *args, **kwargs: None

        def inside(a: list[str], b: list[str]):
            if len(a) > len(b):
                return False
            return all(_a == _b for _a, _b in zip(a, b[: len(a)]))

        mykeys = self.flatten()
        deletespec = {m[:2] for m in mykeys}

        for c in list(flat.keys()):
            if any(inside(m, c) for m in mykeys):
                continue
            if any(inside(d, c) for d in deletespec):
                _print(f"del: {'.'.join(c)}")
                del flat[c]
        return flat

    def make_obs_proprio(self, key: str) -> LowdimTokenizerCfg:
        return LowdimTokenizerCfg(
            name=key, obs_keys=(f"proprio_{key}",), dropout_rate=0.2, token_drop=self.proprio_token_drop_prob
        )

    def make_obs_im(self, key: str, *, encoder: ModuleSpec) -> ImageTokenizerCfg:
        # DINOv3 takes 3-channel inputs only — disable channel-stacked goal images + FiLM.
        if self.vision.use_dino:
            return ImageTokenizerCfg(
                name=key,
                obs_stack_keys=(f"image_{key}",),
                task_stack_keys=(),
                task_film_keys=(),
                encoder=encoder,
            )
        return ImageTokenizerCfg(
            name=key,
            obs_stack_keys=(f"image_{key}",),
            task_stack_keys=(f"image_{key}",),
            task_film_keys=("language_instruction",) if self.vision.use_film else (),
            encoder=encoder,
        )

    def make_obs_im_encoder(self):
        if self.vision.use_dino:
            return ModuleSpec.create(
                DinoV3Encoder,
                model_id=self.vision.dino_model_id,
                target_size=self.vision.dino_target_size,
                patch_only=self.vision.dino_patch_only,
            )
        assert self.vision.encoder in vit_encoder_configs, f"Unknown vision encoder: {self.vision.encoder}"
        return ModuleSpec.create(vit_encoder_configs[self.vision.encoder], use_film=self.vision.use_film)

    def max_horizon(self) -> int:
        return self.window

    def max_action_dim(self) -> int:
        return self.xflow.max_dofs
