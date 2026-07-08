from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Literal

import flax
from rich import print

from crossformer.cn.base import CN
from crossformer.cn.heads import _SINGLE, HeadFactory
from crossformer.model.components.dino_encoder import DinoV3Encoder, MODEL_ID_DEFAULT
from crossformer.model.components.heads.l1 import BundledMSEHead
from crossformer.model.components.heads.pio import PerceiverIOHead
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.vit_encoders import vit_encoder_configs
from crossformer.model.config import (
    ImageTokenizerCfg,
    LowdimTokenizerCfg,
    ModelCfg,
    StackedViewTokenizerCfg,
    TransformerCfg,
    XStateTokenizerCfg,
)
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


def _module_spec(cls: Any, *args: Any, **kwargs: Any) -> ModuleSpec:
    while isinstance(cls, partial):
        args = (*cls.args, *args)
        kwargs = {**(cls.keywords or {}), **kwargs}
        cls = cls.func
    return ModuleSpec(module=cls.__module__, name=cls.__name__, args=args, kwargs=kwargs)


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
    encoder: str = "resnetv2-26-film"
    use_dino: bool = False
    dino_model_id: str = MODEL_ID_DEFAULT
    dino_target_size: tuple[int, int] = (240, 320)
    dino_patch_only: bool = False
    # stacked-multiview path: one tokenizer over observation["image"] (B,T,V,H,W,C)
    # with per-token view ids, instead of one ImageTokenizer per named camera key.
    # tips | dino | any non-FiLM vit_encoder_configs key (e.g. small-stem-16,
    # trained from scratch — pair with --model.vision.no-stacked-freeze).
    stacked: bool = True
    stacked_encoder: str = "tips"
    tips_variant: str = "tips_v2_b14"
    stacked_freeze: bool = True


@dataclass
class XFlow(CN):
    readout_name: str = "action"
    readout_tokens: int = 4  # usually this is related to horizon size, but doesnt have to be
    max_dofs: int = 8
    max_horizon: int = 20
    head_channels: int = 256
    head_depth: int = 2
    head_heads: int = 8
    head_blocks: int = 2
    # factor decoder self-attn over (horizon, dofs) axes: O(H*A^2 + A*H^2)
    # instead of O((H*A)^2). New param tree — not checkpoint-compatible.
    # Ignored by head_type="pio".
    factor_attn: bool = True
    # latent bottleneck size for head_type="pio" (encode-process-decode;
    # process depth = head_blocks * head_depth)
    num_latents: int = 128
    # act_xattn/fuse topology for head_type="pio": compress action tokens into
    # this many latents, then fuse into the context (0 = legacy single-encode).
    act_latents: int = 0
    fuse_layers: int = 2
    # gradient-checkpoint the head's attention blocks (both head types);
    # same param tree, ~10-30% recompute for large batch headroom
    remat: bool = False
    flow_steps: int = 50
    use_guidance: bool = False
    guidance_input_dim: int | None = None
    compress_guidance: bool = False
    num_guidance_latents: int = 4

    def create(self, *, token_dim: int, pio: bool = False) -> ModuleSpec:
        if pio:
            extra = {
                "num_latents": self.num_latents,
                "num_act_latents": self.act_latents,
                "num_fuse_layers": self.fuse_layers,
            }
        else:
            extra = {"factor_attn": self.factor_attn}
        return _module_spec(
            PerceiverIOHead if pio else XFlowHead,
            readout_key=f"readout_{self.readout_name}",
            max_dofs=self.max_dofs,
            max_horizon=self.max_horizon,
            num_query_channels=self.head_channels,
            num_heads=self.head_heads,
            num_blocks=self.head_blocks,
            num_self_attend_layers=self.head_depth,
            flow_steps=self.flow_steps,
            remat=self.remat,
            **extra,
            use_guidance=self.use_guidance,
            guidance_embed_dim=token_dim,
            guidance_input_dim=self.guidance_input_dim,
            compress_guidance=self.compress_guidance,
            num_guidance_latents=self.num_guidance_latents,
        )


@dataclass
class XState(CN):
    use: bool = True
    num_latents: int = 8
    num_channels: int = 256
    num_heads: int = 8
    num_blocks: int = 2
    num_self_attend_layers: int = 1
    widening_factor: int = 4
    dropout_prob: float = 0.0
    input_drop_prob: float = 0.25
    latent_drop_prob: float = 0.25
    skip_missing: bool = True

    def create(self) -> XStateTokenizerCfg:
        return XStateTokenizerCfg(
            name="state",
            num_latents=self.num_latents,
            num_channels=self.num_channels,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            num_self_attend_layers=self.num_self_attend_layers,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            input_drop_prob=self.input_drop_prob,
            latent_drop_prob=self.latent_drop_prob,
            skip_missing=self.skip_missing,
        )


@dataclass
class ModelFactory(CN):
    size: Size = Size.DETR
    # "bela": Perceiver latent-encoder trunk (obs_xattn once per prediction,
    # outside the flow head's ODE loop). Readout latent count = bela_latents.
    # Train scripts must pick BELAModel when trunk="bela".
    trunk: Literal["crossformer", "bela"] = "crossformer"
    bela_latents: int = 128
    window: int = 20
    image_keys: tuple[str, ...] = _DEFAULT_IMAGE_KEYS
    proprio_keys: tuple[str, ...] = _DEFAULT_PROPRIO_KEYS
    vision: Vision = Vision().field()
    xflow: XFlow = XFlow().field()
    state: XState = XState().field()
    # "mse": swap XFlowHead for BundledMSEHead (linear regression baseline;
    # same loss signature, same "action" head key, reuses xflow.max_dofs/max_horizon)
    # "pio": PerceiverIOHead — latent-bottleneck flow head (see xflow.num_latents)
    head_type: Literal["xflow", "mse", "pio"] = "xflow"
    debug: bool = False
    proprio_token_drop_prob: float = 0.0

    @property
    def heads(self) -> list[str]:
        return [self.xflow.readout_name]

    def _obs_tokenizers(
        self,
    ) -> list[ImageTokenizerCfg | LowdimTokenizerCfg | StackedViewTokenizerCfg | XStateTokenizerCfg]:
        toks = []
        if self.vision.stacked:
            toks.append(
                StackedViewTokenizerCfg(
                    name="image",
                    encoder=self.vision.stacked_encoder,
                    tips_variant=self.vision.tips_variant,
                    dino_model_id=self.vision.dino_model_id if self.vision.stacked_encoder == "dino" else None,
                    freeze=self.vision.stacked_freeze,
                )
            )
        elif self.image_keys:
            encoder = self.make_obs_im_encoder()
            toks.extend(self.make_obs_im(key, encoder=encoder) for key in self.image_keys)
        if self.state.use:
            toks.append(self.state.create())
        toks.extend(self.make_obs_proprio(key) for key in self.proprio_keys)
        return toks

    def _head_specs(self, *, token_dim: int) -> dict[str, ModuleSpec]:
        if self.head_type == "mse":
            spec = _module_spec(
                BundledMSEHead,
                readout_key=f"readout_{self.xflow.readout_name}",
                action_horizon=self.xflow.max_horizon,
                action_dim=self.xflow.max_dofs,
            )
            return {self.xflow.readout_name: spec}
        return {self.xflow.readout_name: self.xflow.create(token_dim=token_dim, pio=self.head_type == "pio")}

    def to_model_cfg(self) -> ModelCfg:
        transformer = TransformerCfg.from_size(self.size.value, max_horizon=self.window)
        readout_tokens = self.bela_latents if self.trunk == "bela" else self.xflow.readout_tokens
        return ModelCfg(
            observation_tokenizers=self._obs_tokenizers(),
            readouts={self.xflow.readout_name: readout_tokens},
            heads=self._head_specs(token_dim=transformer.token_embedding_size),
            transformer=transformer,
        )

    def create(self) -> dict[str, Any]:
        # "trunk" is a sidecar key: from_config/load_pretrained only consume
        # config["model"], but it lands in the checkpoint's config.json so
        # inference can dispatch BELAModel vs CrossFormerModel.
        return {"model": self.to_model_cfg().create(), "trunk": self.trunk}

    def build(self) -> Any:
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

    def flatten(self) -> list[tuple[str, ...]]:
        flattened = flax.traverse_util.flatten_dict(self.spec(), keep_empty_nodes=True)
        return list(flattened.keys())

    def delete(self, flat: dict[tuple[str, ...], Any], verbose: bool = False) -> dict[tuple[str, ...], Any]:
        _print = print if verbose else lambda *args, **kwargs: None

        def inside(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
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

    def make_obs_im_encoder(self) -> ModuleSpec:
        if self.vision.use_dino:
            return _module_spec(
                DinoV3Encoder,
                model_id=self.vision.dino_model_id,
                target_size=self.vision.dino_target_size,
                patch_only=self.vision.dino_patch_only,
            )
        assert self.vision.encoder in vit_encoder_configs, f"Unknown vision encoder: {self.vision.encoder}"
        return _module_spec(vit_encoder_configs[self.vision.encoder], use_film=self.vision.use_film)

    def max_horizon(self) -> int:
        return self.window

    def max_action_dim(self) -> int:
        return self.xflow.max_dofs
