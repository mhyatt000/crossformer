from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import flax
from rich import print

from crossformer.cn.base import CN, default
from crossformer.cn.heads import _SINGLE, HeadFactory
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
    encoder: str = "resnetv2-26-film"


@dataclass
class ModelFactory(CN):
    size: Size = Size.DETR
    window: int = 20
    image_keys: tuple[str, ...] = _DEFAULT_IMAGE_KEYS
    proprio_keys: tuple[str, ...] = _DEFAULT_PROPRIO_KEYS
    vision: Vision = Vision().field()
    readouts: dict[str, int] = default({_SINGLE: 20, "k3ds": 20})
    debug: bool = False

    @property
    def heads(self) -> list[str]:
        return list(self.readouts.keys())

    def _obs_tokenizers(self):
        toks = []
        if self.image_keys:
            encoder = self.make_obs_im_encoder()
            toks.extend(self.make_obs_im(key, encoder=encoder) for key in self.image_keys)
        toks.extend(self.make_obs_proprio(key) for key in self.proprio_keys)
        return toks

    def _head_specs(self) -> dict[str, ModuleSpec]:
        specs = {}
        for name, readout_tokens in self.readouts.items():
            assert name in _HEAD_TEMPLATES, f"Unknown head/readout: {name}"
            head = HeadFactory(**_HEAD_TEMPLATES[name].asdict())
            head.horizon = self.window
            specs[name] = head.create()
            assert readout_tokens > 0, f"readouts[{name!r}] must be positive"
        return specs

    def to_model_cfg(self) -> ModelCfg:
        return ModelCfg(
            observation_tokenizers=self._obs_tokenizers(),
            readouts=dict(self.readouts),
            heads=self._head_specs(),
            transformer=TransformerCfg.from_size(self.size.value, max_horizon=self.window),
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
        return LowdimTokenizerCfg(name=key, obs_keys=(f"proprio_{key}",), dropout_rate=0.2)

    def make_obs_im(self, key: str, *, encoder: ModuleSpec) -> ImageTokenizerCfg:
        return ImageTokenizerCfg(
            name=key,
            obs_stack_keys=(f"image_{key}",),
            task_stack_keys=(f"image_{key}",),
            task_film_keys=("language_instruction",) if self.vision.use_film else (),
            encoder=encoder,
        )

    def make_obs_im_encoder(self):
        assert self.vision.encoder in vit_encoder_configs, f"Unknown vision encoder: {self.vision.encoder}"
        return ModuleSpec.create(vit_encoder_configs[self.vision.encoder], use_film=self.vision.use_film)

    def max_horizon(self) -> int:
        return self.window

    def max_action_dim(self) -> int:
        dims = [_HEAD_TEMPLATES[name].dim.value for name in self.readouts]
        return max(dims)
