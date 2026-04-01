from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import flax
from rich import print

from crossformer.cn.base import CN, default
from crossformer.cn.heads import HeadFactory, _SINGLE
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26FILM
from crossformer.utils.spec import ModuleSpec


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
class ModelFactory(CN):
    im: Sequence[str] = default(["primary", "side", "left_wrist"])
    proprio: Sequence[str] = default([_SINGLE])
    heads: Sequence[str] = default([_SINGLE, "k3ds"])
    single: HeadFactory = HeadFactory(name=_SINGLE).field()
    bimanual: HeadFactory = HeadFactory(name="bimanual").field()
    mano: HeadFactory = HeadFactory(name="mano").field()
    k3ds: HeadFactory = HeadFactory(name="k3ds").field()

    size: Size = Size.DETR
    debug: bool = False

    def get_all_heads(self) -> dict[str, HeadFactory]:
        def _make(x):
            try:
                return HeadFactory(**x)
            except Exception:
                return False

        heads = {k: _make(v) for k, v in self.asdict().items() if _make(v)}
        return heads

    def get_selected_heads(self) -> dict[str, HeadFactory]:
        all_heads = self.get_all_heads()
        heads = {k: v for k, v in all_heads.items() if k in self.heads and v.name in self.heads}
        return heads

    def create(self) -> dict[str, Any]:
        token_embedding_size, transformer_kwargs = common_transformer_sizes(self.size.value)

        encoder = self.make_obs_im_encoder()
        im = {k: self.make_obs_im(k, encoder=encoder) for k in self.im}
        prop = {k: self.make_obs_proprio(k) for k in self.proprio}
        tok = im | prop

        heads = self.get_selected_heads()
        assert len(heads) > 0, "No heads selected"
        model = {
            "observation_tokenizers": tok,
            "heads": {k: v.create() for k, v in heads.items()},
            "readouts": {k: v.horizon for k, v in heads.items()},
            "token_embedding_size": token_embedding_size,
            "transformer_kwargs": transformer_kwargs,
        }
        return {"model": model}

    def spec(self) -> dict[str, Any]:
        model = self.create()["model"]
        model = {
            "observation_tokenizers": {k: v["module"] for k, v in model["observation_tokenizers"].items()},
            "heads": {k: v["module"] for k, v in model["heads"].items()},
            "readouts": dict(model["readouts"].items()),
        }
        return {"model": model}

    def flatten(self) -> list[str]:
        flattened = flax.traverse_util.flatten_dict(self.spec(), keep_empty_nodes=True)
        flattened = list(flattened.keys())
        return flattened

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

    def make_obs_proprio(self, key: str):
        return ModuleSpec.create(LowdimObsTokenizer, obs_keys=[f"proprio_{key}"], dropout_rate=0.2)

    def make_obs_im(self, keys: str | Sequence[str], encoder=None):
        if isinstance(keys, str):
            keys = [keys]
        if encoder is None:
            encoder = self.make_obs_im_encoder()

        return ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=[f"image_{k}" for k in keys],
            task_stack_keys=[f"image_{k}" for k in keys],
            task_film_keys=["language_instruction"],
            encoder=encoder,
        )

    def make_obs_im_encoder(self):
        return ModuleSpec.create(ResNet26FILM)

    def max_horizon(self) -> int:
        h = 0
        for head in self.get_selected_heads().values():
            h = max(h, head.horizon)
        return h

    def max_action_dim(self) -> int:
        d = 0
        for head in self.get_selected_heads().values():
            d = max(d, head.dim.value)
        return d
