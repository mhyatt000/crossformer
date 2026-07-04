from __future__ import annotations

from typing import Any

from crossformer.cn.model_factory import ModelFactory, Size, XFlow


def _model_cfg(size: Size, guide_dim: int) -> dict[str, Any]:
    return ModelFactory(
        size=size,
        image_keys=(),
        proprio_keys=(),
        xflow=XFlow(use_guidance=True, guidance_input_dim=guide_dim),
    ).create()["model"]


def test_model_factory_wires_guidance_dims_for_detr() -> None:
    model_cfg = _model_cfg(Size.DETR, guide_dim=17)
    head_kwargs = model_cfg["heads"]["action"]["kwargs"]

    assert model_cfg["token_embedding_size"] == 512
    assert head_kwargs["guidance_embed_dim"] == model_cfg["token_embedding_size"]
    assert head_kwargs["guidance_input_dim"] == 17


def test_model_factory_wires_guidance_dims_for_dummy() -> None:
    model_cfg = _model_cfg(Size.DUMMY, guide_dim=9)
    head_kwargs = model_cfg["heads"]["action"]["kwargs"]

    assert model_cfg["token_embedding_size"] == 256
    assert head_kwargs["guidance_embed_dim"] == model_cfg["token_embedding_size"]
    assert head_kwargs["guidance_input_dim"] == 9
