from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.xflow import Config, make_model_config


def test_make_model_config_wires_guidance_dims_for_detr():
    cfg = Config(transformer_size="detr")
    model_cfg = make_model_config(cfg, max_h=4, max_a=8, max_w=2, guide_dim=17)["model"]
    head_kwargs = model_cfg["heads"]["xflow"]["kwargs"]

    assert model_cfg["token_embedding_size"] == 512
    assert head_kwargs["guidance_embed_dim"] == model_cfg["token_embedding_size"]
    assert head_kwargs["guidance_input_dim"] == 17


def test_make_model_config_wires_guidance_dims_for_dummy():
    cfg = Config(transformer_size="dummy")
    model_cfg = make_model_config(cfg, max_h=4, max_a=8, max_w=2, guide_dim=9)["model"]
    head_kwargs = model_cfg["heads"]["xflow"]["kwargs"]

    assert model_cfg["token_embedding_size"] == 256
    assert head_kwargs["guidance_embed_dim"] == model_cfg["token_embedding_size"]
    assert head_kwargs["guidance_input_dim"] == 9
