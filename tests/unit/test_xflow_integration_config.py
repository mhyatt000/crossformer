"""Smoke tests for scripts.train.xflow_integration.make_model_config.

Replaces the stale tests/unit/test_xflow_guidance_dim.py — the old
Config-based make_model_config(cfg, max_h, max_a, max_w, guide_dim)
signature no longer exists; the current helper is arg-less.
"""

from __future__ import annotations

from scripts.train.xflow_integration import make_model_config, MAX_A, MAX_H


def test_make_model_config_structure():
    cfg = make_model_config()
    model = cfg["model"]
    assert "observation_tokenizers" in model
    assert "proprio" in model["observation_tokenizers"]
    assert set(model["heads"]) == {"xflow"}
    assert set(model["readouts"]) == {"xflow"}


def test_make_model_config_wires_xflow_head_dims():
    cfg = make_model_config()
    head_spec = cfg["model"]["heads"]["xflow"]
    kwargs = head_spec["kwargs"]
    assert kwargs["max_dofs"] == MAX_A
    assert kwargs["max_horizon"] == MAX_H
    assert kwargs["readout_key"] == "readout_xflow"
