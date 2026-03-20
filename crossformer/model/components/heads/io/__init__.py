"""Perceiver IO modules (Flax port of DeepMind's Haiku implementation)."""

from __future__ import annotations

from .attention import Attention, CrossAttention, MLP, SelfAttention
from .perceiver import (
    BasicDecoder,
    Perceiver,
    PerceiverEncoder,
    ProjectionDecoder,
)
from .position_encoding import (
    build_linear_positions,
    build_position_encoding,
    FourierPositionEncoding,
    generate_fourier_features,
    TrainablePositionEncoding,
)

__all__ = [
    "MLP",
    "Attention",
    "BasicDecoder",
    "CrossAttention",
    "FourierPositionEncoding",
    "Perceiver",
    "PerceiverEncoder",
    "ProjectionDecoder",
    "SelfAttention",
    "TrainablePositionEncoding",
    "build_linear_positions",
    "build_position_encoding",
    "generate_fourier_features",
]
