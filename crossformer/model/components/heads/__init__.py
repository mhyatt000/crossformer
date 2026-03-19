"""Action heads for predicting actions from transformer outputs."""

from __future__ import annotations

from crossformer.embody import DOF, ids, MASK_ID, VOCAB_SIZE

from .base import ActionHead, ContinuousActionHead
from .diffusion import DiffusionActionHead
from .dof import (
    build_query_mask,
    CHUNK_PAD,
    chunk_range,
    chunk_strided,
    EMBODIMENTS,
    FactoredQueryEncoding,
    pad_chunk_steps,
    pad_dof_ids,
)
from .flow import AdjFlowHead, FlowMatchingActionHead
from .l1 import L1ActionHead, MSEActionHead
from .losses import continuous_loss, masked_mean, sample_tau
from .xflow import XFlowHead

__all__ = [
    "CHUNK_PAD",
    "DOF",
    "EMBODIMENTS",
    "MASK_ID",
    "VOCAB_SIZE",
    "ActionHead",
    "AdjFlowHead",
    "ContinuousActionHead",
    "DiffusionActionHead",
    "FactoredQueryEncoding",
    "FlowMatchingActionHead",
    "L1ActionHead",
    "MSEActionHead",
    "XFlowHead",
    "build_query_mask",
    "chunk_range",
    "chunk_strided",
    "continuous_loss",
    "ids",
    "masked_mean",
    "pad_chunk_steps",
    "pad_dof_ids",
    "sample_tau",
]
