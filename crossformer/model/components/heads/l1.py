"""L1 and MSE action heads."""

from __future__ import annotations

from .base import ContinuousActionHead


class L1ActionHead(ContinuousActionHead):
    """Action head using L1 loss (mean absolute error)."""

    loss_type: str = "l1"


class MSEActionHead(ContinuousActionHead):
    """Action head using MSE loss with MAP pooling."""

    loss_type: str = "mse"
    pool_strategy: str = "use_map"
