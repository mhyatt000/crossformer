from __future__ import annotations

from enum import StrEnum


class NormalizationType(StrEnum):
    """Defines supported normalization schemes for action and proprio."""

    NORMAL = "normal"
    BOUNDS = "bounds"
