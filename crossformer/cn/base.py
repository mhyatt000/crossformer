from dataclasses import asdict, dataclass, field, Field
import logging
from typing import Dict, Union

from omegaconf import OmegaConf
from rich import print as pprint
import tyro

logger = logging.getLogger(__name__)
logger.info("Importing crossformer.cn")


def default(data, **kwargs):
    """convenience function for field(default_factory=lambda: data)"""
    return field(default_factory=lambda: data, **kwargs)


@dataclass()
class CN:
    name: str = ''

    def field(self):
        """wrap this class as a dataclass field"""
        return field(default_factory=lambda: self)

    def default(self, other):
        """wrap an object as a dataclass field"""
        return field(default_factory=lambda: other)

    def asdict(self):
        return asdict(self)

    def update(self, other: Union[Dict, type]):
        if isinstance(other, CN):
            other = other.asdict()

        for k, v in other.items():
            setattr(self, k, v)
