from dataclasses import asdict, dataclass, field, Field
from enum import Enum
import logging
from typing import Dict, Union, Any
from dataclasses import is_dataclass, fields

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
    name: str = ""

    def field(self):
        """wrap this class as a dataclass field"""
        return field(default_factory=lambda: self)

    def default(self, other):
        """wrap an object as a dataclass field"""
        return field(default_factory=lambda: other)

    def asdict(self):
        return asdict(self)

    def serialize(self):

        def ser(obj: Any) -> Any:
            if is_dataclass(obj):
                return {f.name: ser(getattr(obj, f.name)) for f in fields(obj)}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(ser(v) for v in obj)
            elif isinstance(obj, dict):
                return {ser(k): ser(v) for k, v in obj.items()}
            elif isinstance(obj, Enum):
                return {
                    "class": obj.__class__.__name__,
                    "name": obj.name,
                    "value": obj.value,
                }

            else:
                return obj

        return ser(self)

    def update(self, other: Union[Dict, type]):
        if isinstance(other, CN):
            other = other.asdict()

        for k, v in other.items():
            setattr(self, k, v)
