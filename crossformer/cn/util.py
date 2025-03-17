from dataclasses import dataclass, field, Field
from rich import print as pprint

from enum import Enum
from functools import partial, wraps
import inspect
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# import hydra
# from hydra.core.config_store import ConfigStore
# from hydra.utils import instantiate
# from omegaconf import DictConfig, MISSING, OmegaConf
# from crossformer.log import logger

# CS = ConfigStore.instance()
dataclass = partial(dataclass, kw_only=True)


def default(data, **kwargs):
    """
    convenience function for field(default_factory=lambda: data)
    if you are looking for hydra defaults list, see:
    https://hydra.cc/docs/tutorials/structured_config/defaults/
    """
    return field(default_factory=lambda: data, **kwargs)


def get_group(cls):
    tree = Path(inspect.getfile(cls))
    idx = tree.parts.index("cn")
    path = tree.parts[idx + 1 : -1]
    group = "/".join(path) if path else None
    return group


def store(cls):
    """wrapper stores the class in ConfigStore under the _group and name"""

    name = cls.__name__.lower()
    group = get_group(cls)

    def wrapper(cls):

        logger.debug(f"store {dict(name=name, group=group)}")
        CS.store(name=name, node=cls, group=group)
        return cls

    return wrapper(cls) if name != "cn" else cls


def isCN(typ):
    print(k, typ())
    if issubclass(typ(), CN):
        pass


from dataclasses import is_dataclass


def anno2default(k, anno) -> Dict[str, Any]:
    """convert type annotation to default value
    - output is a dict where k is field name and v is default value(s)
    - these dicts get stacked into List[Dict[str, Any]] in CNMeta
    """

    if isinstance(anno, Field):  # if CN
        typ = anno.type
        # print(k,typ)
        if is_dataclass(anno):
            return {k: typ.__name__.lower()}  # dataclass CN name

        if get_origin(anno) is list:
            var = get_args(anno)[0]
            if var in [str, int, float, bool]:
                return {k: anno.default_factory()}
            return []

        if get_origin(anno) is dict:
            return {}

        if get_origin(anno) is Union:  # Handle Unions
            raise NotImplementedError("TODO fix")
            options = get_args(anno)

        return anno


"""
if isinstance(v, Field):  # if CN... fix for list and dict
    # if isinstance(v, CN):
    typ = v.type
    # if get_origin(typ) is None: # not a generic List, Dict, etc
    defaults.append({k: typ.__name__.lower()})  # node name not cls name
    # if get_origin(typ) is list:
        # print(list)
    # else:
    # print(v.default_factory())
"""


def expand_defaults(thing):
    if isinstance(thing, Field):
        thing = thing.default_factory()
    if isinstance(thing, list):
        return [expand_defaults(v) for v in thing]
    if isinstance(thing, dict):
        return {k: expand_defaults(v) for k, v in thing.items()}
    if is_dataclass(thing):
        # raise ValueError(f"dataclass should not be in defaults: {type(thing)}")
        return str(thing)
    if isinstance(thing, (str, int, float, bool, type(None))):
        return thing
    raise NotImplemented(f"unknown type {type(thing)}")


class CNMeta(type):
    """uses ConfigStore to store the class
    not to be confused with omagaconf.ConfigNode
    """

    logger.warn("TODO: CNMeta needs fix to ignore list and dict")

    def __new__(cls, name, bases, class_dict):

        # auto wrap with default_factory
        N = (
            lambda k, v: k.startswith("_")
            or isinstance(v, Field)
            or inspect.ismethod(v)
        )
        Y = lambda k, v: isinstance(v, (list, dict)) or is_dataclass(v)
        cond = lambda k, v: Y(k, v) and not N(k, v)

        # these are the attr defined in code
        # # that arent private or methods
        # # and need to be parsed for defaults list
        attr = {k: v for k, v in class_dict.items() if cond(k, v)}
        # logger.debug("attr", **attr)

        # Generate the defaults list by parsing for other CN
        defaults = []
        for k, v in attr.items():
            anno = anno2default(k, v)
            if anno is not None:
                defaults.append({k: anno})
        defaults += ["_self_"]

        attr = {k: default(v) for k, v in attr.items()}
        class_dict.update(attr)

        # Attach the defaults list to the class
        # Add List[Any] annotation to appease dataclass
        if len(defaults) > 1:  # condition is meaningless
            logger.debug(f"{dict(defaults=defaults)}")
            anno = class_dict.get("__annotations__", {})
            anno["defaults"] = List[Any]
            class_dict["__annotations__"] = anno

            class_dict["defaults"] = default(expand_defaults(defaults), repr=False)

        new = super().__new__(cls, name, bases, class_dict)
        # Wrap the class in @dataclass and @store
        new = dataclass(new)
        new = store(new)

        return new


class CN(metaclass=CNMeta):
    """base class to wrap objects with dataclass and store"""

    # defaults: List[Any] = default([])

    def _clean(self):
        logger.debug(f"post init <{self.__class__.__name__}>")
        for k in ["defaults"]:
            if hasattr(self, k):
                delattr(self, k)
        return self

    def __str__(self):
        return f"{self.__class__.__name__.lower()}"

    def asdict(self):
        return OmegaConf.to_container(OmegaConf.create(self))


T = TypeVar("T")  # Generic type variable


def tryex(fn):
    def _tryex(fn):
        try:
            return fn()
        except Exception as e:
            # logger.exception(e, exc_info=e, stack_info=True)
            raise e

    def decorator(*args, **kwargs):
        return _tryex(lambda: fn(*args, **kwargs))

    return decorator


def asdataclass(target: Type[T]):
    """
    Decorator to ensure that a Hydra configuration object is converted to the given type.
    https://github.com/facebookresearch/hydra/issues/981 for info

    Args:
        target (Type[T]): The type to which the configuration should be converted.

    Returns:
        Callable: A decorator function.
    """

    def decorator(func: Callable[[T], None]) -> target:
        @wraps(func)
        def wrapper(cfg: Union[DictConfig, T], *args, **kwargs):
            # Convert cfg to the target type if it's a DictConfig
            # if isinstance(cfg, (DictConfig, dict)):
            # cfg = instantiate(cfg)
            cfg: target = OmegaConf.to_object(cfg)  # ._clean()
            pprint(cfg)

            assert isinstance(
                cfg, target
            ), f"cfg must be of type {target}, got {type(cfg)}"
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
