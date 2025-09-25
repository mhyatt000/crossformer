import importlib
import sys
import types
from enum import Enum, IntEnum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _stub_jax() -> None:
    try:
        importlib.import_module("jax")
    except Exception:
        module = types.ModuleType("jax")
        module.process_count = lambda: 1
        module.device_count = lambda: 1
        _ensure_module("jax", module)


def _stub_tyro() -> None:
    try:
        importlib.import_module("tyro")
    except Exception:
        module = types.ModuleType("tyro")
        module.MISSING = object()
        extras = types.ModuleType("tyro.extras")
        extras.overridable_config_cli = lambda mapping: mapping
        module.extras = extras
        module.cli = lambda cfg=None: cfg
        _ensure_module("tyro", module)
        _ensure_module("tyro.extras", extras)


def _stub_rich() -> None:
    try:
        importlib.import_module("rich")
    except Exception:
        module = types.ModuleType("rich")
        pretty = types.ModuleType("rich.pretty")

        def _noop(*args, **kwargs):
            return None

        module.print = _noop
        pretty.pprint = _noop
        module.pretty = pretty
        _ensure_module("rich", module)
        _ensure_module("rich.pretty", pretty)


def _stub_flax() -> None:
    try:
        importlib.import_module("flax.traverse_util")
    except Exception:
        sys.modules.pop("flax", None)
        sys.modules.pop("flax.traverse_util", None)
        flax_module = types.ModuleType("flax")
        traverse_util = types.ModuleType("flax.traverse_util")

        def flatten_dict(tree, keep_empty_nodes: bool = False):
            result: dict[tuple[str, ...], object] = {}

            def _flatten(prefix, value):
                if isinstance(value, dict):
                    if not value and keep_empty_nodes:
                        result[tuple(prefix)] = value
                    for key, val in value.items():
                        _flatten(prefix + [key], val)
                else:
                    result[tuple(prefix)] = value

            _flatten([], tree)
            return result

        traverse_util.flatten_dict = flatten_dict  # type: ignore[attr-defined]
        flax_module.traverse_util = traverse_util  # type: ignore[attr-defined]
        _ensure_module("flax", flax_module)
        _ensure_module("flax.traverse_util", traverse_util)


def _stub_six() -> None:
    try:
        importlib.import_module("six")
    except Exception:
        module = types.ModuleType("six")
        module.u = lambda s: s
        _ensure_module("six", module)


def _stub_omegaconf() -> None:
    try:
        importlib.import_module("omegaconf")
    except Exception:
        sys.modules.pop("omegaconf", None)
        module = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def create(data):
                return data

            @staticmethod
            def to_container(data):
                return data

            @staticmethod
            def to_object(data):
                return data

        module.OmegaConf = _OmegaConf
        _ensure_module("omegaconf", module)


def _stub_data_utils() -> None:
    try:
        importlib.import_module("crossformer.data.utils.data_utils")
    except Exception:
        sys.modules.pop("crossformer.data.utils.data_utils", None)
        module = types.ModuleType("crossformer.data.utils.data_utils")

        class NormalizationType(Enum):
            NORMAL = "normal"

        module.NormalizationType = NormalizationType
        _ensure_module("crossformer.data.utils.data_utils", module)


def _stub_data_oxe() -> None:
    try:
        importlib.import_module("crossformer.data.oxe")
    except Exception:
        for name in [
            "crossformer.data.oxe",
            "crossformer.data.oxe.oxe_dataset_configs",
            "crossformer.data.oxe.oxe_dataset_mixes",
        ]:
            sys.modules.pop(name, None)

        oxe_configs = types.ModuleType("crossformer.data.oxe.oxe_dataset_configs")

        class ActionDim(IntEnum):
            SINGLE = 7
            BIMANUAL = 14
            MANO = 63
            DMANO_6 = 6
            DMANO_7 = 7
            DMANO_35 = 35
            DMANO_51 = 51
            DMANO_52 = 52
            QUADRUPED = 12

        class ProprioDim(IntEnum):
            POS_EULER = 7
            POS_QUAT = 8
            JOINT = 8
            BIMANUAL = 14
            QUADRUPED = 12
            MANO = 8

        oxe_configs.ActionDim = ActionDim
        oxe_configs.ProprioDim = ProprioDim
        _ensure_module("crossformer.data.oxe.oxe_dataset_configs", oxe_configs)

        oxe_mixes = types.ModuleType("crossformer.data.oxe.oxe_dataset_mixes")
        HEAD_TO_DATASET = {
            "single": ["xgym_stack_single"],
            "bimanual": ["xgym_bimanual"],
            "mano": ["xgym_mano"],
            "quadruped": ["xgym_quadruped"],
        }
        OXE_NAMED_MIXES = {
            "xgym_stack_single": [("xgym_stack_single", 1.0)],
            "xgym_duck_single": [("xgym_duck_single", 1.0)],
            "xgym_lift_single": [("xgym_lift_single", 1.0)],
        }
        oxe_mixes.HEAD_TO_DATASET = HEAD_TO_DATASET
        oxe_mixes.OXE_NAMED_MIXES = OXE_NAMED_MIXES
        _ensure_module("crossformer.data.oxe.oxe_dataset_mixes", oxe_mixes)

        oxe_module = types.ModuleType("crossformer.data.oxe")
        oxe_module.ActionDim = ActionDim
        oxe_module.HEAD_TO_DATASET = HEAD_TO_DATASET
        oxe_module.OXE_NAMED_MIXES = OXE_NAMED_MIXES
        _ensure_module("crossformer.data.oxe", oxe_module)


def _stub_data_dataset() -> None:
    try:
        importlib.import_module("crossformer.data.dataset")
    except Exception:
        sys.modules.pop("crossformer.data.dataset", None)
        module = types.ModuleType("crossformer.data.dataset")

        def make_interleaved_dataset(*args, **kwargs):
            return {"kind": "interleaved", "args": args, "kwargs": kwargs}

        def make_single_dataset(*args, **kwargs):
            return {"kind": "single", "args": args, "kwargs": kwargs}

        module.make_interleaved_dataset = make_interleaved_dataset
        module.make_single_dataset = make_single_dataset
        _ensure_module("crossformer.data.dataset", module)


def _stub_action_heads() -> None:
    try:
        importlib.import_module("crossformer.model.components.action_heads")
    except Exception:
        sys.modules.pop("crossformer.model.components.action_heads", None)
        module = types.ModuleType("crossformer.model.components.action_heads")

        class ActionHead:  # minimal stub for ModuleSpec
            pass

        class L1ActionHead(ActionHead):
            pass

        class DiffusionActionHead(ActionHead):
            pass

        module.ActionHead = ActionHead
        module.L1ActionHead = L1ActionHead
        module.DiffusionActionHead = DiffusionActionHead
        _ensure_module("crossformer.model.components.action_heads", module)


def _stub_tokenizers() -> None:
    try:
        importlib.import_module("crossformer.model.components.tokenizers")
    except Exception:
        sys.modules.pop("crossformer.model.components.tokenizers", None)
        module = types.ModuleType("crossformer.model.components.tokenizers")

        class ImageTokenizer:  # pragma: no cover - stub
            pass

        class LowdimObsTokenizer:  # pragma: no cover - stub
            pass

        module.ImageTokenizer = ImageTokenizer
        module.LowdimObsTokenizer = LowdimObsTokenizer
        _ensure_module("crossformer.model.components.tokenizers", module)


def _stub_vit_encoders() -> None:
    try:
        importlib.import_module("crossformer.model.components.vit_encoders")
    except Exception:
        sys.modules.pop("crossformer.model.components.vit_encoders", None)
        module = types.ModuleType("crossformer.model.components.vit_encoders")

        class ResNet26:  # pragma: no cover - stub
            pass

        class ResNet26FILM:  # pragma: no cover - stub
            pass

        module.ResNet26 = ResNet26
        module.ResNet26FILM = ResNet26FILM
        _ensure_module("crossformer.model.components.vit_encoders", module)


_stub_jax()
_stub_tyro()
_stub_rich()
_stub_flax()
_stub_six()
_stub_omegaconf()
_stub_data_utils()
_stub_data_oxe()
_stub_data_dataset()
_stub_action_heads()
_stub_tokenizers()
_stub_vit_encoders()
