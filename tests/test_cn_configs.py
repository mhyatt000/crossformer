from dataclasses import dataclass

import pytest

from crossformer.cn.base import CN, default
from crossformer.cn.dataset.mix import DataSource, MultiDataSource, TFDS
from crossformer.cn.dataset.types import Head
from crossformer.cn.dataset.dataset import Loader, Dataset
from crossformer.cn.dataset.transform import KeepProb, Modality, Transform
from crossformer.cn.dataset.action import DataPrep
from crossformer.cn.device import Slurm
from crossformer.cn.optim import Cosine, LearningRate, Optimizer
from crossformer.cn.wab import Wandb
from crossformer.cn.eval import Eval
from crossformer.cn import HeadFactory, ModelFactory, ModuleE, Train
from crossformer.cn.rollout import Rollout

import jax


def dummy_standardize_fn():
    return "standardized"


@dataclass
class _Child(CN):
    value: int = 1


@dataclass
class _Parent(CN):
    child: _Child = _Child().field()
    numbers: list[int] = default([1, 2])
    mapping: dict[str, float] = default({"a": 0.1})
    keep: KeepProb = KeepProb.HIGH


def test_cn_base_serialization_and_update():
    parent = _Parent()
    serialized = parent.serialize()
    assert serialized["child"]["value"] == 1
    assert serialized["numbers"] == [1, 2]
    assert serialized["mapping"]["a"] == 0.1
    assert serialized["keep"]["name"] == "HIGH"

    parent.update({"numbers": [3], "child": _Child(value=5)})
    assert parent.numbers == [3]
    assert isinstance(parent.child, _Child)
    assert parent.child.value == 5


def test_data_sources_register_and_flatten():
    ds_name = "unit_test_source"
    source = TFDS(name=ds_name, head=Head.SINGLE)
    assert DataSource.REGISTRY[ds_name] is source
    assert source.flatten() == [(ds_name, 1.0)]

    combo = MultiDataSource(name="unit_test_mix", data=[source], weights=[0.5])
    assert combo.flatten() == [(ds_name, 0.5)]

    with pytest.raises(AssertionError):
        MultiDataSource(name="bad_mix", data=[source], weights=[0.2, 0.8])


def test_loader_batch_size_respects_process_count(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    loader = Loader(global_batch_size=32)
    assert loader.batch_size == 8
    assert loader.global_batch_size == 32


def test_dataset_kwargs_generation():
    dataset = Dataset()
    kwargs_list, prep_list = dataset.kwargs_list(oxe_fns={"xgym_stack_single": dummy_standardize_fn})

    assert len(kwargs_list) == len(prep_list) == 1
    kwargs = kwargs_list[0]
    prep = prep_list[0]

    assert prep.name == "xgym_stack_single"
    assert kwargs["name"] == "xgym_stack_single"
    assert kwargs["language_key"] == "language_instruction"
    assert kwargs["action_normalization_mask"][-1] is False
    assert kwargs["standardize_fn"]["name"] == dummy_standardize_fn.__name__

    more_kwargs = dataset.kwargs()
    assert more_kwargs["batch_size"] == dataset.bs
    assert more_kwargs["traj_read_threads"] == dataset.loader.threads_traj_read


def test_transform_adjusts_keep_image_prob_and_creates_configs():
    transform = Transform(name="custom", task_cond=Modality.IMG, keep_image_prob=0.0)
    assert transform.keep_image_prob == 1.0

    frame = transform.frame.create(["primary"])
    assert frame["resize_size"]["primary"] == transform.frame.resize_size
    assert "image_augment_kwargs" in frame

    traj = transform.traj.create()
    assert traj["goal_relabeling_strategy"] == transform.traj.goal_relabeling_strategy.value
    assert "head_to_dataset" in traj


def test_data_prep_create_uses_specification():
    prep = DataPrep(
        name="xgym_stack_single",
        weight=1.0,
        load_camera_views=["primary"],
        load_proprio=True,
        load_language=True,
    )

    config = prep.create({"xgym_stack_single": dummy_standardize_fn})
    assert config["name"] == "xgym_stack_single"
    assert config["proprio_obs_keys"] == {"primary": "proprio"}
    assert config["language_key"] == "language_instruction"
    assert config["action_normalization_mask"][-1] is False


def test_slurm_parses_environment(monkeypatch):
    slurm = Slurm(num_nodes=2, nodelist="node[1-2]")
    assert slurm.nodelist == ["node1", "node2"]

    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu[1-3]")
    from_env = Slurm().from_env()
    assert from_env.num_nodes == 3
    assert from_env.nodelist == ["gpu1", "gpu2", "gpu3"]


def test_learning_rate_and_optimizer_create():
    lr = LearningRate(decay_steps=100, first=0.0, peak=0.5, last=0.01)
    lr_config = lr.create()
    assert lr_config["init_value"] == 0.0
    assert lr_config["peak_value"] == 0.5
    assert lr_config["end_value"] == 0.01
    assert lr_config["decay_steps"] == 100

    opt = Optimizer(lr=Cosine(decay_steps=50))
    opt_config = opt.create()
    assert opt_config["learning_rate"]["decay_steps"] == 50
    assert "mode" not in opt_config


def test_wandb_mode_changes_with_debug():
    wandb = Wandb()
    assert wandb.mode(debug=False) == "online"
    assert wandb.mode(debug=True) == "disabled"


def test_eval_create_attaches_datasets():
    config = Eval(shuffle_buffer=5, nbatch=2).create(["ds"])
    assert config["datasets"] == ["ds"]
    assert config["shuffle_buffer"] == 5


def test_head_factory_creates_module_spec():
    head = HeadFactory(name="single", module=ModuleE.L1, horizon=3)
    spec = head.create()
    assert spec["kwargs"]["action_horizon"] == 3
    assert spec["kwargs"]["action_dim"] == head.dim.value
    assert "diffusion_steps" not in spec["kwargs"]

    diff = HeadFactory(name="bimanual", module=ModuleE.DIFFUSION, horizon=2, steps=5)
    diff_spec = diff.create()
    assert diff_spec["kwargs"]["diffusion_steps"] == 5


def test_model_factory_builds_selected_heads():
    factory = ModelFactory(heads=["single", "mano"])
    factory.single.horizon = 2
    factory.mano.horizon = 5

    model_config = factory.create()["model"]
    assert set(model_config["heads"].keys()) == {"single", "mano"}
    assert factory.max_horizon() == 5
    assert factory.max_action_dim() == max(
        factory.single.dim.value, factory.mano.dim.value
    )

    spec = factory.spec()["model"]
    assert spec["heads"]["single"] == model_config["heads"]["single"]["module"]

    flat = factory.flatten()
    assert any(key[:2] == ("model", "heads") for key in flat)

    flat_dict = {
        ("model", "heads", "single", "module"): "keep",
        ("model", "heads", "unused", "module"): "drop",
    }
    filtered = factory.delete(flat_dict)
    assert ("model", "heads", "unused", "module") not in filtered
    assert ("model", "heads", "single", "module") in filtered


def test_train_post_init_aligns_components():
    train = Train()
    assert train.data.transform.traj.action_horizon == train.model.max_horizon()
    assert train.data.transform.traj.max_action_dim == train.model.max_action_dim()
    assert train.optimizer.lr.decay_steps == train.steps


def test_rollout_configuration():
    rollout = Rollout(num_envs=3, use=True)
    assert rollout.num_envs == 3
    assert rollout.use is True
