"""config nodes"""

from __future__ import annotations

import logging

from crossformer.cn.base import CN, default
from crossformer.cn.dataset import Dataset, DataSource, Head, MultiDataSource, transform
from crossformer.cn.eval import Eval
from crossformer.cn.heads import HeadFactory, ModuleE
from crossformer.cn.model_factory import ModelFactory, Size
from crossformer.cn.optim import Optimizer
from crossformer.cn.rollout import Rollout
from crossformer.cn.train import CONFIGS, DATAS, ET, Experiment, MGRS, MODELS, Sweep, TFORMS, TYP, Train, cli, main
from crossformer.cn.wab import Wandb

log = logging.getLogger(__name__)
log.info("Importing crossformer.cn")
