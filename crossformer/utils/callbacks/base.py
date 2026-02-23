from __future__ import annotations

from crossformer.utils.train_utils import TrainState


class Callback:
    def __call__(self, train_state: TrainState, step: int):
        raise NotImplementedError
