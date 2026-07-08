"""Eval loop: schedule ``EvalCallback``s over a dedicated eval loader.

The loop owns batch fetching, the shared ``EvalContext`` (cached predictions),
and a single wandb log per tick; callbacks own everything else. See
``crossformer.utils.callbacks.base`` for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from crossformer.utils.callbacks.base import EvalCallback, EvalContext, flatten_obs
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer


@dataclass
class EvalLoop:
    """Run due eval callbacks against a shared, lazily-predicting context."""

    loader: Iterable[Mapping[str, Any]]
    callbacks: Sequence[EvalCallback]
    denorm: ActionBatchDenormalizer
    obs_keys: tuple[str, ...] = ()
    pred_rng: Any = None
    stats: Any = None  # raw DatasetStatistics (has .unnormalize)
    use_guidance: bool = False
    guide_keys: tuple[str, ...] = ()
    wandb_log: Callable[..., None] = lambda *a, **k: None
    _it: Iterator[Mapping[str, Any]] | None = field(default=None, init=False, repr=False)

    def __call__(self, model: Any, params: Any, step: int, *, is_last: bool = False) -> None:
        due = [cb for cb in self.callbacks if cb.every > 0 and (step % cb.every == 0 or is_last)]
        if not due:
            return
        ctx = self._make_ctx(model, params, step)
        metrics = {f"{cb.name}/{k}": v for cb in due for k, v in cb(ctx).items()}
        self.wandb_log(metrics, step=step)

    def _make_ctx(self, model: Any, params: Any, step: int) -> EvalContext:
        return EvalContext(
            model=model,
            params=params,
            rng=self.pred_rng,
            step=step,
            batch=self._next_batch(),
            denorm=self.denorm,
            stats=self.stats,
            use_guidance=self.use_guidance,
            guide_keys=self.guide_keys,
            next_ctx=lambda: self._make_ctx(model, params, step),
        )

    def _next_batch(self) -> Mapping[str, Any]:
        if self._it is None:
            self._it = iter(self.loader)
        try:
            batch = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            batch = next(self._it)
        batch = dict(batch)
        batch["observation"] = flatten_obs(
            batch["observation"],
            self.obs_keys,
            view_mask=batch.get("mask", {}).get("view"),
            state=batch.get("state"),
            mask=batch.get("mask"),
        )
        return batch
