"""RTCPolicy: online rollout wrapper that replaces ModelPolicy.step() with RTC.

Drop-in replacement for ModelPolicy in server_compare.py:

    # before:
    policy = ModelPolicy(str(cfg.path), ...)
    policy = ActionDenormWrapper(policy, ...)
    policy = GrainlikeWrapper(policy, ...)

    # after:
    model_policy = ModelPolicy(str(cfg.path), ...)
    policy = RTCPolicy(model_policy, d=d, s=s)
    policy = ActionDenormWrapper(policy, ...)
    policy = GrainlikeWrapper(policy, ...)

GrainlikeWrapper handles all preprocessing as before.
ActionDenormWrapper handles denormalization as before.
RTCPolicy replaces only the inference step with RTC.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from webpolicy.base_policy import BasePolicy

from crossformer.model.components.heads.dof import pad_chunk_steps, pad_dof_ids
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.base_policy import ModelPolicy
from crossformer.model.components.rtc.rtc_algorithm import RTC
from crossformer.run._wrappers import PolicyWrapper

class RTCPolicy(PolicyWrapper):
    """Online rollout policy wrapping ModelPolicy with RTC async inference.

    ModelPolicy is used only for:
      - checkpoint loading (model, params)
      - dof_ids / chunk_steps construction
      - transformer forward pass (extracted from _jit_step)

    RTC replaces predict_action with guided_inference.

    Args:
        inner:       ModelPolicy with loaded checkpoint.
        d:           inference delay in controller steps.
        s:           execution horizon (d <= s <= H - d).
        H:           prediction horizon. Defaults to inner head's max_horizon.
        flow_steps:  denoising steps. Defaults to inner head's flow_steps.
        beta:        guidance weight clipping. Default 5.0.
        b:           delay buffer size. Default 10.
    """

    def __init__(
        self,
        inner: ModelPolicy,
        d: int,
        s: int,
        H: int | None = None,
        flow_steps: int | None = None,
        beta: float = 5.0,
        b: int = 10,
    ):
        self.inner = inner
        module = inner.model.module
        params = inner.params
        head_name = inner.head_name

        # bound head = pi for guided_inference
        bound_head = module.bind({"params": params}).heads[head_name]
        H = H if H is not None else bound_head.max_horizon
        max_A = bound_head.max_dofs
        flow_steps = flow_steps if flow_steps is not None else bound_head.flow_steps

        self._H = H
        self._max_A = max_A
        self._d = d
        self._s = s

        # separate JIT for transformer — guided_inference calls head directly
        @jax.jit
        def _jit_transformer(obs, task, timestep_pad_mask):
            bound = module.bind({"params": params})
            return bound.crossformer_transformer(obs, task, timestep_pad_mask, train=False)

        self._jit_transformer = _jit_transformer

        # RTC controller
        self.rtc = RTC(
            pi=bound_head,
            H=H,
            max_A=max_A,
            s_min=s,
            b=b,
            d_init=d,
            A_init=np.zeros((H, max_A), dtype=np.float32),
            flow_steps=flow_steps,
            beta=beta,
        )
        self._rng = jax.random.PRNGKey(0)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start RTC background inference thread. Call before rollout loop."""
        self.rtc.start()

    def stop(self):
        """Stop RTC background inference thread. Call after rollout loop."""
        self.rtc.stop()
    
    def unwrapped(self):
        return self.inner.unwrapped() if hasattr(self.inner, 'unwrapped') else self.inner


    def reset(self, payload: dict | None = None) -> dict | None:
        # reset RTC state: A_cur back to zeros, t=0
        self.rtc._t = 0
        self.rtc._A_cur = np.zeros((self._H, self._max_A), dtype=np.float32)
        self.rtc._o_cur = None
        return self.inner.reset(payload)

    def warmup(self, accumulate: bool = False) -> dict:
        return self.step(self.inner.model.example_batch)

    # ------------------------------------------------------------------
    # step — called every Delta_t by the controller / GrainlikeWrapper
    # ------------------------------------------------------------------

    def step(self, payload: dict, **kwargs) -> dict:
        self._rng, key = jax.random.split(self._rng)

        obs = payload["observation"]
        B = int(jnp.asarray(obs["timestep_pad_mask"]).shape[0])
        W = 1  # window dimension in CrossFormer

        task = payload.get(
            "task",
            jax.tree.map(lambda x: x[:B], self.inner.model.example_batch["task"]),
        )

        # dof_ids / chunk_steps — same as ModelPolicy
        dof_ids = jnp.tile(self.inner._dof_ids_1, (B, 1))       # (B, max_dofs)
        chunk_steps = jnp.tile(self.inner._chunk_steps_1, (B, 1))  # (B, max_horizon)

        # transformer forward pass (replicated params, no grad)
        transformer_outputs = self._jit_transformer(obs, task, obs["timestep_pad_mask"])

        # package obs dict for guided_inference
        o = {
            "transformer_outputs": transformer_outputs,
            "dof_ids": dof_ids,
            "chunk_steps": chunk_steps,
            "rng": key,
            "B": B,
            "W": W,
        }

        # RTC: update o_cur, get action from A_cur[t]
        # get_action returns (max_A,) — current timestep's action from A_cur
        action = self.rtc.get_action(o)   # (max_A,)

        # A_cur is (H, max_A) — expose the full chunk so ActionDenormWrapper
        # can process all steps. We tile the current action across H.
        # Shape: (B, W=1, H, max_A) — matches ModelPolicy output.
        action_row = np.array(action, dtype=np.float32)           # (max_A,)
        chunk = np.tile(action_row[None, :], (self._H, 1))        # (H, max_A)
        actions_out = np.tile(chunk[None, None, :, :], (B, 1, 1, 1))  # (B, 1, H, max_A)


        action = self.rtc.get_action(o)
        print("A_cur[0]:", self.rtc._A_cur[0])  # bu satırı ekle

        return {
            "actions": actions_out,
            "dof_ids": jax.device_get(dof_ids),
        }
