"""RTCPolicy: online rollout wrapper that replaces ModelPolicy.step() with RTC.

Drop-in replacement for ModelPolicy in server_compare.py:

    # before:
    policy = ModelPolicy(str(cfg.path), ...)
    policy = ActionDenormWrapper(policy, ...)
    policy = GrainlikeWrapper(policy, ...)

    # after:
    model_policy = ModelPolicy(str(cfg.path), ...)
    policy = RTCPolicy(model_policy, d=d, s=s, use_guidance=False)
    policy = ActionDenormWrapper(policy, ...)
    policy = GrainlikeWrapper(policy, ...)

Transformer forward pass ve guided_inference ikisi de background thread'de
calisir. Ana thread sadece get_action ile A_cur'dan action okur.
"""

from __future__ import annotations

import threading
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from crossformer.run._wrappers import PolicyWrapper
from crossformer.run.base_policy import ModelPolicy
from crossformer.model.components.rtc.rtc_algorithm import guided_inference


class RTCPolicy(PolicyWrapper):
    """Online rollout policy wrapping ModelPolicy with RTC async inference.

    Transformer forward pass ve guided_inference background thread'de calisir.
    Ana thread her Delta_t'de step() cagirir, A_cur'dan action okur.

    Args:
        inner:        ModelPolicy with loaded checkpoint.
        d:            inference delay in controller steps.
        s:            execution horizon (d <= s <= H - d).
        H:            prediction horizon. Defaults to inner head's max_horizon.
        flow_steps:   denoising steps. Defaults to inner head's flow_steps.
        beta:         guidance weight clipping. Default 5.0.
        b:            delay buffer size. Default 10.
        use_guidance: if False, plain Euler (no vjp, half the memory). Default False.
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
        use_guidance: bool = False,
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
        self._pi = bound_head
        self._flow_steps = flow_steps
        self._beta = beta
        self._use_guidance = use_guidance

        # transformer JIT -- background thread'de calisir
        @jax.jit
        def _jit_transformer(obs, task, timestep_pad_mask):
            bound = module.bind({"params": params})
            return bound.crossformer_transformer(obs, task, timestep_pad_mask, train=False)

        self._jit_transformer = _jit_transformer

        # RTC shared state
        self._cond = threading.Condition()
        self._t: int = 0
        self._A_cur: np.ndarray = np.zeros((H, max_A), dtype=np.float32)
        self._o_cur: dict | None = None
        self._Q: deque[int] = deque([d], maxlen=b)
        self._s_min: int = s
        self._running: bool = False
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)

        self._rng = jax.random.PRNGKey(0)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Background inference thread'i baslat. Rollout'tan once cagir."""
        self._running = True
        self._thread.start()

    def stop(self):
        """Background thread'i durdur. Rollout'tan sonra cagir."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5.0)

    def reset(self, payload: dict | None = None) -> dict | None:
        with self._cond:
            self._t = 0
            self._A_cur = np.zeros((self._H, self._max_A), dtype=np.float32)
            self._o_cur = None
        return self.inner.reset(payload)

    def warmup(self, accumulate: bool = False) -> dict:
        return self.step(self.inner.model.example_batch)

    # ------------------------------------------------------------------
    # step -- ana thread, her Delta_t'de cagrilir
    # ------------------------------------------------------------------

    def step(self, payload: dict, **kwargs) -> dict:
        self._rng, key = jax.random.split(self._rng)

        obs = payload["observation"]
        B = int(jnp.asarray(obs["timestep_pad_mask"]).shape[0])

        dof_ids = jnp.tile(self.inner._dof_ids_1, (B, 1))
        chunk_steps = jnp.tile(self.inner._chunk_steps_1, (B, 1))

        # ham payload'u background thread icin sakla
        # transformer burada degil, background thread'de calisacak
        o_raw = {
            "payload": payload,
            "dof_ids": dof_ids,
            "chunk_steps": chunk_steps,
            "rng": key,
            "B": B,
            "W": 1,
        }

        with self._cond:
            self._t += 1
            self._o_cur = o_raw
            self._cond.notify_all()
            idx = min(self._t - 1, self._H - 1)
            action = np.array(self._A_cur[idx])  # (max_A,)

        # shape: (B, 1, H, max_A)
        action_row = np.array(action, dtype=np.float32)
        chunk = np.tile(action_row[None, :], (self._H, 1))
        actions_out = np.tile(chunk[None, None, :, :], (B, 1, 1, 1))

        return {
            "actions": actions_out,
            "dof_ids": np.array(dof_ids),
        }

    # ------------------------------------------------------------------
    # background inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self):
        """Transformer + guided_inference -- background thread."""
        import traceback
        try:
            with self._cond:
                while self._running:
                    self._cond.wait_for(
                        lambda: self._t >= self._s_min or not self._running
                    )
                    if not self._running:
                        break

                    s = self._t
                    A_prev = self._A_cur[s:].copy()
                    o_raw = self._o_cur
                    d = max(self._Q)

                    self._cond.release()
                    try:
                        A_new = self._run_inference(o_raw, A_prev, d, s)
                    finally:
                        self._cond.acquire()

                    self._A_cur = np.array(A_new, dtype=np.float32)
                    self._t = self._t - s
                    # self._Q.append(self._t)
                    self._Q.append(min(self._t, self._s))
        except Exception:
            traceback.print_exc()

    def _run_inference(self, o_raw: dict, A_prev: np.ndarray, d: int, s: int) -> np.ndarray:
        """Transformer + guided_inference -- kilitsiz calisir."""
        payload = o_raw["payload"]
        dof_ids = o_raw["dof_ids"]
        chunk_steps = o_raw["chunk_steps"]
        rng = o_raw["rng"]
        B = o_raw["B"]
        W = o_raw["W"]

        obs = payload["observation"]
        task = payload.get(
            "task",
            jax.tree.map(lambda x: x[:B], self.inner.model.example_batch["task"]),
        )

        # transformer background thread'de calisir
        transformer_outputs = self._jit_transformer(obs, task, obs["timestep_pad_mask"])

        o = {
            "transformer_outputs": transformer_outputs,
            "dof_ids": dof_ids,
            "chunk_steps": chunk_steps,
            "rng": rng,
            "B": B,
            "W": W,
        }

        A_new = guided_inference(
            pi=self._pi,
            obs=o,
            A_prev=jnp.array(A_prev),
            d=d,
            s=s,
            flow_steps=self._flow_steps,
            beta=self._beta,
            use_guidance=self._use_guidance,
        )
        # (B, W, H, max_A) -> (H, max_A)
        return np.array(A_new[0, 0])
