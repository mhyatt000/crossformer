"""RTCPolicy — Real-Time Chunking wrapper for ModelPolicy.

Add this class to base_policy.py after the ModelPolicy class.

Usage:
    policy = RTCPolicy(
        path="~/projects/crossformer/0403_super-night-806/params",
        head_name="xflow",
        d=1, s=3, flow_steps=50, beta=5.0,
    )
    policy.start()
    ...
    action = policy.step(payload)   # returns {"actions": (max_A,), "dof_ids": ...}
    ...
    policy.stop()
"""

from crossformer.model.components.rtc.rtc_algorithm import RTC, guided_inference


class RTCPolicy(ModelPolicy):
    """ModelPolicy extended with Real-Time Chunking (Algorithm 1).

    Wraps the RTC async controller around the XFlowHead flow policy.
    start() / stop() control the background inference thread.
    step() returns the current action from the RTC buffer (non-blocking).

    Args:
        path:        Checkpoint path (same as ModelPolicy).
        step:        Checkpoint step (None = latest).
        head_name:   Head key in module.heads (default "xflow").
        d:           Inference delay in controller timesteps.
        s:           Execution horizon (d <= s <= H-d).
        flow_steps:  Denoising steps for guided_inference.
        beta:        Guidance weight clipping.
        b:           Delay buffer size for RTC.
        guide_keys:  Keys for guide_input (passed to lookup_guide).
        use_guidance: Whether to use guide_input.
    """

    def __init__(
        self,
        path: str,
        *,
        step: int | None = None,
        head_name: str = "xflow",
        d: int = 1,
        s: int = 3,
        flow_steps: int | None = None,
        beta: float = 5.0,
        b: int = 10,
        guide_keys: tuple[str, ...] = ("action.position", "action.orientation"),
        use_guidance: bool = True,
    ):
        # Initialise base ModelPolicy (loads checkpoint, builds _jit_step, etc.)
        super().__init__(
            path,
            step=step,
            head_name=head_name,
            guide_keys=guide_keys,
            use_guidance=use_guidance,
            flow_steps=flow_steps,
        )

        # Bind XFlowHead with params — pi is what guided_inference expects
        self._pi = (
            self.model.module
            .bind({"params": self.params})
            .heads[head_name]
        )

        H     = self._pi.max_horizon
        max_A = self._pi.max_dofs

        self._d = d
        self._s = s
        self._beta = beta
        self._flow_steps = (
            flow_steps
            if flow_steps is not None
            else self.model.module.heads[head_name].flow_steps
        )

        self._rtc = RTC(
            pi=self._pi,
            H=H,
            max_A=max_A,
            s_min=s,
            b=b,
            d_init=d,
            flow_steps=self._flow_steps,
            beta=beta,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background RTC inference thread."""
        self._rtc.start()

    def stop(self) -> None:
        """Stop the background RTC inference thread."""
        self._rtc.stop()

    def reset(self, payload: dict | None = None) -> dict | None:
        """Reset RTC state (reinitialise A_cur to zeros)."""
        H     = self._pi.max_horizon
        max_A = self._pi.max_dofs
        self._rtc._A_cur = np.zeros((H, max_A), dtype=np.float32)
        self._rtc._t     = 0
        return None

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, payload: dict, **kwargs) -> dict:
        """Get current action from RTC buffer and update shared obs.

        Builds the obs dict expected by guided_inference (transformer_outputs,
        dof_ids, chunk_steps, rng, B, W) then calls rtc.get_action().

        Returns:
            {"actions": np.ndarray (max_A,), "dof_ids": np.ndarray (max_A,)}
        """
        self.rng, key = jax.random.split(self.rng)

        obs    = payload["observation"]
        B      = jnp.asarray(obs["timestep_pad_mask"]).shape[0]
        task   = payload.get(
            "task",
            jax.tree.map(lambda x: x[:B], self.model.example_batch["task"]),
        )

        # Run transformer — same as ModelPolicy._jit_step preamble
        obs_jax  = jax.tree_util.tree_map(jnp.array, obs)
        task_jax = jax.tree_util.tree_map(jnp.array, task)
        pad_mask = obs_jax["timestep_pad_mask"]

        transformer_outputs = self.model.run_transformer(
            obs_jax, task_jax, pad_mask, train=False
        )

        dof_ids     = jnp.tile(self._dof_ids_1,     (B, 1))
        chunk_steps = jnp.tile(self._chunk_steps_1, (B, 1))

        guide_input = (
            lookup_guide(payload, self.guide_keys) if self.use_guidance else None
        )

        # obs dict for guided_inference (matches eval_rtc_checkpoint_off._make_obs)
        rtc_obs = {
            "transformer_outputs": transformer_outputs,
            "dof_ids":             dof_ids,
            "chunk_steps":         chunk_steps,
            "slot_pos":            None,
            "guide_input":         guide_input,
            "guidance_mask":       None,
            "train":               False,
            "B":                   B,
            "W":                   1,
            "rng":                 key,
        }

        # GETACTION — returns (max_A,) from A_cur buffer
        action = self._rtc.get_action(rtc_obs)

        return {
            "actions": action,                          # (max_A,)
            "dof_ids": np.asarray(self._dof_ids_1[0]), # (max_A,)
        }
