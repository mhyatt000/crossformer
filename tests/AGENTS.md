# THINGS I LIKE

* Parametrize ruthlessly:
One test → many cases (@pytest.mark.parametrize).

* Property tests:
Add at least 1–2 hypothesis properties for your core transforms (shape-safe, monotone, reversible).

* Golden math checks:
Two or three hand-computed cases (identity rotation, 90° axes) catch tons of bugs.

* automated and scalable test tools. makeing debug dataset or debug inputs is great for testing one component of pipeline

# THINGS I DONT LIKE

* testing stubs. they make it easy to write tests that pass, but obscure whether or not the production code truly passes.
no excessive monkey patches. especially, dont mock these: jax, tensorflow, pytorch, dlimp
tests should not behave differently in test vs prod
instead just import the original code and dependencies

* Randomness without seeding
Flaky or—worse—quietly wrong when distribution shifts.
Fix: Seed everything (Python, NumPy, TF/JAX/torch), or use property tests that check invariants over many seeds.

* No negative tests
Code should raise on bad inputs, but you never assert the error type/message.
Fix: Add with pytest.raises(..., match=...) for invalid shapes/dtypes/configs.

* Happy-path only
No tests for empty inputs, NaNs, shape mismatches, bad dtypes, missing keys.
Fix: Parametrize edge cases: [], 1-element, huge tensors, non-contiguous arrays, mixed precision.

* No shape/contract assertions in ML
You check loss value once, not that shapes/dtypes/masks stay consistent across the pipeline.
Fix: Add quick asserts on shapes, masks, monotonicity, conservation properties, etc.
