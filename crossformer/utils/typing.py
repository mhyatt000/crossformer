from typing import Any, Mapping, Sequence, TypeAlias

import jax

PRNGKey: TypeAlias = jax.Array  # keys are just arrays: jax.random.PRNGKey
PyTree: TypeAlias = jax.typing.ArrayLike | Mapping[str, "PyTree"]
Config: TypeAlias = Any | Mapping[str, "Config"]
Params: TypeAlias = Mapping[str, PyTree]
Data: TypeAlias = Mapping[str, PyTree]
Shape: TypeAlias = Sequence[int]
Dtype: TypeAlias = jax.typing.DTypeLike
