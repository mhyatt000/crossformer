from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from crossformer.data.grain.datasets import _DecodedArrayRecord
from crossformer.utils.callbacks.flow_viz import _DEFAULT_K, load_camera_extrinsics, world_to_uv

shards = sorted(Path("~/.cache/arrayrecords/sweep_mano/0.0.2/to_step").expanduser().glob("*.arrayrecord"))
records = _DecodedArrayRecord(shards)

step = records[0]
xyz = step["observation"]["proprio"]["k3ds"][:, :3]
img = step["observation"]["image"]["low"]

R, t = load_camera_extrinsics("low")

uv = world_to_uv(xyz, R=R, t=t, K=_DEFAULT_K)


print("uv range:", uv.min(axis=0), "->", uv.max(axis=0))
print("xyz:\n", xyz)
print("xyz min/max:", xyz.min(axis=0), xyz.max(axis=0))
print("x/z ratio:", (xyz[:, 0] / xyz[:, 2]).mean())


fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)
ax.scatter(uv[:, 0], uv[:, 1], c="red", s=30, linewidth=1, edgecolors="white")
ax.set_title("Projected k3ds on 'low' camera")
plt.savefig("/tmp/projection_check.png", dpi=150)
