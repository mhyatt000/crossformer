from __future__ import annotations


def _remap_lang(traj: dict) -> dict:
    traj = dict(traj)
    task = traj.get("task", {})
    if "language_instruction" in task and "language_instruction" not in traj:
        traj["language_instruction"] = task["language_instruction"]
    return traj
