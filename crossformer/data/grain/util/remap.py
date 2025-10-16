from __future__ import annotations


def rekey(tree: dict, inp: list[str], out: list[str]) -> dict:
    assert len(inp) == len(out), f"Input and output key lists must have same length, got {inp} and {out}"
    for i, o in zip(inp, out):
        x = tree[i]  # asserts
        tree[o] = x
        tree.pop(i)
    return tree


def _remap_lang(tree: dict) -> dict:
    tree = dict(tree)
    task = tree.get("task", {})
    if "language.instruction" in task and "language.instruction" not in tree:
        tree["language.instruction"] = task["language.instruction"]
    return tree
