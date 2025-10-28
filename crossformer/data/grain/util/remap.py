from __future__ import annotations


def rekey(tree: dict, inp: list[str], out: list[str]) -> dict:
    assert len(inp) == len(out), f"Input and output key lists must have same length, got {inp} and {out}"
    for i, o in zip(inp, out):
        x = tree[i]  # asserts
        tree[o] = x
        tree.pop(i)
    return tree


def _remap_lang(tree: dict, k="language.instruction") -> dict:
    """
    If `k` is in tree["task"] but not in tree,
    copy it to tree["language.instruction"].
    """
    tree = dict(tree)
    task = tree.get("task", {})
    if k in task and k not in tree:
        tree[k] = task[k]
    return tree
