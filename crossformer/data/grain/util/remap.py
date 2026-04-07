from __future__ import annotations

import fnmatch


def rekey(tree: dict, inp: list[str], out: list[str]) -> dict:
    assert len(inp) == len(out), f"Input and output key lists must have same length, got {inp} and {out}"
    for i, o in zip(inp, out):
        x = tree[i]  # asserts
        tree[o] = x
        tree.pop(i)
    return tree


def rekeym(tree: dict, match, src, tgt) -> dict:
    """rekey but with matching"""
    srckey = fnmatch.filter(tree.keys(), match)
    tgtkey = [x.replace(src, tgt) for x in srckey]
    assert srckey != tgtkey  # BUG. assertion is patch
    return rekey(tree, inp=srckey, out=tgtkey)


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
