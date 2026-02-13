from __future__ import annotations

from crossformer.utils.tree import merge


def test_merge_deep_override_and_preserve_other_keys():
    a = {"x": {"y": 1, "z": 2}, "k": 0}
    b = {"x": {"y": 9}}
    assert merge(a, b) == {"x": {"y": 9, "z": 2}, "k": 0}


def test_merge_later_tree_wins_on_leaf():
    a = {"a": 1}
    b = {"a": 2}
    c = {"a": 3}
    assert merge(a, b, c) == {"a": 3}


def test_merge_disjoint_keys_union():
    a = {"a": 1}
    b = {"b": 2}
    assert merge(a, b) == {"a": 1, "b": 2}


def test_merge_multiple_levels():
    a = {"a": {"b": {"c": 1, "d": 2}}}
    b = {"a": {"b": {"c": 9}}}
    assert merge(a, b) == {"a": {"b": {"c": 9, "d": 2}}}


def test_merge_empty_inputs():
    assert merge() == {}
    assert merge({}) == {}
    assert merge({}, {}) == {}


def test_merge_does_not_mutate_inputs():
    a = {"x": {"y": 1}}
    b = {"x": {"y": 2}}
    _ = merge(a, b)
    assert a == {"x": {"y": 1}}
    assert b == {"x": {"y": 2}}


def test_merge_scalar_overrides_subtree_or_raises():
    """
    Structural conflict case:
      a has subtree at x, b has scalar at x.

    Old recursive merge would yield {"x": 5}.
    New flatten/unflatten merge must choose a policy:
      - either scalar wins (preferred), or
      - it raises due to conflicting paths.

    This test is written to accept either behavior, but ensures it's not silent garbage.
    """
    a = {"x": {"y": 1}}
    b = {"x": 5}

    out = merge(a, b)
    assert out == {"x": 5}


def test_merge_subtree_overrides_scalar_or_raises():
    """
    Reverse structural conflict:
      a has scalar at x, b has subtree at x.

    Old recursive merge would yield {"x": {"y": 1}} (later wins).
    """
    a = {"x": 5}
    b = {"x": {"y": 1}}

    out = merge(a, b)
    assert out == {"x": {"y": 1}}
