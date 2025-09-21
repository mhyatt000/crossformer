import numpy as np
import pytest

from crossformer.data.grain import utils


def test_tree_map_applies_function_recursively():
    tree = {"a": 1, "b": {"c": 2, "d": 3}}
    result = utils.tree_map(lambda x: x * 2, tree)
    assert result == {"a": 2, "b": {"c": 4, "d": 6}}
    assert tree == {"a": 1, "b": {"c": 2, "d": 3}}


def test_tree_merge_prefers_rightmost_values():
    base = {"a": 1, "nested": {"left": 5, "shared": {"x": 1}}}
    override = {"nested": {"right": 6, "shared": {"y": 2}}, "extra": 3}
    merged = utils.tree_merge(base, override)
    assert merged == {
        "a": 1,
        "nested": {"left": 5, "right": 6, "shared": {"x": 1, "y": 2}},
        "extra": 3,
    }


def test_clone_structure_preserves_numpy_arrays():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    original = {"x": array, "nested": {"y": [1, 2, 3]}}
    clone = utils.clone_structure(original)
    assert np.array_equal(clone["x"], array)
    assert clone["x"] is not array
    clone["nested"]["y"].append(4)
    assert original["nested"]["y"] == [1, 2, 3]


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=bool)),
        (np.array(["", "foo"], dtype="<U3"), np.array([True, False])),
        (np.array([True, False], dtype=bool), np.array([False, True])),
    ],
)
def test_is_padding_scalar_cases(value, expected):
    mask = utils.is_padding(value)
    assert np.array_equal(mask, expected)


def test_is_padding_nested_dict_combines_masks():
    nested = {"a": np.zeros((2,), dtype=np.float32), "b": np.array(["", ""], dtype="<U1")}
    mask = utils.is_padding(nested)
    assert np.array_equal(mask, np.array([True, True]))


def test_to_padding_matches_dtype():
    assert utils.to_padding(np.array([1, 2], dtype=np.int32)).dtype == np.int32
    assert utils.to_padding(np.array(["a"], dtype="U1")).dtype.kind in {"U", "S"}
    assert utils.to_padding(np.array([True, False], dtype=bool)).dtype == bool


def test_ensure_numpy_and_as_dict():
    assert isinstance(utils.ensure_numpy([1, 2, 3]), np.ndarray)
    mapping = utils.as_dict({"a": 1})
    mapping["b"] = 2
    assert mapping == {"a": 1, "b": 2}
    assert utils.as_dict(None) == {}
