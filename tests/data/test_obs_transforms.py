import pytest
import tensorflow as tf

from crossformer.data import dataset as dataset_module
from crossformer.data import obs_transforms


class _DummyTransforms:
    def __init__(self):
        self.resize_calls = []
        self.resize_depth_calls = []
        self.augment_calls = []

    def resize_image(self, image, *, size):
        shape = tuple(tf.shape(image).numpy().tolist())
        self.resize_calls.append((shape, tuple(size)))
        return tf.identity(image)

    def resize_depth_image(self, image, *, size):
        shape = tuple(tf.shape(image).numpy().tolist())
        self.resize_depth_calls.append((shape, tuple(size)))
        return tf.identity(image)

    def augment_image(self, image, *, seed, **kwargs):
        shape = tuple(tf.shape(image).numpy().tolist())
        self.augment_calls.append({
            "shape": shape,
            "seed": tuple(tf.reshape(seed, [-1]).numpy().tolist()),
            "kwargs": kwargs,
        })
        return tf.identity(image)


def _fake_vmap(fn):
    def _index(struct, idx):
        if isinstance(struct, dict):
            return {key: _index(value, idx) for key, value in struct.items()}
        return struct[idx]

    def _stack(elems):
        sample = elems[0]
        if isinstance(sample, dict):
            return {key: _stack([elem[key] for elem in elems]) for key in sample}
        return tf.stack(elems, axis=0)

    def _length(struct):
        if isinstance(struct, dict):
            for value in struct.values():
                length = _length(value)
                if length is not None:
                    return length
            return None
        rank = tf.rank(struct).numpy()
        return int(tf.shape(struct)[0].numpy()) if rank > 0 else None

    def mapped(chunked):
        length = _length(chunked)
        results = []
        for i in range(length):
            single = {key: _index(value, i) for key, value in chunked.items()}
            results.append(fn(single))
        return _stack(results)

    return mapped


@pytest.fixture
def dummy_transforms(monkeypatch):
    dummy = _DummyTransforms()
    transforms_module = obs_transforms.dl.transforms
    monkeypatch.setattr(transforms_module, "resize_image", dummy.resize_image)
    monkeypatch.setattr(
        transforms_module, "resize_depth_image", dummy.resize_depth_image
    )
    monkeypatch.setattr(
        transforms_module, "augment_image", dummy.augment_image
    )
    return dummy


def test_decode_and_resize_decodes_strings_and_preserves_tensors(dummy_transforms):
    rgb_tensor = tf.constant(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=tf.uint8
    )
    rgb_string = tf.io.encode_png(rgb_tensor)

    depth_tensor = tf.constant(
        [[[0], [128]], [[255], [64]]], dtype=tf.uint16
    )
    depth_string = tf.io.encode_png(depth_tensor)

    predecoded_rgb = tf.ones((2, 2, 3), dtype=tf.uint8)
    predecoded_depth = tf.ones((2, 2), dtype=tf.float32)

    obs = {
        "image_rgb": rgb_string,
        "image_static": predecoded_rgb,
        "depth_rgb": depth_string,
        "depth_static": predecoded_depth,
    }

    result = obs_transforms.decode_and_resize(
        obs,
        resize_size={"rgb": (2, 2)},
        depth_resize_size={"rgb": (2, 2)},
    )

    assert result["image_rgb"].dtype == tf.uint8
    assert result["depth_rgb"].dtype == tf.float32
    tf.debugging.assert_equal(result["image_static"], predecoded_rgb)
    tf.debugging.assert_equal(result["depth_static"], predecoded_depth)
    assert len(dummy_transforms.resize_calls) == 1
    assert dummy_transforms.resize_calls[0][1] == (2, 2)
    assert len(dummy_transforms.resize_depth_calls) == 1
    assert dummy_transforms.resize_depth_calls[0][1] == (2, 2)


def test_train_mode_applies_dropout_and_augment(monkeypatch, dummy_transforms):
    monkeypatch.setattr(dataset_module.dl, "vmap", _fake_vmap)

    decode_calls = []

    def fake_decode(obs, *, resize_size, depth_resize_size):
        decode_calls.append(set(obs.keys()))
        return obs

    monkeypatch.setattr(obs_transforms, "decode_and_resize", fake_decode)

    dropout_calls = []

    def fake_dropout(obs, seed, dropout_prob, always_keep_key=None):
        dropout_calls.append({"keys": tuple(sorted(obs.keys())), "prob": dropout_prob})
        return obs

    augment_calls = []

    def fake_augment(obs, seed, augment_kwargs):
        augment_calls.append({"keys": tuple(sorted(obs.keys()))})
        return obs

    monkeypatch.setattr(obs_transforms, "image_dropout", fake_dropout)
    monkeypatch.setattr(obs_transforms, "augment", fake_augment)

    frame = {
        "task": {
            "language_instruction": tf.constant(b"pick"),
            "pad_mask_dict": {"language_instruction": tf.constant(True)},
        },
        "observation": {
            "image_rgb": tf.zeros((2, 2, 2, 3), dtype=tf.uint8),
            "pad_mask_dict": {"image_rgb": tf.ones((2,), dtype=tf.bool)},
        },
    }

    class _Dataset:
        def __init__(self, frames):
            self.frames = frames

        def frame_map(self, fn, _num_parallel_calls):
            self.frames = [fn(frame) for frame in self.frames]
            return self

    dataset = _Dataset([frame])

    dataset_module.apply_frame_transforms(
        dataset,
        train=True,
        image_augment_kwargs={},
        resize_size={},
        depth_resize_size={},
        image_dropout_prob=0.3,
        image_dropout_keep_key="image_rgb",
        num_parallel_calls=1,
    )

    assert decode_calls, "decode_and_resize should be invoked"
    assert any("image_rgb" in call["keys"] for call in dropout_calls)
    assert any("image_rgb" in call["keys"] for call in augment_calls)


def test_eval_mode_skips_dropout_and_augment(monkeypatch, dummy_transforms):
    monkeypatch.setattr(dataset_module.dl, "vmap", _fake_vmap)

    def fake_decode(obs, *, resize_size, depth_resize_size):
        return obs

    monkeypatch.setattr(obs_transforms, "decode_and_resize", fake_decode)

    dropout_calls = []
    augment_calls = []

    def fake_dropout(obs, seed, dropout_prob, always_keep_key=None):
        dropout_calls.append(True)
        return obs

    def fake_augment(obs, seed, augment_kwargs):
        augment_calls.append(True)
        return obs

    monkeypatch.setattr(obs_transforms, "image_dropout", fake_dropout)
    monkeypatch.setattr(obs_transforms, "augment", fake_augment)

    frame = {
        "task": {
            "language_instruction": tf.constant(b"place"),
            "pad_mask_dict": {"language_instruction": tf.constant(True)},
        },
        "observation": {
            "image_rgb": tf.zeros((2, 2, 2, 3), dtype=tf.uint8),
            "pad_mask_dict": {"image_rgb": tf.ones((2,), dtype=tf.bool)},
        },
    }

    class _Dataset:
        def __init__(self, frames):
            self.frames = frames

        def frame_map(self, fn, _num_parallel_calls):
            self.frames = [fn(frame) for frame in self.frames]
            return self

    dataset = _Dataset([frame])

    dataset_module.apply_frame_transforms(
        dataset,
        train=False,
        image_augment_kwargs={},
        resize_size={},
        depth_resize_size={},
        image_dropout_prob=0.3,
        image_dropout_keep_key="image_rgb",
        num_parallel_calls=1,
    )

    assert not dropout_calls
    assert not augment_calls


def test_image_dropout_respects_always_keep_key():
    seed = tf.constant([0, 1], dtype=tf.int32)
    keep_image = tf.ones((2, 2, 3), dtype=tf.float32)
    drop_image = tf.fill((2, 2, 3), 2.0)

    obs = {
        "image_keep": keep_image,
        "image_drop": drop_image,
        "pad_mask_dict": {
            "image_keep": tf.constant(True),
            "image_drop": tf.constant(True),
        },
    }

    result = obs_transforms.image_dropout(
        obs,
        seed=seed,
        dropout_prob=1.0,
        always_keep_key="image_keep",
    )

    assert bool(result["pad_mask_dict"]["image_keep"].numpy())
    tf.debugging.assert_equal(result["image_keep"], keep_image)
    assert not bool(result["pad_mask_dict"]["image_drop"].numpy())
    tf.debugging.assert_equal(result["image_drop"], tf.zeros_like(drop_image))
