from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
import grain
from mcap.reader import make_reader, McapReader
from mcap.records import Channel, Schema
from mcap_protobuf.decoder import DecoderFactory
import numpy as np
from tqdm import tqdm


def _stack(xs: list[Any]) -> Any:
    if not xs:
        return np.asarray([])
    if all(isinstance(x, dict) and x.keys() == xs[0].keys() for x in xs):
        return {key: _stack([x[key] for x in xs]) for key in xs[0]}
    if all(isinstance(x, np.ndarray) and x.shape == xs[0].shape for x in xs):
        return np.stack(xs)
    try:
        return np.asarray(xs)
    except ValueError:
        out = np.empty(len(xs), dtype=object)
        out[:] = xs
        return out


def _protobuf_value(field: FieldDescriptor, value: Any) -> Any:
    if field.message_type is not None:
        if field.message_type.GetOptions().map_entry:
            value_field = field.message_type.fields_by_name["value"]
            return {key: _protobuf_value(value_field, value[key]) for key in value}
        if field.is_repeated:
            return _stack([_protobuf(x) for x in value])
        return _protobuf(value)
    if field.type == FieldDescriptor.TYPE_BYTES:
        if field.is_repeated:
            return _stack([np.frombuffer(x, dtype=np.uint8) for x in value])
        return np.frombuffer(value, dtype=np.uint8)
    if field.is_repeated:
        return np.asarray(value)
    return value


def _protobuf(msg: Message) -> dict[str, Any]:
    return {field.name: _protobuf_value(field, getattr(msg, field.name)) for field in msg.DESCRIPTOR.fields}


def decode_message(
    decoder_factory: DecoderFactory,
    schema: Schema | None,
    channel: Channel,
    data: bytes,
) -> Any:
    if channel.message_encoding == "json":
        return json.loads(data)
    decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
    if decoder is not None:
        return _protobuf(decoder(data))
    return np.frombuffer(data, dtype=np.uint8)


def _message_progress(reader: McapReader) -> tuple[int, int] | None:
    summary = reader.get_summary()
    if summary is None or summary.statistics is None:
        return None
    counts = summary.statistics.channel_message_counts
    if not counts:
        return None
    length = int(median(counts.values()))
    channel_id = min(counts, key=lambda key: abs(counts[key] - length))
    return length, channel_id


def read_mcap(
    idx: int,
    path: Path,
    *,
    max_messages_per_topic: int | None = None,
    progress_offset: int | None = None,
    progress_slots: int = 1,
) -> dict[str, Any]:
    topics: dict[str, dict[str, Any]] = {}
    decoder_factory = DecoderFactory()
    with path.open("rb") as stream:
        reader = make_reader(stream)
        progress = _message_progress(reader) if progress_offset is not None else None
        pbar = tqdm(
            total=progress[0] if progress else None,
            desc=path.name,
            position=progress_offset + idx % progress_slots if progress_offset is not None else None,
            leave=False,
            unit="step",
            disable=progress is None,
        )
        try:
            for schema, channel, message in reader.iter_messages():
                if progress is not None and channel.id == progress[1]:
                    pbar.update()
                topic = topics.setdefault(
                    channel.topic,
                    {
                        "schema": schema.name if schema else "",
                        "schema_encoding": schema.encoding if schema else "",
                        "message_encoding": channel.message_encoding,
                        "log_time": [],
                        "publish_time": [],
                        "sequence": [],
                        "data": [],
                    },
                )
                if max_messages_per_topic is not None and len(topic["data"]) >= max_messages_per_topic:
                    continue
                topic["log_time"].append(message.log_time)
                topic["publish_time"].append(message.publish_time)
                topic["sequence"].append(message.sequence)
                topic["data"].append(decode_message(decoder_factory, schema, channel, message.data))
        finally:
            pbar.close()

    return {
        "info": {
            "episode": np.asarray(idx, dtype=np.int32),
            "path": str(path),
        },
        "topics": {
            name: {
                **{
                    key: value
                    for key, value in topic.items()
                    if key not in {"log_time", "publish_time", "sequence", "data"}
                },
                "log_time": np.asarray(topic["log_time"], dtype=np.uint64),
                "publish_time": np.asarray(topic["publish_time"], dtype=np.uint64),
                "sequence": np.asarray(topic["sequence"], dtype=np.uint32),
                "data": _stack(topic["data"]),
            }
            for name, topic in topics.items()
        },
    }


def _constant(x: Any, name: str) -> Any:
    x = np.asarray(x)
    if not len(x) or not np.all(x == x[0]):
        raise ValueError(f"RawImage {name} must be constant")
    return x[0].item()


def decode_raw_image(value: dict) -> np.ndarray:
    """Decode a ``foxglove.RawImage`` topic into an ``[T, H, W, 3]`` RGB array."""
    import cv2

    data = value["data"]
    h = _constant(data["height"], "height")
    w = _constant(data["width"], "width")
    step = _constant(data["step"], "step")
    if step % w:
        raise ValueError(f"RawImage step={step} is not divisible by width={w}")
    c = step // w
    images = data["data"]
    if images.shape[1] != h * step:
        raise ValueError(f"RawImage data width={images.shape[1]} does not match height x step={h * step}")
    images = images.reshape(len(images), h, w, c)
    if c == 2:  # YUYV422 -> RGB
        images = np.stack([cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2) for image in images])
    return images


def camera_name(topic: str) -> str:
    """Turn an image topic path into a short, stable camera key.

    ``/cam/side/image_raw`` -> ``cam_side``;
    ``/camera/camera/color/image_raw/compressed`` -> ``camera``.
    """
    drop = {"image_raw", "compressed", "color", "image", "raw"}
    parts = [p for p in topic.strip("/").split("/") if p not in drop]
    deduped: list[str] = []
    for p in parts:  # collapse consecutive duplicates (camera/camera -> camera)
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    return "_".join(deduped) if deduped else topic.strip("/")


def decode_cameras(tree: dict) -> dict[str, np.ndarray]:
    """Decode every RawImage topic into a name-keyed ``{cam: images}``, truncated to the shortest stream."""
    images = {
        camera_name(topic): decode_raw_image(value)
        for topic, value in tree["topics"].items()
        if value["schema"] == "foxglove.RawImage"
    }
    if not images:
        raise ValueError(f"episode has no RawImage topics: {sorted(tree['topics'])}")
    n = min(len(v) for v in images.values())
    return {k: v[:n] for k, v in images.items()}


class McapLoader:
    """Load one MCAP file as one episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        max_messages_per_topic: int | None = None,
    ):
        self.path = Path(path).expanduser()
        self.max_messages_per_topic = max_messages_per_topic
        self.files = self._find_files(recursive)

    def _find_files(self, recursive: bool) -> tuple[Path, ...]:
        if self.path.is_file():
            if self.path.suffix != ".mcap":
                raise ValueError(f"expected an .mcap file: {self.path}")
            return (self.path,)
        pattern = "**/*.mcap" if recursive else "*.mcap"
        files = tuple(sorted(self.path.glob(pattern)))
        if not files:
            raise ValueError(f"no .mcap files under {self.path}")
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.iter_dataset())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return read_mcap(
            idx,
            self.files[idx],
            max_messages_per_topic=self.max_messages_per_topic,
        )

    def dataset(
        self,
        *,
        progress_offset: int | None = None,
        progress_slots: int = 1,
    ) -> grain.MapDataset[dict[str, Any]]:
        read = partial(
            read_mcap,
            max_messages_per_topic=self.max_messages_per_topic,
            progress_offset=progress_offset,
            progress_slots=progress_slots,
        )
        return grain.MapDataset.source(self.files).map_with_index(read)

    def iter_dataset(
        self,
        *,
        read_threads: int = 4,
        prefetch_buffer_size: int = 2,
        stop: int | None = None,
        show_message_progress: bool = False,
    ) -> Iterable[dict[str, Any]]:
        ds = self.dataset(
            progress_offset=1 if show_message_progress else None,
            progress_slots=max(1, read_threads),
        )
        if stop is not None:
            ds = ds[:stop]
        return ds.to_iter_dataset(
            grain.ReadOptions(
                num_threads=read_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
