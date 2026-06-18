from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import grain
from rich import print
from tqdm import tqdm
import tyro

from crossformer.data.grain.loader import _apply_fd_limit
from crossformer.data.mcap import McapLoader
from crossformer.utils.spec import spec


@dataclass
class Config:
    path: Path
    preview: int | None = None  # n preview
    recursive: bool = True
    max_messages_per_topic: int | None = None
    read_threads: int = 4
    prefetch_buffer_size: int = 2

    verbose: bool = False
    vbar: bool = False  # show message-level progress bars

    mp: int = 4
    mp_buf: int = 4  # per worker buffer size


def main(cfg: Config) -> None:
    loader = McapLoader(
        cfg.path,
        recursive=cfg.recursive,
        max_messages_per_topic=cfg.max_messages_per_topic,
    )
    print(f"Root: {loader.path}")
    print(f"n_episodes={len(loader)}")

    n = max(0, min(cfg.preview, len(loader))) if cfg.preview else len(loader)
    ds = loader.iter_dataset(
        read_threads=cfg.read_threads,
        prefetch_buffer_size=min(cfg.prefetch_buffer_size, n),
        stop=n,
        show_message_progress=cfg.vbar,
    )

    lim = _apply_fd_limit(512**2)
    ds = ds.mp_prefetch(
        grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_buf),
    )

    for i, episode in enumerate(tqdm(ds, total=n, desc="Loading episodes", position=0)):
        if cfg.verbose:
            print(f"\n[bold]episode={i} path={episode['info']['path']}[/bold]")
            print(spec(episode))


if __name__ == "__main__":
    main(tyro.cli(Config))
