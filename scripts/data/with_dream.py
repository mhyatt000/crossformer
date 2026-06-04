from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol

import grain
from grain.experimental import ThreadPrefetchIterDataset
import jax
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client

from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.write import add_episode_id, add_step_id, add_traj_len, BuildMGR, init_info
from crossformer.utils.spec import spec


@dataclass
class Endpoint:
    host: str
    port: int


@dataclass
class Sam(Endpoint):
    prompt: str = "robot"  # SAM3 text prompt
    confidence: float = 0.3  # SAM3 confidence threshold

    # raw_webpolicy: bool = True  # Send raw payloads for older SAM3 webpolicy servers

    close_kernel_size: int = 3  # Morphological close kernel; 0 disables
    min_component_area: int = 16  # Remove tiny mask islands; 0 disables


@dataclass
class Roboreg(Endpoint):
    pass


@dataclass
class Dream(Endpoint):
    units: Literal["deg", "rad"] = "deg"  # DREAM server converts deg to rad internally


def make_intr(fx, fy, w, h):
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])


@dataclass
class MyBuildMGR(BuildMGR):
    take: int | None = None  # debug. take n steps

    image_size: int = 200  # Square size for SAM, Dream, and DR
    fxy: float = 515.0  # Focal length for DR depth-to-3D conversion
    mask_area: list[float] = default([0.005, 0.8])  # Reject tiny/large masks

    dr: bool = True  # whether to use sam+roboreg or only dream

    sam: Sam = default(Sam(host="localhost", port=8080))
    reg: Roboreg = default(Roboreg(host="localhost", port=8081))
    dream: Dream = default(Dream(host="localhost", port=8082))

    def __post_init__(self):
        super().__post_init__()
        assert self.name in Arec.REGISTRY, f"mix should be one of {list(Arec.REGISTRY.keys())}"

    @property
    def mix(self):
        return Arec.REGISTRY[self.name]


class ClientLike(Protocol):
    def step(self, payload: dict) -> dict: ...


class SamClientWrapper:
    def __init__(self, client: Client, cfg: Sam):
        self.client = client
        self.cfg = cfg

    def step(self, image: np.ndarray) -> np.ndarray:
        """Segment the image using SAM and return a mask."""
        h, w, c = image.shape  # noqa
        payload = {
            "image": image,
            "type": "image",
            "text": self.cfg.prompt,
            "confidence": self.cfg.confidence,
        }
        out = self.client.step(payload)

        valid = np.prod(out["masks"].shape) > 0
        return {
            "seg": out["masks"].any(axis=0).reshape(h, w, 1) if valid else np.zeros((h, w, 1)).astype(bool),
            "valid": valid,
        }


from crossformer.utils.autobox import Box


def do_segmentation(x: dict, sam: ClientLike):
    """Run SAM segmentation on all images"""
    seg = [sam.step(im) for im in x["image"]]
    seg = jax.tree.map(lambda *xs: np.stack(xs), *seg)  # check all seg outputs have same shape

    # some masks are not valid and have shape (0,*) but this is handled in part by the client wrapper
    x["seg"] = seg["seg"]
    x = Box(x)
    # x['mask']['obs']['seg'] = seg['valid']  # add seg validity to mask obs
    x.auto().mask.obs.seg = seg["valid"]  # add seg validity to mask obs
    print(spec(x))

    return x


def do_dream(x: dict, dream: ClientLike, cfg: MyBuildMGR):
    K = make_intr(fx=cfg.fxy, fy=cfg.fxy, w=cfg.image_size, h=cfg.image_size)
    # TODO resize images according to cfg
    payload = {
        "image": x.image,  # expects list of images
        "K": K,  # camera intrinsics for depth-to-3D conversion
        "q": x.proprio.joints[0],  # robot joint angles
        "type": "image",
        "calibrate": True,  # ???
    }
    out = Box(dream.step(payload))
    print(spec(out))
    print(out.w2c)
    print(out.mask_iou, out.mask_iou_reject)
    x.auto().extr.w2c = out.w2c
    x.auto().extr.w2c_cv = out.w2c  # TODO im assuming its cv convention. need to verify
    x.auto().mask.extr.w2c = np.logical_not(np.isnan(out.w2c).any(axis=(1, 2)))
    print(x.mask.extr.w2c)
    return x


def do_registration(x: dict, roboreg: ClientLike, cfg: MyBuildMGR):
    # TODO filter out bad SAM masks according to cfg.mask_area before registration

    # pick_best_w2c
    # filter_dr_frames

    w2c_sam = roboreg.step(x)
    # expects images, joints,  masks, intrinsics and HT initial and ht_is_cv_w2c

    x["extr"]["w2c_sam"] = w2c_sam
    payload = {
        "images": x.image,
        "joints": x.proprio.joints[0],
        "masks": x.seg,  # or x.mask.obs.seg if we want to use seg validity
        "intrinsics": make_intr(fx=cfg.fxy, fy=cfg.fxy, w=cfg.image_size, h=cfg.image_size),
        "HT": x.extr.w2c,
        "ht_is_cv_w2c": True,
        "mode": "dr",  # Literal['icp', 'dr', 'both']
    }
    out = roboreg.step(payload)
    print(spec(out))
    quit()
    return x


def main(cfg: MyBuildMGR):
    sam = SamClientWrapper(Client(host=cfg.sam.host, port=cfg.sam.port), cfg=cfg.sam)
    roboreg = Client(host=cfg.reg.host, port=cfg.reg.port)
    dream = Client(host=cfg.dream.host, port=cfg.dream.port)

    ds = grain.MapDataset.source(cfg.mix.source)

    ds = ds.map(init_info).map(add_traj_len).map(add_step_id).map_with_index(add_episode_id)

    # materialize to compute total steps for progress bar
    # total, n = sum([x["info"]["len"][0] for x in tqdm(ds, desc="compute total")]), len(ds)
    total = len(ds)
    # print(f"total steps: {total} across {n} episodes")

    if cfg.take:  # debug
        dsit = iter(ds)
        ds = grain.MapDataset.source([next(dsit) for _ in range(cfg.take)])

    # force clients runs serially
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)

    ds = ds.map(partial(do_segmentation, sam=sam))
    ds = ds.map(partial(do_dream, dream=dream, cfg=cfg))
    ds = ds.map(partial(do_registration, roboreg=roboreg, cfg=cfg)) if cfg.dr else ds

    # ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.len", use_np=True))

    dsit = iter(ds)
    valids = []
    for i in tqdm(range(5000)):
        x = next(dsit)
        print(spec(x.dict))
        # print(x['mask']['obs']['valid'])
        print(x.mask.obs.seg)
        valids.append(x.mask.obs.seg.mean())
        print(np.array(valids).mean())

        print()
    quit()

    ds = ds.map(cfg.progress(total))
    cfg.build(cfg.yield_from_ds(ds))


if __name__ == "__main__":
    main(tyro.cli(MyBuildMGR))
