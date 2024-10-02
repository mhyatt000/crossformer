""" saves the oakink sequences (with horizon=4) to disk as gifs """

import argparse
import json
import os
from typing import Callable, Tuple

# from manotorch.manolayer import ManoLayer
# from oikit.oi_image.oi_image import OakInkImageSequence
# from oikit.oi_image.utils import persp_project
from _oikit import (  # OpenDRRenderer,
    caption_view,
    edge_color_hand,
    edge_list_hand,
    edge_list_obj,
    vert_color_hand,
    vert_type_hand,
)
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from termcolor import cprint
from tqdm import tqdm

import wandb


def persp_project(points3d, cam_intr):
    hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
    points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
    return points2d.astype(np.float32)


def imgs2gif(imgs, gif_path, fps=10):
    import imageio

    imageio.mimsave(gif_path, imgs, fps=fps)
    return


import matplotlib.pyplot as plt

# Get the magma colormap
cmap = plt.get_cmap("magma")
colors_middle_50 = [cmap(i / 19 + 5 / 19) for i in range(10)]  # Middle 50% range
colors_middle_75 = [cmap(i / 19 + 2.5 / 19) for i in range(25)]  # Middle 75% range
colors_all = np.array([cmap(i / 9) for i in range(10)])
horizon_colors = colors_middle_75


def scaleto(n: int, mode: Callable):
    """
    scales the image so that the mode dimension is n pixels,
    and the other dimension is scaled by the same ratio.

    Args:
        n (int): The dim of the mode dimension of the image.
        mode (Callable): The mode to scale the image, either max or min dim.

    Returns:
        Callable: The transform function
    """

    def _scale(image):
        height, width = image.shape[:2]
        primary = mode(height, width)
        # other = [x for x in [height, width] if x != primary][0]

        ratio = n / primary
        shape = (int(height * ratio), int(width * ratio))
        return tf.image.resize(image, shape).numpy().astype(np.uint8)

    return _scale


def draw_wireframe(
    img,
    vert_list,
    vert_color=horizon_colors[0],
    edge_color=horizon_colors[0],
    edge_list=edge_list_obj,
    vert_size=1,  # 3
    edge_size=1,
    vert_type=None,
    vert_mask=None,
):

    vert_list = np.asarray(vert_list)
    n_vert = len(vert_list)
    n_edge = len(edge_list)
    vert_color = np.asarray(vert_color)
    edge_color = np.asarray(edge_color)

    # expand edge color
    if edge_color.ndim == 1:
        edge_color = np.tile(edge_color, (n_edge, 1))

    # expand edge size
    if isinstance(edge_size, (int, float)):
        edge_size = [edge_size] * n_edge

    # # expand vert color
    if vert_color.ndim == 1:
        vert_color = np.tile(vert_color, (n_vert, 1))

    # expand vert size
    if isinstance(vert_size, (int, float)):
        vert_size = [vert_size] * n_vert

    # set default vert type
    if vert_type is None:
        vert_type = ["circle"] * n_vert

    # draw edge
    for edge_id, connection in enumerate(edge_list):
        if vert_mask is not None:
            if not vert_mask[int(connection[1])] or not vert_mask[int(connection[0])]:
                continue
        coord1 = vert_list[int(connection[1])]
        coord2 = vert_list[int(connection[0])]
        cv2.line(
            img,
            coord1.astype(np.int32),
            coord2.astype(np.int32),
            color=edge_color[edge_id] * 255,
            thickness=edge_size[edge_id],
        )

    for vert_id in range(vert_list.shape[0]):
        if vert_mask is not None:
            if not vert_mask[vert_id]:
                continue
        draw_type = vert_type[vert_id]

        markers = {
            # if vert_id in [1, 5, 9, 13, 17]:  # mcp joint
            "circle": cv2.MARKER_CROSS,
            # if vert_id in [2, 6, 10, 14, 18]:  # proximal joints
            "square": cv2.MARKER_SQUARE,
            # if vert_id in [3, 7, 11, 15, 19]:  # distal joints:
            "triangle_up": cv2.MARKER_TRIANGLE_UP,
            # if vert_id in [4, 8, 12, 16, 20]: # fingertip joints
            "diamond": cv2.MARKER_DIAMOND,
            "star": cv2.MARKER_STAR,
        }

        center = (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1]))
        size = vert_size[vert_id] * 2

        cv2.drawMarker(
            img,
            center,
            color=vert_color[vert_id] * 255,
            markerType=markers.get(draw_type, cv2.MARKER_CROSS),
            markerSize=size,
        )


def draw_wireframe_hand(img, hand_joint_arr, hand_joint_mask, hcolor=horizon_colors[0]):
    draw_wireframe(
        img,
        hand_joint_arr,
        edge_list=edge_list_hand,
        vert_color=hcolor,
        edge_color=hcolor,
        vert_type=vert_type_hand,
        vert_mask=hand_joint_mask,
    )


"""
def viz_a_seq( oi_seq: OakInkImageSequence, draw_mode="wireframe", render=None, hand_faces=None):

    imgs = []
    for i in range(len(oi_seq)):
        image = oi_seq.get_image(i)

        for h, color in enumerate(horizon_colors):
            if i + h < len(oi_seq):

                if draw_mode == "wireframe":
                    joints_2d = oi_seq.get_joints_2d(i + h)
                    corners_2d = oi_seq.get_corners_2d(i + h)
                else:
                    hand_verts = oi_seq.get_verts_3d(i + h)
                    obj_verts = oi_seq.get_obj_verts_3d(i + h)
                    obj_faces = oi_seq.get_obj_faces(i + h)

                hand_over_info = oi_seq.get_hand_over(i + h)
                cam_intr = oi_seq.get_cam_intr(i + h)
                if hand_over_info is not None:
                    alt_joints_3d = hand_over_info["alt_joints"]
                    alt_verts_3d = hand_over_info["alt_verts"]
                    alt_joints_2d = persp_project(alt_joints_3d, cam_intr)

                # draw
                draw_wireframe_hand(
                    image, joints_2d, hand_joint_mask=None, hcolor=horizon_colors[h]
                )
                # draw_wireframe(image, vert_list=corners_2d)
                if hand_over_info is not None:
                    draw_wireframe_hand(
                        image,
                        alt_joints_2d,
                        hand_joint_mask=None,
                        hcolor=horizon_colors[h],
                    )

        image = caption_view(image, caption=f"::{oi_seq._name}")
        imgs.append(image)

        # cv2.imshow("tempf/x.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

    imgs2gif(imgs, f"tempf/{oi_seq._name.replace('/','.')}.gif", fps=10)
    return
"""


def denormalize(thing, stats, ds_key="rlds_oakink", key="action"):
    """denormalizes the thing with mean and std where mask is True"""
    mask = stats[ds_key][key]["mask"]
    mean = stats[ds_key][key]["mean"]
    std = stats[ds_key][key]["std"]
    thing = np.where(mask, (thing * std + mean), thing)
    return thing


class SequenceViz:

    videos = []

    def __init__(self, imgs, joints, deltas, cam):
        self.imgs = imgs
        self.joints = joints
        self.deltas = deltas
        self.cam = cam
        self.nstep = len(imgs) if len(imgs.shape) == 4 else len(imgs[0])
        self.task = None

        self.scaler = scaleto(480, min)
        self.batched = False  # by default
        self.resolution = 2

    @staticmethod
    def from_batch(batch, stats):

        proprio = np.array(batch["observation"]["proprio_mano"])

        actions = denormalize(np.array(batch["action"]), stats=stats)[
            :, :, :, : 21 * 3
        ]  # h dim

        joints = proprio[:, :, : 21 * 3]
        cam_intr = proprio[:, :, -3 * 3 :]
        s = SequenceViz(
            imgs=np.array(batch["observation"]["image_primary"]),
            joints=joints,
            deltas=actions,
            cam=cam_intr,
        )
        s.task = [None] * len(s.imgs)
        s.batched = True
        s.stats = stats
        return s

    @staticmethod
    def compute_deltas_from_joints(joints):
        deltas = joints[1:] - joints[:-1]
        deltas = np.concatenate([deltas, np.zeros_like(joints[-1:])], axis=0)
        return deltas

    def _wandb(self, imgs, joints, deltas, cams, task=None):
        """returns an gif for wandb logging"""

        frames = []
        # steps = [s for s in example["steps"]]

        # print(imgs.shape)
        for i, image in enumerate(imgs):
            # print(image.shape)
            # image = self.scaler(image)

            # task = s["language_instruction"].numpy().decode("utf-8")

            # print(deltas.shape)
            # print(deltas[i].shape)
            # print('enter for')
            # bs,window,horizon,dims .. we are selecting window from this sample
            j2d = persp_project(joints[i].reshape(21, 3), cams[i].reshape(3, 3))
            draw_wireframe_hand(
                image, j2d, hand_joint_mask=None, hcolor=horizon_colors[0]
            )

            dw = deltas[i]
            for h, dh in enumerate(dw):
                if (i + h) % self.resolution:
                    continue
                # if i + h < self.nstep:
                # # print(f"plotting for horizon step {i+h}")

                # print("deltas", dw[: i + h+1].shape)
                j = joints[i] + dw[: i + h + 1].sum(axis=0)
                # print("delta sum", dw[: i + h+1].sum(axis=0).shape)
                # print("joints", j.shape)
                cam = cams[i]

                j2d = persp_project(j.reshape(21, 3), cam.reshape(3, 3))
                draw_wireframe_hand(
                    image, j2d, hand_joint_mask=None, hcolor=horizon_colors[i + h]
                )

            # image = caption_view(image, caption=f"::{task}")
            frames.append(image)
        frames = np.array(frames).transpose(0, 3, 1, 2)  # colors first
        return wandb.Video(frames, fps=10)

        # imgs2gif(frames, f"tempf/{task}.gif", fps=10)

    def wandb(self):
        if self.batched:
            for i in range(len(self.imgs)):
                video = self._wandb(
                    self.imgs[i],
                    self.joints[i],
                    self.deltas[i],
                    self.cam[i],
                    task=self.task[i],
                )
                SequenceViz.videos.append(video)
        else:
            video = self._wandb(
                self.imgs, self.joints, self.deltas, self.cam, task=self.task
            )
            SequenceViz.videos.append(video)

    def flush(i, limit=32):
        wandb.log({f"videos/mesh.vid": SequenceViz.videos[:limit]}, step=i)
        SequenceViz.videos = []


def from_rlds():

    ds = tfds.load("rlds_oakink", split="train", shuffle_files=True)

    for example in tqdm(ds.take(10)):

        imgs = []
        ns = len(example["steps"])
        steps = [s for s in example["steps"]]

        joints = tf.constant(
            [s["observation"]["state"]["joints_3d"].numpy() for s in steps]
        )

        deltas = joints[1:] - joints[:-1]
        deltas = tf.concat([deltas, tf.zeros_like(joints[-1:])], axis=0)

        # Verify that joints[i] + deltas[i] matches joints[i+1]
        # for i in range(len(joints) - 1):
        # # print(f"Checking joint {i}:")
        # # print(f"Delta: {deltas[i]}")
        # # print(f"Joint {i}: {joints[i]}")
        # # print(f"Joint {i+1}: {joints[i+1]}")

        # Assertion to check if the sum of joints[i] and deltas[i] equals joints[i+1]
        # print( (joints[i] + deltas[i] == joints[i + 1]).numpy().all())
        # , f"Assertion failed at index {i}"

        for i, s in enumerate(steps):
            image = s["observation"]["image"].numpy()
            image = scaler(image)
            task = s["language_instruction"].numpy().decode("utf-8")

            for h, color in enumerate(horizon_colors):
                if i + h < ns:

                    future = steps[i + h]
                    j = joints[0].numpy() + deltas[: i + h].numpy().sum(axis=0)
                    cam = future["observation"]["state"]["cam_intr"].numpy()

                    size = image.shape[:2]

                    joints_2d = persp_project(j, cam)
                    draw_wireframe_hand(
                        image, joints_2d, hand_joint_mask=None, hcolor=horizon_colors[h]
                    )

            image = caption_view(image, caption=f"::{task}")
            imgs.append(image)

        imgs2gif(imgs, f"tempf/{task}.gif", fps=10)
    return