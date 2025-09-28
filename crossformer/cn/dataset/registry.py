from __future__ import annotations

from crossformer.cn.dataset.action import (
    ActionSpace,
    BasicDataSpec,
    DATA_SPECS,
    DataSpec,
    DIMOBS,
    IMOBS,
)

fractal = BasicDataSpec(
    name="fractal20220817_data",
)
kuka = BasicDataSpec(name="kuka")
# NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
# can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
bridge = BasicDataSpec(name="bridge_dataset", image_obs_keys=IMOBS(primary="image_0"))

taco_play = BasicDataSpec(
    name="taco_play",
    image_obs_keys=IMOBS(primary="rgb_static"),
    depth_obs_keys=DIMOBS(primary="depth_static", wrist="depth_gripper"),
)
taco_extra = BasicDataSpec(
    name="taco_extra", image_obs_keys=IMOBS(primary="rgb_static")
)
jaco_play = BasicDataSpec(name="jaco_play")

bcable = BasicDataSpec(
    name="berkeley_cable_routing",
    proprio_encoding=ActionSpace.JOINT,
    action_encoding=ActionSpace.POS_EULER,
)

roboturk = BasicDataSpec(
    name="roboturk",
    image_obs_keys=IMOBS(primary="front_rgb"),
    proprio_encoding=ActionSpace.NONE,
    action_encoding=ActionSpace.POS_EULER,
)

nyu_door = BasicDataSpec(
    name="nyu_door_opening_surprising_effectiveness",
    image_obs_keys=IMOBS(left_wrist="image", right_wrist="image"),
    proprio_encoding=ActionSpace.NONE,
    action_encoding=ActionSpace.POS_EULER,
)

viola = BasicDataSpec(
    name="viola",
    image_obs_keys=IMOBS(primary="agentview_rgb"),
    proprio_encoding=ActionSpace.JOINT,
    action_encoding=ActionSpace.POS_EULER,
)

DATA_SPECS = DATA_SPECS | {
    x.name: x
    for x in [
        fractal,
        kuka,
        bridge,
        taco_play,
        taco_extra,
        jaco_play,
        bcable,
        roboturk,
        nyu_door,
        viola,
    ]
}

print(DataSpec.REGISTRY.keys())
