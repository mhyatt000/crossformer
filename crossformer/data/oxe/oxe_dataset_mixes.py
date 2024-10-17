"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""

HEAD_TO_DATASET = {
    "mano": [
        "rlds_oakink",
    ],
    # 'binamo': [ ],
    # 'smplx': [ ],
    "nav": ["omnimimic_gnm_dataset"],
    "single_arm": [
        "xgym_single",
        #
        "berkeley_mvp_converted_externally_to_rlds",
        "nyu_rot_dataset_converted_externally_to_rlds",
        "ucsd_kitchen_dataset_converted_externally_to_rlds",
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        "utokyo_xarm_bimanual_converted_externally_to_rlds",
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
        #
        "bridge_dataset",
        "fractal20220817_data",
        "kuka",
        "taco_play",
        "taco_extra",
        "jaco_play",
        "berkeley_cable_routing",
        "roboturk",
        "nyu_door_opening_surprising_effectiveness",
        "viola",
        "berkeley_autolab_ur5",
        "toto",
        "language_table",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "austin_buds_dataset_converted_externally_to_rlds",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "bc_z",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "utaustin_mutex",
        "berkeley_fanuc_manipulation",
        "cmu_stretch",
        "droid",
        "droid_wipe",
        "droid_flip_pot_upright",
    ],
    "bimanual": [
        "aloha_pen_uncap_diverse_dataset",
        "aloha_new_sushi_dataset",
        "aloha_dough_cut_dataset",
        "aloha_lucy_dataset",
        "aloha_drawer_dataset",
        "aloha_pick_place_dataset",
        "aloha_static_dataset",
        "aloha_sushi_cut_full_dataset",
        "aloha_new_sushi_dataset,",
    ],
    "quadruped": ["go1_real_dataset", "a1", "go1"],
}


OXE_MAGIC_SOUP_BALANCED = [
    ("kuka", 0.14503701874493363),
    ("taco_play", 0.06657998827701668),
    ("taco_extra", 0.015452958868388737),
    ("jaco_play", 0.010914534155076169),
    ("berkeley_cable_routing", 0.005925612796973822),
    ("roboturk", 0.052499238268860826),
    ("nyu_door_opening_surprising_effectiveness", 0.0028565519070650833),
    ("viola", 0.021369612129854),
    ("berkeley_autolab_ur5", 0.027421498380401588),
    ("toto", 0.045595496181288435),
    ("language_table", 0.09863155061985435),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 0.10030032010542056),
    ("austin_buds_dataset_converted_externally_to_rlds", 0.004775432426062442),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 0.01884652293499813),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.05526993262706029),
    ("austin_sailor_dataset_converted_externally_to_rlds", 0.04943059735717906),
    ("austin_sirius_dataset_converted_externally_to_rlds", 0.03918942829266809),
    ("bc_z", 0.14503701874493363),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 0.00124985520344411),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 0.020472678629801757),
    ("utaustin_mutex", 0.05066099356944051),
    ("berkeley_fanuc_manipulation", 0.017530731149920712),
    ("cmu_stretch", 0.003502058441908362),
    ("droid", 0.001450370187449336),
]

CROSS_EMBODIMENT_TARGET = [
    ("aloha_pen_uncap_diverse_dataset", 0.1),
    ("aloha_new_sushi_dataset", 0.2),
    ("bridge_dataset", 0.2),
    ("a1", 0.1),
    ("droid_wipe", 0.1),
    ("droid_flip_pot_upright", 0.1),
    ("omnimimic_gnm_dataset", 0.2),
]

CROSS_EMBODIMENT = [
    (name, weight * 0.15) for name, weight in OXE_MAGIC_SOUP_BALANCED
] + [(name, weight * 0.85) for name, weight in CROSS_EMBODIMENT_TARGET]


import rlds_oakink

allweight = lambda arr, w: [(name, weight * w) for name, weight in arr]

BAFL_SOUP = [
    ("rlds_oakink", 1.0),
    ("bridge_dataset", 0.2),
    ("berkeley_mvp_converted_externally_to_rlds", 0.2),
    ("nyu_rot_dataset_converted_externally_to_rlds", 0.2),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 0.2),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 0.2),
    # ("utokyo_xarm_bimanual_converted_externally_to_rlds", 0.2),
    # ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 0.2),
    ("xgym_single", 1.0),
    # ("xgym_mano", 1.0),
]  # + allweight(OXE_MAGIC_SOUP_BALANCED, 0.01)


XGYM = [
    ("xgym_single", 1.0),
]

OXE_NAMED_MIXES = {
    "cross_embodiment": CROSS_EMBODIMENT,
    "bafl": BAFL_SOUP,
    "bridge": [("bridge_dataset", 1.0)],
    "xgym": XGYM,
}
