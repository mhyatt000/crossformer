from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from crossformer.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from crossformer.data.utils.text_processing import UniversalSentenceEncoder
from crossformer.model.components.action_heads import L1ActionHead
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.train_utils import resnet_26_loader

BIMANUAL_ACTION_DIM = 14
SINGLE_ARM_ACTION_DIM = 7
NAV_ACTION_DIM = 2
QUADRUPED_ACTION_DIM = 12
MANO_ACTION_DIM = 111

HEAD_TO_DATASET = {
    "mano": [
        "rlds_oakink",
    ],
    # 'binamo': [ ],
    # 'smplx': [ ],
    "nav": ["omnimimic_gnm_dataset"],
    "single_arm": [
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


def get_dataset_config(task_cond, window_size, action_horizon, mix="bafl"):
    traj_transform_kwargs, frame_transform_kwargs = get_augmentation_config(
        task_cond, window_size, action_horizon
    )

    assert all(
        [
            any([name in datasets for datasets in HEAD_TO_DATASET.values()])
            for name, weight in OXE_NAMED_MIXES[mix]
        ]
    ), f"Dataset in mix: {mix} doesn't have assigned head."

    return dict(
        oxe_kwargs=dict(
            data_mix=mix,
            data_dir="",
            # dont need the extra views
            load_camera_views=("primary",), #  "high"), # , "nav", "left_wrist", "right_wrist"),
            load_proprio=False,
            load_depth=False,
        ),
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        batch_size=512, # used over finetune batch size bc of make_interleaved
        shuffle_buffer_size=50000,
        balance_weights=False,
        traj_transform_threads=48,
        traj_read_threads=48,
    )


def get_augmentation_config(task_cond, window_size, action_horizon):
    if task_cond == "image":
        keep_image_prob = 1.0
    elif task_cond == "lang":
        keep_image_prob = 0.0
    elif task_cond == "multi":
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=action_horizon,
        max_action_dim=MANO_ACTION_DIM,
        max_proprio_dim=MANO_ACTION_DIM,
        head_to_dataset=HEAD_TO_DATASET,
        goal_relabeling_strategy="uniform",
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        subsample_length=100,
    )

    aloha_image_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.75, 4.0 / 3]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    bridge_image_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (224, 224),
            "high": (224, 224),
            "nav": (224, 224),
            "left_wrist": (224, 224),
            "right_wrist": (224, 224),
        },
        image_augment_kwargs={
            "primary": bridge_image_augment_kwargs,
            "high": aloha_image_augment_kwargs,
            "nav": bridge_image_augment_kwargs,
            "left_wrist": aloha_image_augment_kwargs,
            "right_wrist": aloha_image_augment_kwargs,
        },
        num_parallel_calls=200,
    )
    return traj_transform_kwargs, frame_transform_kwargs


def get_config():
    # whether to finetune the entire model or just the action head
    mode = "full"

    # whether to finetune with image conditioning, language conditioning, or both
    task = "multimodal"

    # the name of the action head to finetune
    head_name = "mano"

    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only"]

    # fill this in to configure data loading for your dataset.
    FINETUNING_KWARGS = dict(
        name="bridge_dataset",
        data_dir="",
        image_obs_keys={"primary": "image_0"},
        proprio_obs_keys={},
        language_key="language_instruction",
        action_proprio_normalization_type="normal",
        # We want to avoid normalizing the gripper
        action_normalization_mask=[True] * 111,
        # standardize_fn is dynamically loaded from a file
        standardize_fn=ModuleSpec.create(
            "crossformer.data.oxe.oxe_standardization_transforms:oakink_dataset_transform",
            # "crossformer.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
        ),
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    )

    # an example of how to add a new observation tokenizer and action head
    UPDATE_CONFIG = dict(
        model=dict(
            # observation_tokenizers=dict( new_primary=ModuleSpec.create( ImageTokenizer, obs_stack_keys=["image_primary"], task_stack_keys=["image_primary"], task_film_keys=["language_instruction"], encoder=ModuleSpec.create(ResNet26FILM),)),
            heads=dict(
                mano=ModuleSpec.create(
                    L1ActionHead,
                    action_horizon=1,
                    action_dim=MANO_ACTION_DIM,
                    num_preds=MANO_ACTION_DIM,
                    pool_strategy="use_map",
                    # pool_strategy="pass",
                    readout_key="readout_mano",
                    clip_pred=False,
                    loss_weight=1.0,
                    constrain_loss_dims=True,
                ),
            ),
            readouts=dict(mano=1),
        )
    )

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("crossformer_transformer.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)

    dataset_kwargs = get_dataset_config("multi", window_size, 100)
    config = dict(
        update_config=UPDATE_CONFIG,  # uncomment this line to add new observation tokenizer and action head
        pretrained_path="hf://rail-berkeley/crossformer",
        pretrained_step=placeholder(int),
        batch_size=dataset_kwargs['batch_size'],  
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000,
        save_interval=1000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="crossformer_finetune",
            group=placeholder(str),
            entity=placeholder(str),
        ),
        # dataset_kwargs=FINETUNING_KWARGS,
        dataset_kwargs=dataset_kwargs,
        prefetch_num_batches=2,
        modality=task,
        finetuning_mode=mode,
        head_name=head_name,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        eval_datasets=(),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (224, 224),  # workspace (3rd person) camera is at 224x224
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    config["frame_transform_threads"] = (
        16  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    )

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
