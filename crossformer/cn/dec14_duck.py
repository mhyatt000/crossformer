import os
import os.path as osp

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import tensorflow as tf

from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from crossformer.data.utils.text_processing import UniversalSentenceEncoder
from crossformer.model.components.action_heads import DiffusionActionHead, L1ActionHead
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.train_utils import resnet_26_loader


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
            data_dir=os.environ.get(
                "BAFL_DATA",
                os.path.join(os.path.expanduser("~"), "tensorflow_datasets"),
            ),
            # dont need the extra views
            load_camera_views=(
                "primary",
                'side',
                'high',
                "left_wrist",
                ),
                # , "nav", "left_wrist", "right_wrist"),
            load_proprio=True,
            load_depth=False,
        ),
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        batch_size=256,  # used over finetune batch size bc of make_interleaved
        shuffle_buffer_size=50_000,
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
        max_action_dim=ActionDim.BIMANUAL,
        max_proprio_dim=ActionDim.BIMANUAL,
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
            # "random_resized_crop",
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
            # "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (224, 224),
            "side": (224, 224),
            "high": (224, 224),
            "nav": (224, 224),
            "left_wrist": (224, 224),
            "right_wrist": (224, 224),
        },
        image_augment_kwargs={
            "primary": bridge_image_augment_kwargs ,
            'side': bridge_image_augment_kwargs ,
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

    # borrowed from pretrain cfg
    # token_embedding_size, transformer_kwargs = common_transformer_sizes( transformer_size)
    # encoder = ModuleSpec.create(ResNet26FILM)

    # an example of how to add a new observation tokenizer and action head
    UPDATE_CONFIG = dict(
        model=dict(
            observation_tokenizers=dict(
                side=ModuleSpec.create( 
                    ImageTokenizer, 
                      obs_stack_keys=["image_side"], 
                      task_stack_keys=["image_side"], 
                      task_film_keys=["language_instruction"], 
                      encoder=ModuleSpec.create(ResNet26FILM),
                ),
                single=ModuleSpec.create(
                    LowdimObsTokenizer,
                    obs_keys=["proprio_single"],
                    dropout_rate=0.2,
                ),
            ),
            heads=dict(
                single_arm=ModuleSpec.create(
                    DiffusionActionHead,
                    action_horizon=4,
                    action_dim=ActionDim.SINGLE,
                    num_preds=ActionDim.SINGLE,
                    pool_strategy="mean",  # isnt there another/better strategy
                    readout_key="readout_single_arm",
                    clip_pred=False,
                    loss_weight=1.0,
                    constrain_loss_dims=True,
                    diffusion_steps=20,
                ),
                bimanual=ModuleSpec.create(
                    L1ActionHead,
                    action_horizon=4,
                    action_dim=ActionDim.BIMANUAL,
                    # num_preds=ActionDim.BIMANUAL,
                    pool_strategy="pass",
                    readout_key="readout_bimanual",
                    clip_pred=False,
                    loss_weight=1.0,
                    constrain_loss_dims=True,
                ),
                mano=ModuleSpec.create(
                    DiffusionActionHead,
                    action_horizon=4,
                    action_dim=ActionDim.DMANO_7,
                    pool_strategy="mean",
                    readout_key="readout_mano",
                    clip_pred=False,
                    loss_weight=1.0,
                    constrain_loss_dims=True,
                    diffusion_steps=5,
                ),
            ),
            readouts=dict(single_arm=4, mano=4, bimanual=4),
        )
    )

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("crossformer_transformer.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(300_000)
    grad_acc = None
    max_steps = max_steps * (grad_acc or 1)

    #
    ### should this be higher??
    # was 5 during pretraining
    #
    window_size = FieldReference(default=1)
    # window_size = 3

    action_horizon = 4
    dataset_kwargs = get_dataset_config(
        "multi", window_size, action_horizon=action_horizon, mix="xgym_duck_single"
    )
    config = dict(
        pretrained_path="hf://rail-berkeley/crossformer",
        pretrained_step=placeholder(int),
        resume_path=placeholder(str),
        wandb=dict(
            project="bafl",
            group=placeholder(str),
            entity=placeholder(str),
        ),
        wandb_resume_id=placeholder(str),
        #
        update_config=UPDATE_CONFIG,  # uncomment this line to add new observation tokenizer and action head
        skip_norm_keys=["proprio_bimanual, proprio_mano"],
        config_delete_keys={
            "model": {
                "readouts": {
                    "bimanual": 4,
                    "quadruped": None,
                    "nav": None,
                    "single_arm": 4,
                },
                "observation_tokenizers": {
                    "bimanual": None,
                    "quadruped": None,
                    # "high": None,
                    "nav": None,
                },
                "heads": {
                    "single_arm": "diffusion",
                    "quadruped": None,
                    "nav": None,
                },
            },
        },
        batch_size=dataset_kwargs["batch_size"],
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=2000,  # 2000
        save_interval=2000,
        save_dir=os.path.expanduser(
            os.environ.get("BAFL_SAVE", os.path.expanduser("~"))
        ),
        seed=42,
        # dataset_kwargs=FINETUNING_KWARGS,
        dataset_kwargs=dataset_kwargs,
        prefetch_num_batches=64,
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
            grad_accumulation_steps=grad_acc,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=8,  # 16
        ),
        eval_datasets=("xgym_duck_single", "xgym_lift_single"),
        rollout_kwargs=dict(
            num_envs=4,
            use_rollout=False,
        ),
        debug=False,
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
        action_horizon=action_horizon,  # was 4
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
    # for the most CPU-intensive ops (decoding, resizing, augmenting)
    config["frame_transform_threads"] = 32

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
