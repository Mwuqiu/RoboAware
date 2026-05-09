# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey


DATASET_DIR = "/root/autodl-tmp/cosmos_training_data_world_arena"
OVERFIT_VIDEO_PATHS = [
    f"{DATASET_DIR}/videos/TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000040.mp4",
    f"{DATASET_DIR}/videos/TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000041.mp4",
    f"{DATASET_DIR}/videos/TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000042.mp4",
    f"{DATASET_DIR}/videos/TianxingChen_RoboTwin2.0_adjust_bottle_aloha-agilex_ep_000043.mp4",
]

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

checkpoint_conf = dict(
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
    save_iter=0,
)

defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /conditioner": "pc_video_prediction_conditioner"},
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

pc_encoder_config = dict(
    DATASET="robotwin",
    CONFIG="semseg-pt-v3m1-0-base",
    EXP_NAME="semseg-pt-v3m1-0-base-cosmos-pcenc",
    WEIGHT_NAME="model_last",
)

dataset_def = L(VideoDataset)(
    dataset_dir=DATASET_DIR,
    video_paths=OVERFIT_VIDEO_PATHS,
    num_frames=93,
    video_size=(480, 832),
    pc_latent_source="precomputed",
    pc_latent_amp=True,
    pc_encoder_config=pc_encoder_config,
)

dataloader_train = L(get_generic_dataloader)(
    dataset=dataset_def,
    sampler=L(get_sampler)(dataset=dataset_def),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

dataloader_val = L(get_generic_dataloader)(
    dataset=dataset_def,
    sampler=L(get_sampler)(dataset=dataset_def),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

optimizer_conf = dict(
    lr=1e-4,
    weight_decay=0.001,
)

scheduler_conf = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[50],
    cycle_lengths=[100000],
)

trainer_conf = dict(
    run_validation=False,
    validation_iter=1000000,
    logging_iter=10,
    max_iter=300,
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=1000000, save_s3=False),
        every_n_sample_ema=dict(every_n=1000000, save_s3=False),
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=0,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=True,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=0,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=True,
        ),
    ),
)

model_conf = dict(
    config=dict(
        use_lora=False,
        min_num_conditional_frames=0,
        max_num_conditional_frames=2,
        conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},
        conditional_frame_timestep=-1.0,
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
        net=dict(
            point_adapter_d_a=None,
            point_adapter_num_adapter_blocks=4,
            point_adapter_block_depth=1,
            point_adapter_num_heads=None,
            point_adapter_inject_block_ids=[5, 11, 17, 23],
            point_adapter_inject_every_k=6,
            point_adapter_mlp_ratio=None,
            point_adapter_dropout=0.0,
        ),
    ),
)

model_parallel_conf = dict(
    context_parallel_size=1,
)

EXPERIMENT_NAME = "predict2_point_adapter_training_2b_world_arena_full_point_condition_overfit4"

experiment_config = dict(
    defaults=defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="point_adapter",
        name="2b_world_arena_full_point_condition_overfit4",
    ),
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    checkpoint=checkpoint_conf,
    optimizer=optimizer_conf,
    scheduler=scheduler_conf,
    trainer=trainer_conf,
    model=model_conf,
    model_parallel=model_parallel_conf,
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=experiment_config,
)
