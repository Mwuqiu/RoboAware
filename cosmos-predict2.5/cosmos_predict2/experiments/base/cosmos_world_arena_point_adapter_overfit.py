# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

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

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

_adapter_checkpoint_base = dict(
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
    save_iter=2500,
)

_adapter_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /conditioner": "pc_video_prediction_conditioner"},
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

_dataset_pc_encoder_config = dict(
    DATASET="robotwin",
    CONFIG="semseg-pt-v3m1-0-base",
    EXP_NAME="semseg-pt-v3m1-0-base-cosmos-pcenc",
    WEIGHT_NAME="model_last",
)

_overfit_dataset_dir = "/root/autodl-tmp/cosmos_training_data_world_arena"
_overfit_video_dir = os.path.join(_overfit_dataset_dir, "videos")
_overfit_video_paths = [
    os.path.join(_overfit_video_dir, name)
    for name in sorted(os.listdir(_overfit_video_dir))
    if name.endswith(".mp4")
][:4]

# Tiny dense-point reconstruction overfit set: one video per GPU rank with nproc_per_node=4.
dataset_def = L(VideoDataset)(
    dataset_dir=_overfit_dataset_dir,
    video_paths=_overfit_video_paths,
    num_frames=93,
    video_size=(480, 832),
    pc_latent_source="online",
    pc_latent_amp=True,
    pc_encoder_config=_dataset_pc_encoder_config,
)

dataloader_train = L(get_generic_dataloader)(
    dataset=dataset_def,
    sampler=L(get_sampler)(dataset=dataset_def),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

dataloader_val = L(get_generic_dataloader)(
    dataset=dataset_def,
    sampler=L(get_sampler)(dataset=dataset_def),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

optimizer_conf = dict(
    lr=1.0e-4,
    weight_decay=0.001,
)

scheduler_conf = dict(
    f_max=[1.0],
    f_min=[1.0],
    warm_up_steps=[20],
    cycle_lengths=[1000],
)

trainer_conf = dict(
    run_validation=False,
    validation_iter=2000,
    logging_iter=1,
    max_iter=5000,
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=1000000000, save_s3=False, guidance=[3.0], do_x0_prediction=False),
        every_n_sample_ema=dict(every_n=1000000000, save_s3=False, guidance=[3.0], do_x0_prediction=False),
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
        point_diffusion_loss_weight=0.05,
        point_condition_frames=2,
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

EXPERIMENT_NAME = "predict2_point_adapter_dense_point_overfit4"

experiment_config = dict(
    defaults=_adapter_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="point_adapter",
        name="2b_cosmos_world_arena_point_adapter_dense_point_overfit4",
    ),
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    checkpoint=_adapter_checkpoint_base,
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