# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# V5 ControlNet experiment: D4-B' architecture + ControlNet-style weight copy
# from backbone blocks at inject points (5, 11, 17, 23) into adapter blocks 0..3.
#
# Why: prior D4-B' iter_1750 weight_diff showed cross_attn / mlp weights only
# drifted 0.3-1.5% (basically random init). PC swap inference confirmed model
# never learned to use PC for arm-side (3/4 outputs L despite PC=R signals).
# Hypothesis: adapter starts at random trunc_normal, has to learn video-token
# processing + PC injection simultaneously. With ControlNet-style copy, ~70%
# of adapter params start at backbone's pretrained values, only k_proj/v_proj
# (modality-different) + adaLN modulation (zero-init "zero conv") need to learn.
#
# Diffs vs v5_cross_attn_then_mlp_4tasks (parent):
#   - point_adapter_controlnet_copy=True
#   - batch_size=1 (single 96GB GPU on westd)
#   - new experiment_name + output dir

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


DATASET_ROOT = "/root/autodl-tmp/cosmos_training_data_world_arena_ablation_4tasks"
TRAIN_DATASET_DIR = f"{DATASET_ROOT}/training"
VAL_DATASET_DIR = f"{DATASET_ROOT}/test"

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

checkpoint_conf = dict(
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
    save_iter=100,    # frequent ckpts early — test PC swap behavior often
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
    v5_layer="dec_0",
    v5_feat_input="coord",
)

train_pc_conditioning_mode_probs = dict(full=1.0)

dataset_train = L(VideoDataset)(
    dataset_dir=TRAIN_DATASET_DIR,
    num_frames=93,
    video_size=(480, 640),
    pc_latent_source="precomputed",
    pc_latent_k=104,
    pc_latent_amp=True,
    pc_encoder_config=pc_encoder_config,
    pc_conditioning_mode_probs=train_pc_conditioning_mode_probs,
    pc_conditioning_prefix_frames=[1, 2],
)

dataset_val = L(VideoDataset)(
    dataset_dir=VAL_DATASET_DIR,
    num_frames=93,
    video_size=(480, 640),
    pc_latent_source="precomputed",
    pc_latent_k=104,
    pc_latent_amp=True,
    pc_encoder_config=pc_encoder_config,
    pc_conditioning_mode_probs=dict(full=1.0),
    pc_conditioning_prefix_frames=[1, 2],
)

dataloader_train = L(get_generic_dataloader)(
    dataset=dataset_train,
    sampler=L(get_sampler)(dataset=dataset_train),
    batch_size=1,    # single 96GB GPU on westd
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

dataloader_val = L(get_generic_dataloader)(
    dataset=dataset_val,
    sampler=L(get_sampler)(dataset=dataset_val),
    batch_size=1,    # single 96GB GPU on westd
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

optimizer_conf = dict(
    lr=2 ** (-14.5),
    weight_decay=0.001,
)

scheduler_conf = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[500],
    cycle_lengths=[100000],
)

trainer_conf = dict(
    run_validation=False,
    validation_iter=2000,
    logging_iter=25,
    max_iter=5000,
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=1000, save_s3=False),
        every_n_sample_ema=dict(every_n=1000000, save_s3=False),
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=True,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=2,
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
            point_adapter_d_pc=512,
            point_adapter_use_layernorm=True,
            point_adapter_adapter_mode="cross_attn_then_mlp",   # D4-B'
            point_adapter_controlnet_copy=True,                  # ★ ControlNet weight copy
        ),
    ),
)

model_parallel_conf = dict(
    context_parallel_size=1,
)

EXPERIMENT_NAME = "predict2_point_adapter_v5_controlnet_4tasks"

experiment_config = dict(
    defaults=defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="point_adapter",
        name="v5_controlnet_4tasks_480x640",
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
