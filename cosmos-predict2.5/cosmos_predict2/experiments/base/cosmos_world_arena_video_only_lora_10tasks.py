# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Video-only LoRA baseline on the WorldArena 10-task subset.
# Derived from cosmos_nemo_assets_lora.py (upstream LoRA template); only the
# dataset (10-task subset), num_frames (121) and resolution (480x640) differ.
# No point conditioning — pure video2world LoRA fine-tune of the 2B base model,
# to serve as the baseline against the point-adapter model.

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Same 10-task subset used by the point-adapter run; metas/*.txt -> caption_format="text".
DATASET_ROOT = "/root/autodl-tmp/cosmos_training_data_world_arena_10tasks"
TRAIN_DATASET_DIR = f"{DATASET_ROOT}/training"
VAL_DATASET_DIR = f"{DATASET_ROOT}/test"

dataset_train = L(VideoDataset)(
    dataset_dir=TRAIN_DATASET_DIR,
    num_frames=121,
    video_size=(480, 640),
    caption_format="text",
)
dataloader_train = L(get_generic_dataloader)(
    dataset=dataset_train,
    sampler=L(get_sampler)(dataset=dataset_train),
    batch_size=2,        # 4×A800 -> effective batch 8 (matches point-adapter run)
    drop_last=True,
    num_workers=12,
    pin_memory=True,
)

dataset_val = L(VideoDataset)(
    dataset_dir=VAL_DATASET_DIR,
    num_frames=121,
    video_size=(480, 640),
    caption_format="text",
)
dataloader_val = L(get_generic_dataloader)(
    dataset=dataset_val,
    sampler=L(get_sampler)(dataset=dataset_val),
    batch_size=2,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

_lora_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

_lora_checkpoint_base = dict(
    load_path=DEFAULT_CHECKPOINT.s3.uri,
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
)

_lora_optimizer = dict(
    lr=2 ** (-14.5),
    weight_decay=0.001,
)

_lora_scheduler = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[2_000],
    cycle_lengths=[100000],
)

_lora_trainer = dict(
    run_validation=False,
    validation_iter=2000,
    logging_iter=100,
    max_iter=10000,
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
            do_x0_prediction=False,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=False,
        ),
    ),
)

_lora_model_config = dict(
    config=dict(
        # Enable LoRA training (no point adapter — this is the video-only baseline)
        use_lora=True,
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
        # 0/1/2 conditional frames -> text2world / image2world / video2world
        min_num_conditional_frames=0,
        max_num_conditional_frames=2,
        conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},
        conditional_frame_timestep=-1.0,
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
    ),
)

_lora_model_parallel = dict(
    context_parallel_size=1,
)

EXPERIMENT_NAME = "predict2_lora_training_2b_world_arena_video_only_10tasks"

experiment_config = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="video_only_lora_10tasks_480x640",
    ),
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=500,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=experiment_config,
)
