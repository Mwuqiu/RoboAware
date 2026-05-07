_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 24  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = False

model = dict(
    type="DefaultSegmentorV2",
    num_classes=16,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2, 2),
        enc_channels=(32, 64, 128, 256, 512, 1024),
        enc_num_head=(2, 4, 8, 16, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256, 512),
        dec_num_head=(4, 4, 8, 16, 16),
        dec_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=False,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 2
eval_epoch = 1
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "RobotWinDataset"
data_root = "/home/wuqiu/segment_training_dataset_robotwin"
ignore_index = -1
names = [
    "fl_link1",
    "fl_link2",
    "fl_link3",
    "fl_link4",
    "fl_link5",
    "fl_link6",
    "fl_link7",
    "fl_link8",
    "fr_link1",
    "fr_link2",
    "fr_link3",
    "fr_link4",
    "fr_link5",
    "fr_link6",
    "fr_link7",
    "fr_link8",
]

data = dict(
    num_classes=16,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="training",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.005,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord",),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
        ignore_index=ignore_index,
    ),
)
