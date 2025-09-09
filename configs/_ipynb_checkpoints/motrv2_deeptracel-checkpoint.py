# configs/motrv2_custom.py

# MOTRv2 Custom Dataset Training Config
# Adapted for your own MOTChallenge-style dataset + custom det_db_motrv2.json

# ---- PATHS ----
dataset_root = "Data/MyDataset/post_split"     # <-- change this
det_db_path  = "DeepTracel/Transformer_Based/MOTRv2-fork/det_db_motrv2.json"  # <-- change this

# ---- DATASET ----
train_dataset = dict(
    type='MOTChallengeDataset',
    ann_file=f"{dataset_root}/train",
    img_prefix=f"{dataset_root}/train"
)

val_dataset = dict(
    type='MOTChallengeDataset',
    ann_file=f"{dataset_root}/val",
    img_prefix=f"{dataset_root}/val"
)

# ---- DATALOADER ----
data = dict(
    samples_per_gpu=15,   # batch size per GPU
    workers_per_gpu=1, # changed by Ziyad
    train=train_dataset,
    val=val_dataset,
    test=val_dataset
)

# ---- MODEL ----
model = dict(
    type='MOTR',
    backbone=dict(
        type='ResNet50',
        pretrained=True
    ),
    transformer=dict(
        type='DeformableTransformer'
    ),
    num_classes=1,   # ⚠️ change if you have more than 1 class
    det_db=det_db_path
)

# ---- OPTIMIZER ----
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-4
)

lr_config = dict(
    policy='Step',
    step=[40]
)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=50
)

# ---- CHECKPOINTS & LOGS ----
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

work_dir = './work_dirs/motrv2_deeptracel.py'