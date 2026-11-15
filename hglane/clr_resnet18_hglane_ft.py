# NOTE: keep this file ASCII-only (CLRNet Config reads it with the locale encoding).
#
# Common fine-tuning config on HG-Lane train (= E1 full fine-tuning).
# E2 / E4 inherit this file so that data, epochs, batch, lr, schedule stay identical.
# Run with:  --finetune_from culane_r18.pth   (loads matching keys from the CULane model)
#
# Inherits clr_resnet18_hglane.py -> clr_resnet18_culane.py (model, augmentation, test params).
_base_ = './clr_resnet18_hglane.py'

dataset_path = './data/CULane_6x5k'
dataset = dict(
    train=dict(type='HGLane', data_root=dataset_path, split='train'),  # list/train_gt.txt (21,000)
    # base CULane config validates on split='test' -> model selection on test = leakage.
    # Use HG-Lane val (list/val.txt, 3,000) during training; HGRunner saves ckpt/best.pth by val F1@50.
    val=dict(type='HGLane', data_root=dataset_path, split='val'),
)

# ---- schedule: TEAM DECISION, keep identical across E1/E2/E3/E4 ----
epochs = 15
batch_size = 24
optimizer = dict(type='AdamW', lr=1e-4)  # CULane training used 6e-4; lower lr for fine-tuning
# base computes total_iter from CULane size (88880); recompute for HG-Lane train (21000)
total_iter = (21000 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)
eval_ep = 1        # validate on HG-Lane val every epoch
save_ep = epochs   # periodic ckpt only at the end (best.pth is saved by HGRunner)
workers = 8        # matches --cpus-per-gpu=8 in the sbatch script

work_dirs = 'work_dirs/clr/r18_hglane_e1'
