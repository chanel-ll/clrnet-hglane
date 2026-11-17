# NOTE: keep this file ASCII-only (CLRNet Config reads it with the locale encoding).
#
# E3 training from scratch: ImageNet-pretrained ResNet18 backbone (base backbone pretrained=True),
# neck + head randomly initialized, NO CULane weights. Same data / epochs / batch as E1.
# Run WITHOUT --finetune_from.
_base_ = './clr_resnet18_hglane_ft.py'

# lr: CLRNet's original CULane value for batch 24 (training from scratch, not fine-tuning).
# TEAM DECISION: set 1e-4 here instead if E3 must share E1's lr exactly.
optimizer = dict(type='AdamW', lr=6e-4)

work_dirs = 'work_dirs/clr/r18_hglane_e3'
