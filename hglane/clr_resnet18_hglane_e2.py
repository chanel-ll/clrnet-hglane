# NOTE: keep this file ASCII-only (CLRNet Config reads it with the locale encoding).
#
# E2 partial fine-tuning: backbone frozen (weights + BatchNorm statistics), train neck + head only.
# Everything else (data, epochs, batch, lr, schedule) is inherited from the common
# fine-tuning config so E1 vs E2 differ only in what is trained.
# Run with:  --finetune_from culane_r18.pth
_base_ = './clr_resnet18_hglane_ft.py'

net = dict(type='HGDetector')  # hglane/hglane_net.py

work_dirs = 'work_dirs/clr/r18_hglane_e2'
