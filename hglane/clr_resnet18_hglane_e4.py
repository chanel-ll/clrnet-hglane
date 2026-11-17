# NOTE: keep this file ASCII-only (CLRNet Config reads it with the locale encoding).
#
# E4 weather-invariant consistency fine-tuning (hglane/hglane_pair.py):
#   student = trainable model, input <cond>_N (lane loss as in E1)
#   teacher = frozen CULane model (backbone + FPN from --finetune_from), input normal_N
#   loss = lane loss + consistency_weight * mean over 3 FPN levels of (1 - cosine) per location
# Both images get the identical augmentation. Everything else is inherited from E1.
# Run with:  --finetune_from culane_r18.pth   (initializes the student AND builds the teacher)
_base_ = './clr_resnet18_hglane_ft.py'

net = dict(type='HGConsistencyDetector')
consistency_weight = 1.0  # TEAM DECISION (lambda); cons_loss is logged next to the lane losses

dataset = dict(train=dict(type='HGLanePair'))  # adds batch['img_ref'] (paired normal_N)

work_dirs = 'work_dirs/clr/r18_hglane_e4'
