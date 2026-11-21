# NOTE: keep this file ASCII-only (CLRNet Config reads it with the locale encoding).
#
# Inference on real dashcam frames converted by scripts/convert_dashcam_to_culane.py (CULane format,
# one folder per scene, empty .lines.txt = no GT -> logged F1 is meaningless; use the lanes/scores
# in hglane_results.json). Works with any E0-E4 checkpoint:  --test --load_from <ckpt>
_base_ = './clr_resnet18_hglane.py'

dataset_path = './data/rainy_drivescene_culane'
dataset = dict(test=dict(type='HGLane', data_root=dataset_path))

work_dirs = 'work_dirs/clr/r18_dashcam'
