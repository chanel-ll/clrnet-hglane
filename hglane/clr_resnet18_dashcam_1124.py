# NOTE: keep this file ASCII-only (CLRNet Config reads it with the locale encoding).
#
# Same as clr_resnet18_dashcam.py, for the second dashcam set data/rainy_1124_culane
# (rain day x3 + rain night x1, no GT). Use with --test --load_from <any E0-E4 checkpoint>.
_base_ = './clr_resnet18_dashcam.py'

dataset_path = './data/rainy_1124_culane'
dataset = dict(test=dict(data_root=dataset_path))

work_dirs = 'work_dirs/clr/r18_dashcam_1124'
