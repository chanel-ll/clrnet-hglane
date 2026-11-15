# NOTE: keep this file ASCII-only. CLRNet's Config reads it with the locale's
# default encoding, so non-ASCII comments can break on some systems.

# Inherit the CULane r18 config: model / preprocessing / test_parameters
# (conf_threshold 0.4, ...) stay identical to the CULane baseline.
# Path is relative to this file: <CLRNet>/clrnet_hglane/hglane/ -> <CLRNet>/configs/
_base_ = '../../configs/clrnet/clr_resnet18_culane.py'

heads = dict(type='HGCLRHead')  # other head args are merged from base

dataset_path = './data/CULane_6x5k'  # ln -s <HG-Lane path> data/CULane_6x5k
# base already built `dataset` with type='CULane', so override type directly
dataset = dict(
    val=dict(type='HGLane', data_root=dataset_path),
    test=dict(type='HGLane', data_root=dataset_path),
)

work_dirs = 'work_dirs/clr/r18_hglane'
