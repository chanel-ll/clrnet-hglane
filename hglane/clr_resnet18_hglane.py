# CULane r18 config 상속 → 모델/전처리/test_parameters(conf 0.4 등)는 CULane 베이스라인과 동일
_base_ = '../configs/clrnet/clr_resnet18_culane.py'

heads = dict(type='HGCLRHead')  # 나머지 head 인자는 base에서 병합됨

dataset_path = './data/CULane_6x5k'  # CLRNet 관례: ln -s <HG-Lane 경로> data/CULane_6x5k
# base의 dataset dict는 이미 type='CULane'으로 만들어져 있으므로 type을 직접 덮어씀
dataset = dict(
    val=dict(type='HGLane', data_root=dataset_path),
    test=dict(type='HGLane', data_root=dataset_path),
)

work_dirs = 'work_dirs/clr/r18_hglane'
