"""로컬 검증 (GPU 불필요): 설치 안 된 서버 전용 의존성(mmcv, CUDA nms 등)만 stub, 나머지는 실제 CLRNet 코드.

사용 (CLRNet 루트에서): python clrnet_hglane/test_hglane_local.py [CLRNet 경로] [HG-Lane 경로]
(기본값: CLRNet = 이 폴더의 상위, HG-Lane = <CLRNet>/data/CULane_6x5k)
- HGCLRHead: 좌표는 원본과 동일, score == get_lanes의 점수, 후보별 독립
- HGLane test: GT를 예측으로 넣어 F1≈1, 가짜 FP 차선은 correct=False, 차선별 TP 합 == 공식 TP,
  경로 기반 카테고리 == 공식 test_split 목록
- HGLane val: 3,000장, hglane_results_val.json, last_metric
- HGLane train: 21,000장, 세그 라벨 파일 존재, lane_exist
- HGDetector(E2): 백본 requires_grad=False, train() 후에도 백본 BN은 eval
- E4: HGLanePair 짝 경로(같은 번호 normal), HGConsistencyDetector 교사 고정·일관성 손실
- HGRunner: 학습 중 val 점수가 오를 때만 best 저장
- config (addict 있을 때): ft/e2 상속·병합 결과
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent  # hglane/ 패키지가 있는 곳
CLRNET = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else CLRNET / 'data' / 'CULane_6x5k'
sys.path[:0] = [str(CLRNET), str(HERE)]

# 실제로 설치 안 된 것만 stub (서버에서는 아무것도 stub하지 않음)
_CANDIDATES = ('mmcv', 'imgaug', 'p_tqdm', 'shapely', 'addict', 'timm', 'pytorch_warmup', 'ptflops')
_MISSING = tuple(m for m in _CANDIDATES if importlib.util.find_spec(m) is None)
_NO_NMS = not list((CLRNET / 'clrnet' / 'ops').glob('nms_impl*'))  # CUDA nms 미빌드 (로컬)
if _NO_NMS:
    _MISSING += ('clrnet.ops',)


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if _MISSING and name.startswith(_MISSING):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__spec__ = spec; return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Stub())
logging.basicConfig(level=logging.INFO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import hglane  # noqa: E402  (registry 등록)
from hglane.hglane_dataset import CATEGORIES, HGLane  # noqa: E402
from hglane.hglane_head import HGCLRHead  # noqa: E402
from hglane.hglane_net import HGDetector  # noqa: E402
from hglane import hglane_pair  # noqa: E402
from hglane.hglane_runner import HGRunner  # noqa: E402
from clrnet.datasets.registry import DATASETS  # noqa: E402
from clrnet.engine.runner import Runner  # noqa: E402
from clrnet.models.heads.clr_head import CLRHead  # noqa: E402
from clrnet.models.nets.detector import Detector  # noqa: E402
from clrnet.models.registry import HEADS, NETS  # noqa: E402
from clrnet.utils.lane import Lane  # noqa: E402

CFG = types.SimpleNamespace(ori_img_h=590, ori_img_w=1640, cut_height=270, load_from='x.pth',
                            test_parameters=types.SimpleNamespace(conf_threshold=0.4, nms_thres=50))


def test_head():
    assert HEADS.get('HGCLRHead') is HGCLRHead and DATASETS.get('HGLane') is HGLane
    h = HGCLRHead.__new__(HGCLRHead); h.cfg = CFG; h.n_strips = 71
    h.prior_ys = torch.linspace(1, 0, 72)
    torch.manual_seed(0)
    P = torch.zeros(5, 6 + 72)
    P[:, :2] = torch.randn(5, 2) * 3
    P[:, 2] = 0.0; P[:, 3] = 0.5; P[:, 5] = 40
    P[:, 6:] = torch.linspace(0.2, 0.6, 5)[:, None] + torch.linspace(0, 0.1, 72)
    ref = torch.softmax(P[:, :2], dim=1)[:, 1]                     # get_lanes와 같은 점수
    lanes = h.predictions_to_pred(P.clone())
    orig = CLRHead.predictions_to_pred(h, P.clone())
    assert len(lanes) == len(orig) == 5
    for i, (a, b) in enumerate(zip(lanes, orig)):
        assert np.allclose(a.points, b.points)                        # 좌표 원본과 동일
        assert abs(a.metadata['score'] - float(ref[i])) < 1e-6
        assert abs(a.metadata['score'] - 1 / (1 + np.exp(-a.metadata['logit_diff']))) < 1e-6
    solo = h.predictions_to_pred(P[:1].clone())                        # 다른 후보 없어도 같은 점수
    assert abs(solo[0].metadata['score'] - lanes[0].metadata['score']) < 1e-9
    print('head OK', [round(l.metadata['score'], 3) for l in lanes])


def gt_predictions(ds, fake_every=10):
    """GT를 예측으로 사용 (F1 ≈ 1), fake_every장마다 가짜 FP 차선 추가"""
    preds = []
    for k, info in enumerate(ds.data_infos):
        lanes = [Lane(points=np.array(l, dtype=float) / [1640, 590],
                      metadata={'conf': 1.0, 'score': 0.9 - 0.1 * j, 'logit_diff': 2.0})
                 for j, l in enumerate(info['lanes'])]
        if k % fake_every == 0 and lanes:
            lanes.append(Lane(points=np.array([[0.95, 0.6], [0.97, 0.8], [0.99, 1.0]]),
                              metadata={'conf': 0.0, 'score': 0.41, 'logit_diff': -0.3}))
        preds.append(lanes)
    return preds


def test_dataset_test(out):
    ds = HGLane(str(DATA), 'test', processes=[], cfg=CFG)
    assert len(ds.data_infos) == 6000
    f1 = ds.evaluate(gt_predictions(ds), str(out))
    r = json.load(open(out / 'hglane_results.json', encoding='utf-8'))
    print('[test] F1@50', round(f1, 4), {c: round(v['F1'], 4) for c, v in r['metrics']['categories'].items()})
    assert f1 > 0.9 and len(r['images']) == 6000 and ds.last_metric == f1
    assert list(r['metrics']['categories']) == list(CATEGORIES)
    # 경로 기반 카테고리 목록 == 공식 test_split 목록
    for i, cate in enumerate(CATEGORIES):
        official = {l.strip().lstrip('/') for l in open(DATA / f'list/test_split/test{i}_{cate}.txt') if l.strip()}
        mine = {l.strip() for l in open(out / f'hglane_lists/test_{cate}.txt') if l.strip()}
        assert official == mine, cate
    fake = [l for i in r['images'] for l in i['lanes'] if l['score'] == 0.41]
    real = [l for i in r['images'] for l in i['lanes'] if l['score'] != 0.41]
    assert not any(l['correct'] for l in fake)
    tp = sum(l['correct'] for l in real + fake)
    assert tp == r['metrics']['overall']['F1@50']['TP'], (tp, r['metrics']['overall']['F1@50']['TP'])
    print(f'[test] categories == test_split, fake correct 0/{len(fake)}, per-lane TP == official TP: {tp}')


def test_dataset_val(out):
    ds = HGLane(str(DATA), 'val', processes=[], cfg=CFG)
    assert len(ds.data_infos) == 3000
    f1 = ds.evaluate(gt_predictions(ds), str(out))
    r = json.load(open(out / 'hglane_results_val.json', encoding='utf-8'))
    counts = {c: sum(i['category'] == c for i in r['images']) for c in CATEGORIES}
    print('[val] F1@50', round(f1, 4), counts)
    assert f1 > 0.9 and ds.last_metric == f1 and r['meta']['split'] == 'val'
    assert counts == {c: 500 for c in CATEGORIES}
    assert all(3500 <= int(i['img_name'].split('_')[-1][:-4]) < 4000 for i in r['images'])  # 번호 3500-3999


def test_dataset_train():
    ds = HGLane(str(DATA), 'train', processes=[], cfg=CFG)
    assert len(ds.data_infos) == 21000
    missing = [i['mask_path'] for i in ds.data_infos if not os.path.exists(i['mask_path'])]
    assert not missing, missing[:3]
    assert all(len(i['lane_exist']) == 4 for i in ds.data_infos)
    assert all(int(i['img_name'].split('_')[-1][:-4]) < 3500 for i in ds.data_infos)
    print(f'[train] 21000 infos, seg labels all exist, max_lanes {ds.max_lanes}')


def test_e2_net():
    assert NETS.get('HGDetector') is HGDetector

    def fake_init(self, cfg):  # 실제 백본/neck 대신 BN이 들어간 작은 모듈 (HGDetector 로직만 검증)
        nn.Module.__init__(self)
        self.cfg = cfg
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        self.aggregator = None
        self.neck = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.heads = nn.Linear(4, 2)

    orig, Detector.__init__ = Detector.__init__, fake_init
    try:
        net = HGDetector(cfg=None)
    finally:
        Detector.__init__ = orig
    assert not any(p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in list(net.neck.parameters()) + list(net.heads.parameters()))
    wrapper = nn.Sequential(net).train()  # DataParallel처럼 감싼 뒤 train()
    assert net.training and net.neck.training and not net.backbone.training
    bn = net.backbone[1]
    before = bn.running_mean.clone()
    net.backbone(torch.randn(2, 3, 8, 8) * 5 + 3)
    assert torch.equal(before, bn.running_mean)  # 백본 BN 통계 고정
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    w0 = net.backbone[0].weight.clone()
    net.heads(net.neck(net.backbone(torch.randn(2, 3, 8, 8))).mean((2, 3))).sum().backward()
    opt.step()
    assert torch.equal(w0, net.backbone[0].weight)  # 옵티마이저가 백본을 건드리지 않음
    print('[E2] backbone frozen (grad, BN stats, AdamW) OK')


def test_e4_pair():
    ds = hglane_pair.HGLanePair.__new__(hglane_pair.HGLanePair)
    ds.data_root = str(DATA)
    infos = HGLane(str(DATA), 'train', processes=[], cfg=CFG).data_infos
    refs = [ds.ref_path(i['img_name']) for i in infos]
    assert all(r.endswith('/normal/normal_' + i['img_name'].split('_')[-1]) for r, i in zip(refs, infos))
    assert all(os.path.exists(r) for r in set(refs)) and len(set(refs)) == 3500
    print('[E4] 21000 train -> 3500 paired normal images, all exist')


def test_e4_net():
    NetCls = hglane_pair.HGConsistencyDetector
    assert NETS.get('HGConsistencyDetector') is NetCls and DATASETS.get('HGLanePair') is hglane_pair.HGLanePair

    class Bb(nn.Module):  # 백본: 특징 3개 레벨 (list)
        def __init__(self):
            super().__init__(); self.c = nn.Conv2d(3, 4, 1); self.bn = nn.BatchNorm2d(4)
        def forward(self, x):
            f = self.bn(self.c(x)); return [f, f[..., ::2, ::2], f[..., ::4, ::4]]

    class Nk(nn.Module):  # FPN처럼 tuple 반환
        def __init__(self):
            super().__init__(); self.c = nn.Conv2d(4, 4, 1)
        def forward(self, xs):
            return tuple(self.c(x) for x in xs)

    class Hd(nn.Module):
        def __init__(self):
            super().__init__(); self.w = nn.Parameter(torch.ones(1))
        def forward(self, fea, batch=None):
            lane = sum(f.mean() for f in fea) * self.w
            return {'loss': lane, 'loss_stats': {'loss': lane}} if self.training else lane

    def fake_init(self, cfg):
        nn.Module.__init__(self)
        self.cfg, self.backbone, self.aggregator, self.neck, self.heads = cfg, Bb(), None, Nk(), Hd()

    torch.manual_seed(0)
    culane = nn.Module()
    fake_init(culane, None)
    ckpt = Path(tempfile.mkdtemp()) / 'culane.pth'
    torch.save({'net': {'module.' + k: v for k, v in nn.ModuleDict(
        dict(backbone=culane.backbone, neck=culane.neck, heads=culane.heads)).state_dict().items()}}, ckpt)

    orig = (Detector.__init__, hglane_pair.build_backbones, hglane_pair.build_necks)
    Detector.__init__, hglane_pair.build_backbones, hglane_pair.build_necks = fake_init, lambda c: Bb(), lambda c: Nk()
    try:
        cfg = types.SimpleNamespace(finetune_from=str(ckpt), consistency_weight=2.0)
        net = NetCls(cfg)
        test_net = NetCls(types.SimpleNamespace(finetune_from=None))
    finally:
        Detector.__init__, hglane_pair.build_backbones, hglane_pair.build_necks = orig
    assert test_net.teacher is None  # 평가용(--load_from)에는 교사 없음
    t_sd = net.teacher.state_dict()
    assert all(torch.equal(t_sd[k], v) for k, v in culane.backbone.state_dict().items() for k in ['backbone.' + k])
    assert not any(p.requires_grad for p in net.teacher.parameters())
    nn.Sequential(net).train()
    assert net.backbone.training and not net.teacher.training  # 교사 BN 고정

    net.load_state_dict({k: v for k, v in t_sd.items()}, strict=False)  # 학생 = 교사 (finetune_from 효과)
    img = torch.randn(2, 3, 8, 8)
    net.backbone.bn.eval()  # 학생 BN도 running 통계로 맞춰야 교사와 완전히 같아짐 (학습 중엔 배치 통계라 0이 아님)
    out = net({'img': img, 'img_ref': img})
    assert abs(float(out['loss_stats']['cons_loss'].detach())) < 1e-6  # 같은 모델·같은 이미지 → 일관성 0
    net.backbone.bn.train()
    out = net({'img': img + torch.randn_like(img), 'img_ref': img})
    cons = out['loss_stats']['cons_loss']
    assert float(cons) > 0 and torch.allclose(out['loss'], out['loss_stats']['loss'])
    t0 = [p.clone() for p in net.teacher.parameters()]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    w0 = net.backbone.c.weight.clone()
    out['loss'].backward(); opt.step()
    assert net.backbone.c.weight.grad is not None and not torch.equal(w0, net.backbone.c.weight)
    assert all(torch.equal(a, b) for a, b in zip(t0, net.teacher.parameters()))  # 교사는 안 변함
    net.eval()
    assert torch.is_tensor(net({'img': img}))  # 평가 경로는 원본 Detector.forward
    print(f'[E4] teacher frozen, cons_loss {float(cons):.4f} (weight 2.0), eval path = Detector OK')


def test_runner():
    r = HGRunner.__new__(HGRunner)
    r.metric = 0.
    r.val_loader = types.SimpleNamespace(dataset=types.SimpleNamespace(last_metric=None))
    r.recorder = types.SimpleNamespace(epoch=0, logger=logging.getLogger('test'))
    saved = []
    r.save_ckpt = lambda is_best=False: saved.append((r.recorder.epoch, is_best))
    orig, Runner.validate = Runner.validate, lambda self: None
    try:
        r.val_loader.dataset.last_metric = 0.8; r.validate()      # 학습 중이 아님 → 저장 안 함
        assert saved == []
        r._training = True
        for ep, m in enumerate([0.80, 0.85, 0.83, 0.86]):
            r.recorder.epoch = ep; r.val_loader.dataset.last_metric = m; r.validate()
    finally:
        Runner.validate = orig
    assert saved == [(0, True), (1, True), (3, True)] and r.metric == 0.86, saved
    print('[runner] best saved at epochs', [e for e, _ in saved])


def test_configs():
    if 'addict' in _MISSING:
        print('[config] addict 없음 → 건너뜀 (서버에서 실행하면 검사됨)')
        return
    if os.name == 'nt':  # Windows 전용 우회: 열린 임시파일 재오픈 불가 (Linux는 불필요)
        import functools
        tempfile.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete=False)
        tempfile.TemporaryDirectory = functools.partial(tempfile.TemporaryDirectory, ignore_cleanup_errors=True)
    from clrnet.utils.config import Config
    c = Config.fromfile(str(HERE / 'hglane/clr_resnet18_hglane_e2.py'))
    d = c.dataset
    assert (d.train.type, d.train.split, d.val.type, d.val.split, d.test.type, d.test.split) == \
        ('HGLane', 'train', 'HGLane', 'val', 'HGLane', 'test')
    assert len(d.train.processes) == 2 and len(d.val.processes) == 2       # 증강/전처리는 base에서 상속
    assert c.net.type == 'HGDetector' and c.heads.type == 'HGCLRHead' and c.heads.num_priors == 192
    assert c.total_iter == (21000 // 24) * 15 == c.scheduler.T_max and c.optimizer.lr == 1e-4
    assert c.work_dirs == 'work_dirs/clr/r18_hglane_e2' and c.test_parameters.conf_threshold == 0.4
    e1 = Config.fromfile(str(HERE / 'hglane/clr_resnet18_hglane_ft.py'))
    assert e1.net.type == 'Detector' and e1.dataset.val.split == 'val'
    e3 = Config.fromfile(str(HERE / 'hglane/clr_resnet18_hglane_e3.py'))
    assert e3.net.type == 'Detector' and e3.backbone.pretrained and e3.optimizer.lr == 6e-4
    assert (e3.epochs, e3.batch_size, e3.total_iter, e3.dataset.val.split) == (15, 24, c.total_iter, 'val')
    e4 = Config.fromfile(str(HERE / 'hglane/clr_resnet18_hglane_e4.py'))
    assert e4.net.type == 'HGConsistencyDetector' and e4.consistency_weight == 1.0 and e4.optimizer.lr == 1e-4
    assert (e4.dataset.train.type, e4.dataset.train.split, len(e4.dataset.train.processes)) == ('HGLanePair', 'train', 2)
    assert e4.dataset.val.type == 'HGLane' and e4.work_dirs == 'work_dirs/clr/r18_hglane_e4'
    print(f'[config] e2: net {c.net.type}, train/val/test splits OK, total_iter {c.total_iter}, lr {c.optimizer.lr}')


if __name__ == '__main__':
    print('stubbed:', _MISSING or 'none')
    test_head()
    test_dataset_test(Path(tempfile.mkdtemp()))
    test_dataset_val(Path(tempfile.mkdtemp()))
    test_dataset_train()
    test_e2_net()
    test_e4_pair()
    test_e4_net()
    test_runner()
    test_configs()
    print('ALL OK')
