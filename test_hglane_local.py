"""로컬 검증 (GPU 불필요): 서버 전용 의존성(mmcv, CUDA nms 등)만 stub, 나머지는 실제 CLRNet 코드.

사용 (CLRNet 루트에서): python -I clrnet_hglane/test_hglane_local.py [CLRNet 경로] [HG-Lane 경로]
(기본값: CLRNet = 이 폴더의 상위, HG-Lane = <CLRNet>/data/CULane_6x5k)
- HGCLRHead: 좌표는 원본과 동일, score == get_lanes의 점수, 후보별 독립
- HGLane.evaluate: GT를 예측으로 넣어 F1≈1, 가짜 FP 차선은 correct=False, 차선별 TP 합 == 공식 TP
"""
import tempfile
import sys, json, logging, types
from unittest.mock import MagicMock
from pathlib import Path
HERE = Path(__file__).resolve().parent  # hglane/ 패키지가 있는 곳
CLRNET = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else CLRNET / 'data' / 'CULane_6x5k'
sys.path[:0] = [str(CLRNET), str(HERE)]
import importlib.abc, importlib.machinery
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIXES = ('mmcv', 'imgaug', 'p_tqdm', 'shapely', 'addict', 'clrnet.ops', 'timm')
    def find_spec(self, name, path, target=None):
        if name.startswith(self.PREFIXES):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__spec__ = spec; return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, _Stub())
logging.basicConfig(level=logging.INFO)
import numpy as np, torch
import hglane  # registry 등록
from hglane.hglane_head import HGCLRHead
from hglane.hglane_dataset import HGLane
from clrnet.models.heads.clr_head import CLRHead
from clrnet.utils.lane import Lane
from clrnet.datasets.registry import DATASETS
from clrnet.models.registry import HEADS

def test_head():
    assert HEADS.get('HGCLRHead') is HGCLRHead and DATASETS.get('HGLane') is HGLane
    cfg = types.SimpleNamespace(ori_img_h=590, ori_img_w=1640, cut_height=270)
    h = HGCLRHead.__new__(HGCLRHead); h.cfg = cfg; h.n_strips = 71
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

def test_dataset(out):
    cfg = types.SimpleNamespace(ori_img_h=590, ori_img_w=1640, cut_height=270, load_from='x.pth',
                                test_parameters=types.SimpleNamespace(conf_threshold=0.4, nms_thres=50))
    ds = HGLane(str(DATA), 'test', processes=[], cfg=cfg)
    assert len(ds.data_infos) == 6000
    preds = []
    for k, info in enumerate(ds.data_infos):  # GT를 예측으로 사용 → F1 ≈ 1 이어야 함
        lanes = []
        for j, lane in enumerate(info['lanes']):
            pts = np.array(lane, dtype=float) / [1640, 590]
            l = Lane(points=pts, metadata={'conf': 1.0, 'score': 0.9 - 0.1 * j, 'logit_diff': 2.0})
            lanes.append(l)
        if k % 10 == 0 and lanes:  # 일부 이미지에 가짜 FP 차선 추가
            lanes.append(Lane(points=np.array([[0.95, 0.6], [0.97, 0.8], [0.99, 1.0]]),
                              metadata={'conf': 0.0, 'score': 0.41, 'logit_diff': -0.3}))
        preds.append(lanes)
    f1 = ds.evaluate(preds, str(out))
    r = json.load(open(out / 'hglane_results.json', encoding='utf-8'))
    print('overall F1@50', f1, {c: round(v['F1'], 4) for c, v in r['metrics']['categories'].items()})
    print('overall keys', list(r['metrics']['overall']))
    assert f1 > 0.9 and len(r['images']) == 6000
    assert {i['category'] for i in r['images']} == {'normal', 'night', 'dusk', 'rain', 'snow', 'fog'}
    fake = [l for i in r['images'] for l in i['lanes'] if l['score'] == 0.41]
    real = [l for i in r['images'] for l in i['lanes'] if l['score'] != 0.41]
    print('fake lanes correct:', sum(l['correct'] for l in fake), '/', len(fake),
          '| real lanes correct:', sum(l['correct'] for l in real), '/', len(real))
    assert not any(l['correct'] for l in fake)
    # per-lane 매칭 집계가 공식 metric TP와 일치해야 함
    tp = sum(l['correct'] for l in real + fake)
    assert tp == r['metrics']['overall']['F1@50']['TP'], (tp, r['metrics']['overall']['F1@50']['TP'])
    print('per-lane TP == official TP:', tp)
    print('sample image:', json.dumps(r['images'][0])[:400])

if __name__ == '__main__':
    test_head()
    test_dataset(Path(tempfile.mkdtemp()))
