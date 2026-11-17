"""E4 날씨 불변 일관성 학습: 같은 번호 normal 이미지 쌍 데이터셋 + 교사/학생 Detector.

HG-Lane은 normal_N을 날씨만 바꿔 <조건>_N을 생성 → 차선 위치·장면 구도가 같다.
- HGLanePair: 학습 샘플 <조건>_N 과 normal_N 을 **같은 증강**으로 함께 반환 (batch['img_ref'])
- HGConsistencyDetector: 학생(학습) = <조건>_N 입력, 교사(고정 CULane 모델) = normal_N 입력.
  loss = 원본 차선 손실 + consistency_weight * mean_level(1 - cos(학생 FPN 특징, 교사 FPN 특징))
  (FPN 3개 레벨, 위치별 채널 벡터의 cosine. normal 샘플은 교사와 같은 이미지 → 망각 방지 역할)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clrnet.datasets.process.generate_lane_line import GenerateLaneLine
from clrnet.datasets.registry import DATASETS
from clrnet.models.nets.detector import Detector
from clrnet.models.registry import NETS, build_backbones, build_necks

from .hglane_dataset import HGLane


class _SameAug:
    """GenerateLaneLine.transform 대체: 호출마다 증강 파라미터를 고정(to_deterministic)해서 기억.
    GenerateLaneLine은 실패 시 최대 30번 다시 뽑으므로, 마지막(성공한) 호출의 증강이 last에 남는다."""

    def __init__(self, aug):
        self.aug, self.last = aug, None

    def __call__(self, **kwargs):
        self.last = self.aug.to_deterministic()
        return self.last(**kwargs)


@DATASETS.register_module
class HGLanePair(HGLane):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        gens = [p for p in self.processes.processes if isinstance(p, GenerateLaneLine)]
        assert len(gens) == 1, 'train_process에 GenerateLaneLine이 하나여야 함'
        self._aug = gens[0].transform = _SameAug(gens[0].transform)
        self._checked = False  # worker마다 한 번: normal 샘플로 "같은 증강" 실제 동작 확인

    def ref_path(self, img_name):
        """'snow/snow_123.jpg' -> <data_root>/normal/normal_123.jpg"""
        return f'{self.data_root}/normal/normal_{img_name.rsplit("_", 1)[1]}'

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not self.training:
            return sample
        info = self.data_infos[idx]
        is_normal = info['img_name'].split('/')[0] == 'normal'
        if is_normal and self._checked:
            sample['img_ref'] = sample['img']  # 자기 자신이 짝
            return sample
        ref = cv2.imread(self.ref_path(info['img_name']))[self.cfg.cut_height:, :, :]
        ref = self._aug.last(image=ref)  # 같은 Resize/Flip/Affine/색 증강
        ref = torch.from_numpy(ref.astype(np.float32) / 255.).permute(2, 0, 1).contiguous()
        if is_normal:
            assert torch.equal(ref, sample['img']), 'E4: 쌍 이미지에 같은 증강이 적용되지 않음'
            self._checked = True
        sample['img_ref'] = ref
        return sample


@NETS.register_module
class HGConsistencyDetector(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.teacher = None
        if cfg.finetune_from:  # 학습 때만 교사 생성 (평가 --load_from 때는 학생만 사용, teacher.* 키는 무시됨)
            self.teacher = nn.ModuleDict(dict(backbone=build_backbones(cfg), neck=build_necks(cfg)))
            sd = torch.load(cfg.finetune_from, map_location='cpu')['net']
            sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
            self.teacher.load_state_dict({k: v for k, v in sd.items() if k.startswith(('backbone.', 'neck.'))})
            for p in self.teacher.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()  # 교사 BN 통계 고정
        return self

    def forward(self, batch):
        if not (self.training and self.teacher is not None):
            return super().forward(batch)
        fea = self.backbone(batch['img'])
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])
        fea = self.neck(fea)
        output = self.heads(fea, batch=batch)
        with torch.no_grad():
            ref = self.teacher['neck'](self.teacher['backbone'](batch['img_ref']))
        cons = sum((1 - F.cosine_similarity(s, t, dim=1)).mean() for s, t in zip(fea, ref)) / len(fea)
        cons = cons * self.cfg.consistency_weight
        output['loss'] = output['loss'] + cons
        output['loss_stats'].update(loss=output['loss'], cons_loss=cons)
        return output
