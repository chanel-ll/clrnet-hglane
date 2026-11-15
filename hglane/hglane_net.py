"""E2 부분 파인튜닝용 Detector: 백본을 고정하고 neck과 head만 학습한다.

config에서 net = dict(type='HGDetector') 로 사용. 구조/가중치 키는 원본 Detector와 같아서
CULane 체크포인트를 그대로 --finetune_from / --load_from 할 수 있다.
"""
import logging

from clrnet.models.nets.detector import Detector
from clrnet.models.registry import NETS


@NETS.register_module
class HGDetector(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        for p in self.backbone.parameters():
            p.requires_grad = False  # 기울기 없음 → AdamW가 이 파라미터를 건너뜀 (weight decay도 안 걸림)
        n_all = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.getLogger(__name__).info(
            f'HGDetector: backbone frozen, trainable {n_train:,} / {n_all:,} params')

    def train(self, mode=True):
        super().train(mode)
        # requires_grad만 끄면 학습 모드에서 BN의 running mean/var가 계속 갱신됨 → 백본은 항상 eval 모드로 고정
        self.backbone.eval()
        return self
