"""원본 Runner + 학습 중 val F1@50 기준 best 체크포인트 저장.

원본 Runner.validate()는 metric을 로그에만 남기고 저장하지 않는다 (save_ckpt(is_best=True) 호출 없음).
→ 학습 중 검증 점수가 최고일 때 work_dir/ckpt/best.pth 저장. 나머지 동작은 원본과 동일.
main_hglane.py가 원본 main 모듈의 Runner를 이 클래스로 바꿔 끼운다.
"""
from clrnet.engine.runner import Runner


class HGRunner(Runner):
    def train(self):
        self._training = True  # --validate 단독 실행 때는 체크포인트를 만들지 않도록
        super().train()

    def validate(self):
        super().validate()
        metric = getattr(self.val_loader.dataset, 'last_metric', None)
        if getattr(self, '_training', False) and metric is not None and metric > self.metric:
            self.metric = metric
            self.save_ckpt(is_best=True)
            self.recorder.logger.info(
                f'best val F1@50 {metric:.4f} (epoch {self.recorder.epoch}) -> ckpt/best.pth')
