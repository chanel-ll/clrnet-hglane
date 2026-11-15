"""CLRHead + 차선별 신뢰도 점수.

원본 CLRHead.get_lanes는 softmax(cls)[1] >= conf_threshold 로 차선을 거른 뒤 그 점수를 버리고,
Lane.metadata['conf']에는 softmax 전 logit(lane[1])만 남긴다.
여기서는 필터링에 쓰인 바로 그 점수를 metadata['score']에 추가한다. 나머지 동작은 원본과 동일.

score는 차선 후보 하나의 (배경, 차선) 2-class softmax = sigmoid(차선 logit - 배경 logit).
후보끼리는 섞이지 않으므로 다른 검출 차선의 영향을 받지 않는다.
보정(temperature scaling 등)용으로 logit 차이도 metadata['logit_diff']에 남긴다.
"""
from clrnet.models.heads.clr_head import CLRHead
from clrnet.models.registry import HEADS


@HEADS.register_module
class HGCLRHead(CLRHead):
    def predictions_to_pred(self, predictions):
        lanes = []
        for p in predictions:
            score = float(p[:2].softmax(0)[1])  # get_lanes의 scores와 같은 계산 (후보 1개 안의 2-class)
            logit_diff = float(p[1] - p[0])
            out = super().predictions_to_pred(p[None])  # 점이 1개 이하인 차선은 원본처럼 버려짐
            if out:
                out[0].metadata.update(score=score, logit_diff=logit_diff)
                lanes += out
        return lanes
