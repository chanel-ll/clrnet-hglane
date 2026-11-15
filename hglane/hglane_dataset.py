"""HG-Lane(CULane_6x5k) 데이터셋. 데이터 포맷은 CULane과 동일 → CULane 클래스를 상속.

원본 CULane과 다른 점 (원본 파일은 수정하지 않음):
1. CATEGORYS: CULane 9개 → HG-Lane 6개 (normal/night/dusk/rain/snow/fog)
2. 캐시 미사용: 원본은 cache/culane_{split}.pkl을 재사용해 CULane 목록이 섞임. 6천 장이라 매번 읽어도 빠름
3. evaluate: 원본과 같은 .lines.txt 출력 + 공식 metric 후, 시각화용 hglane_results.json 저장
   (카테고리별 지표, 이미지별 차선 좌표·신뢰도·GT 매칭 결과, 프레임 가시성 점수)
"""
import json
import os
import os.path as osp
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

import clrnet.utils.culane_metric as culane_metric
from clrnet.datasets.culane import CULane
from clrnet.datasets.registry import DATASETS

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'night': 'list/test_split/test1_night.txt',
    'dusk': 'list/test_split/test2_dusk.txt',
    'rain': 'list/test_split/test3_rain.txt',
    'snow': 'list/test_split/test4_snow.txt',
    'fog': 'list/test_split/test5_fog.txt',
}
IOU_THR = 0.5
IMG_SHAPE = (590, 1640, 3)


def match_lanes(pred, anno):
    """culane_metric.culane_metric(official=True)과 같은 매칭을 차선 단위로 반환.

    pred, anno: 차선별 [(x, y), ...]. 반환: pred 차선마다 (매칭된 GT 번호 or None, IoU).
    공식 metric처럼 점이 2개 미만인 예측 차선은 평가에서 제외(None, 0).
    """
    out = [(None, 0.0)] * len(pred)
    valid = [i for i, lane in enumerate(pred) if len(lane) >= 2]
    if not valid or not anno:
        return out
    ip = np.array([culane_metric.interp(pred[i], n=5) for i in valid], dtype=object)
    ia = np.array([culane_metric.interp(a, n=5) for a in anno], dtype=object)
    ious = culane_metric.discrete_cross_iou(ip, ia, width=30, img_shape=IMG_SHAPE)
    for r, c in zip(*linear_sum_assignment(1 - ious)):
        out[valid[r]] = (int(c), float(ious[r, c]))
    return out


def _metric_dict(ret):
    """eval_predictions 결과 {thr: {...}, 'mean': {...}} → {'F1@50': {...}, ..., 'mF1': {...}}"""
    return {('mF1' if k == 'mean' else f'F1@{int(round(k * 100))}'): v for k, v in ret.items()}


@DATASETS.register_module
class HGLane(CULane):
    def load_annotations(self):
        self.logger.info('Loading HG-Lane annotations (no cache)...')
        with open(self.list_path) as f:
            self.data_infos = [self.load_annotation(l.split()) for l in f if l.strip()]
        self.max_lanes = max(len(a['lanes']) for a in self.data_infos)

    def lane_points(self, lane):
        """get_prediction_string과 같은 샘플링 → 원본 해상도 픽셀 [[x, y], ...] (아래→위)"""
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        xs = lane(ys)
        m = (xs >= 0) & (xs < 1)
        xs, ys = xs[m][::-1] * self.cfg.ori_img_w, ys[m][::-1] * self.cfg.ori_img_h
        return [[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs, ys)]

    def evaluate(self, predictions, output_basedir):
        # 1) 원본과 동일한 예측 .lines.txt
        print('Generating prediction output...')
        for info, pred in zip(self.data_infos, predictions):
            out_dir = osp.join(output_basedir, osp.dirname(info['img_name']))
            os.makedirs(out_dir, exist_ok=True)
            with open(osp.join(out_dir, osp.basename(info['img_name'])[:-3] + 'lines.txt'), 'w') as f:
                f.write(self.get_prediction_string(pred))

        # 2) 공식 metric: 카테고리별 F1@50, 전체 F1@50~95
        categories = {}
        for cate, cate_file in CATEGORYS.items():
            ret = culane_metric.eval_predictions(output_basedir, self.data_root,
                                                 osp.join(self.data_root, cate_file),
                                                 iou_thresholds=[IOU_THR], official=True)
            categories[cate] = ret[IOU_THR]
        overall = culane_metric.eval_predictions(output_basedir, self.data_root, self.list_path,
                                                 iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                 official=True)

        # 3) 이미지별 시각화 데이터
        cate_of = {}
        for cate, cate_file in CATEGORYS.items():
            with open(osp.join(self.data_root, cate_file)) as f:
                for l in f:
                    if l.strip():
                        cate_of[l.strip().lstrip('/')] = cate

        images, jobs = [], []
        for info, pred in zip(self.data_infos, predictions):
            lanes = []
            for lane in pred:
                pts = self.lane_points(lane)
                if not pts:  # get_prediction_string도 빈 차선은 파일에 안 씀
                    continue
                score = lane.metadata.get('score')  # HGCLRHead를 써야 채워짐
                diff = lane.metadata.get('logit_diff')
                lanes.append({'score': None if score is None else round(score, 4),
                              'logit_diff': None if diff is None else round(diff, 4),
                              'points': pts})
            gt = culane_metric.load_culane_img_data(info['img_path'][:-3] + 'lines.txt')
            scores = [l['score'] for l in lanes if l['score'] is not None]
            images.append({
                'img_name': info['img_name'],
                'category': cate_of.get(info['img_name']),
                'visibility': round(float(np.mean(scores)), 4) if scores else 0.0,
                'lanes': lanes,
                'gt_lanes': [[[round(x, 2), round(y, 2)] for x, y in g] for g in gt],
            })
            jobs.append(([l['points'] for l in lanes], gt))

        with Pool() as p:
            matches = p.starmap(match_lanes, jobs)
        for img, match in zip(images, matches):
            for lane, (gt_idx, iou) in zip(img['lanes'], match):
                lane.update(matched_gt=gt_idx, iou=round(iou, 4), correct=iou > IOU_THR)

        if any(l['score'] is None for img in images for l in img['lanes']):
            self.logger.warning('score 없음: config의 heads.type을 HGCLRHead로 설정해야 신뢰도가 기록됨')

        result = {
            'meta': {
                'dataset': 'HG-Lane', 'data_root': self.data_root, 'list': self.list_path,
                'checkpoint': getattr(self.cfg, 'load_from', None),
                'conf_threshold': self.cfg.test_parameters.conf_threshold,
                'nms_thres': self.cfg.test_parameters.nms_thres,
                'iou_threshold': IOU_THR,
                'img_w': self.cfg.ori_img_w, 'img_h': self.cfg.ori_img_h,
                'score': 'sigmoid(logit_diff) = 후보별 2-class softmax. 미보정 값, conf_threshold 미만은 출력 안 됨',
                'logit_diff': '차선 logit - 배경 logit (temperature scaling 보정용)',
                'visibility': '프레임 내 차선 score 평균 (차선 없으면 0)',
                'created': datetime.now().isoformat(timespec='seconds'),
            },
            'metrics': {'overall': _metric_dict(overall), 'categories': categories},
            'images': images,
        }
        json_path = osp.join(output_basedir, 'hglane_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        self.logger.info(f'HG-Lane 결과 저장: {json_path}')
        for cate, r in categories.items():
            self.logger.info(f'{cate:7s} F1@50 {r["F1"] * 100:.2f}')

        return overall[IOU_THR]['F1']
