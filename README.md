# CLRNet × HG-Lane 평가 확장

[CLRNet](https://github.com/Turoad/CLRNet)(CVPR 2022)으로 [HG-Lane](https://github.com/zdc233/HG-Lane) 악천후 차선 벤치마크(`CULane_6x5k`)를 평가하는 확장 코드입니다. (ai_programming Team Project)

> **CLRNet 저장소가 필요합니다.** 이 저장소는 단독 실행 코드가 아니라 CLRNet 루트에 복사해 쓰는 확장입니다. CLRNet의 `CULane`·`CLRHead`·`culane_metric`·`main.py`와 빌드된 CUDA NMS를 그대로 사용합니다.

- **CLRNet 원본 파일은 하나도 수정하지 않습니다.** 새 파일만 CLRNet 루트에 추가합니다.
- 모델, 전처리, 추론, NMS, 공식 평가(`culane_metric`)는 CULane 평가와 **완전히 같습니다.**
- 원래 출력(예측 `.lines.txt`)에 더해, 날씨별 지표와 차선별 좌표·신뢰도를 담은 `hglane_results.json`을 만듭니다.

> 기준 버전: Turoad/CLRNet `7269e9d` (main)

---

## 1. 원본 CLRNet 대비 변경 사항

### 추가한 파일 (수정한 원본 파일: 없음)

| 파일 | 역할 |
|---|---|
| `main_hglane.py` | 실행 진입점. `hglane` 패키지를 import해 registry에 등록한 뒤 원본 `main.main()`을 그대로 실행 (인자 동일) |
| `hglane/__init__.py` | import 시 `HGLane`(데이터셋), `HGCLRHead`(헤드)를 CLRNet registry에 등록 |
| `hglane/hglane_dataset.py` | `HGLane(CULane)`: 데이터셋 클래스 (아래 참고) |
| `hglane/hglane_head.py` | `HGCLRHead(CLRHead)`: 차선별 신뢰도 기록 (아래 참고) |
| `hglane/clr_resnet18_hglane.py` | config. `_base_ = configs/clrnet/clr_resnet18_culane.py`를 상속하고 head·dataset 종류와 경로만 바꿈 |
| `test_hglane_local.py` | GPU 없이 하는 로컬 검증 (선택) |

### 원본과 다르게 동작하는 부분

HG-Lane은 이미지(1640×590)와 라벨(`.lines.txt`) 포맷이 CULane과 같아서 `CULane` 클래스를 그대로 상속합니다. 원본을 그대로 쓰면 안 되는 이유는 데이터가 아니라 **CULane 전용으로 박혀 있는 코드 세 군데**입니다.

| # | 원본 CLRNet | 문제 | HG-Lane 확장 |
|---|---|---|---|
| ① | `culane.py`의 `CATEGORYS`에 CULane 카테고리 9개(crowd, hlight …)가 고정 | 추론이 끝난 뒤 `test_split/test1_crowd.txt`를 열려다 **FileNotFoundError** | HG-Lane 카테고리 6개 (normal, night, dusk, rain, snow, fog) |
| ② | `cache/culane_{split}.pkl`이 있으면 목록을 다시 읽지 않음 | CULane을 평가한 뒤 돌리면 **CULane 목록으로 조용히 평가됨** | 캐시 미사용 (6,000장이라 매번 읽어도 몇 초) |
| ③ | `get_lanes`가 차선 점수로 필터링만 하고 버림. `metadata['conf']`는 softmax 이전 logit 하나 | 신뢰도 기반 시각화를 할 수 없음 | `HGCLRHead`가 `metadata`에 `score`, `logit_diff` 추가 (좌표·필터링·NMS는 원본과 같음) |

- `score` = 차선 후보 하나의 (배경, 차선) 2-class softmax = `sigmoid(logit_diff)`. 원본 `get_lanes`가 `conf_threshold` 필터링에 쓰는 값과 같습니다. 후보마다 따로 계산되므로 다른 차선의 영향을 받지 않습니다.
- `score`는 **보정(calibration)되지 않은 값**입니다. 보정이 필요하면 `logit_diff`로 temperature scaling을 하거나, JSON의 `correct` 플래그로 점수 구간별 정확도를 계산하세요.

---

## 2. 폴더 구조

```
$CLRNET_ROOT/                         # git clone https://github.com/Turoad/CLRNet
├── main.py                           # 원본 (CULane)
├── main_hglane.py                    # ★ 추가
├── test_hglane_local.py              # ★ 추가 (선택)
├── hglane/                           # ★ 추가
│   ├── __init__.py
│   ├── hglane_dataset.py
│   ├── hglane_head.py
│   └── clr_resnet18_hglane.py
├── clrnet/                           # 원본 패키지 (수정 없음)
├── configs/clrnet/clr_resnet18_culane.py   # 원본 (HG-Lane config가 상속)
├── weights/
│   └── culane_r18.pth                # CULane 사전학습 가중치 (CLRNet release)
├── data/
│   └── CULane_6x5k -> <HG-Lane 경로> # 심볼릭 링크
└── work_dirs/clr/r18_hglane/<실행시각>/   # 실행 결과 (자동 생성)
```

HG-Lane 데이터셋 (`CULane_6x5k`) 구조:

```
CULane_6x5k/
├── normal/ night/ dusk/ rain/ snow/ fog/   # 조건별 5,000장: <조건>_<번호>.jpg + .lines.txt
├── laneseg_label_w16/<조건>/*.png          # 세그멘테이션 라벨 (학습용, 테스트에는 안 씀)
└── list/
    ├── train.txt, train_gt.txt             # 21,000 (번호 0–3499 × 6조건)
    ├── val.txt, val_gt.txt                 # 3,000  (3500–3999)
    ├── test.txt, test_gt.txt               # 6,000  (4000–4999)
    └── test_split/test0_normal.txt test1_night.txt test2_dusk.txt
                  test3_rain.txt test4_snow.txt test5_fog.txt   # 조건별 1,000
```

---

## 3. 준비

### 3.1 CLRNet 설치
[CLRNet README](https://github.com/Turoad/CLRNet#installation)대로 환경을 만들고 `python setup.py build develop`까지 합니다. CULane 평가가 돌아가는 환경이면 추가로 설치할 것은 없습니다.

### 3.2 확장 파일 복사
```bash
cp -r <이 저장소>/main_hglane.py <이 저장소>/test_hglane_local.py <이 저장소>/hglane $CLRNET_ROOT/
```
`hglane/`은 반드시 `$CLRNET_ROOT` 바로 아래에 있어야 합니다. config의 `_base_`가 `../configs/...` 상대경로이기 때문입니다.

### 3.3 가중치
[CLRNet releases (models)](https://github.com/Turoad/CLRNet/releases/tag/models)에서 CULane 사전학습 가중치를 받습니다.
```bash
mkdir -p $CLRNET_ROOT/weights && cd $CLRNET_ROOT/weights
wget https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip
unzip culane_r18.pth.zip        # → culane_r18.pth
```
| 가중치 | config | CULane F1@50 (공식) |
|---|---|---|
| `culane_r18.pth` | `hglane/clr_resnet18_hglane.py` | 79.58 |

r34·r101·dla34를 쓰려면 같은 방식으로 `_base_`만 바꾼 config를 만들면 됩니다.

### 3.4 데이터
[HG-Lane](https://github.com/zdc233/HG-Lane)의 Dataset 링크에서 `CULane_6x5k`를 받아 풀고 링크를 겁니다.
```bash
cd $CLRNET_ROOT && mkdir -p data
ln -s /path/to/CULane_6x5k data/CULane_6x5k
```
경로를 직접 지정하려면 `hglane/clr_resnet18_hglane.py`의 `dataset_path`를 수정하세요.

---

## 4. 실행

```bash
cd $CLRNET_ROOT
python main_hglane.py hglane/clr_resnet18_hglane.py --test --load_from weights/culane_r18.pth --gpus 0
```
- 인자는 원본 `main.py`와 같습니다. `--validate`도 같은 HG-Lane test 목록으로 평가합니다. `--view`를 붙이면 시각화 이미지도 저장합니다.
- 비교용 CULane 평가는 원본 그대로 `python main.py configs/clrnet/clr_resnet18_culane.py --test --load_from weights/culane_r18.pth --gpus 0`로 합니다.

로그 끝부분 예:
```
normal  F1@50 xx.xx
night   F1@50 xx.xx
...
metric: 0.xxxx          # 전체 6,000장 F1@50
```

---

## 5. 출력

`work_dirs/clr/r18_hglane/<실행시각>/`
```
├── log.txt, config.py                 # CLRNet 기본 출력
├── normal/normal_4000.lines.txt ...   # 예측 차선 (CULane 포맷, 원본과 같음)
├── night/ dusk/ rain/ snow/ fog/
└── hglane_results.json                # ★ 추가
```

### `hglane_results.json`
```jsonc
{
  "meta": {
    "checkpoint": "weights/culane_r18.pth",
    "conf_threshold": 0.4, "nms_thres": 50, "iou_threshold": 0.5,
    "img_w": 1640, "img_h": 590, "created": "..."
  },
  "metrics": {
    "overall":    { "F1@50": {"TP","FP","FN","Precision","Recall","F1"}, "F1@55", ..., "F1@95", "mF1" },
    "categories": { "normal": {...}, "night": {...}, "dusk": {...},
                    "rain": {...}, "snow": {...}, "fog": {...} }        // 각각 F1@50 기준
  },
  "images": [
    {
      "img_name": "rain/rain_4000.jpg",
      "category": "rain",
      "visibility": 0.82,                  // 프레임 안 차선 score 평균 (차선이 없으면 0)
      "lanes": [
        {
          "score": 0.93,                   // 차선 신뢰도 (미보정)
          "logit_diff": 2.59,              // 차선 logit − 배경 logit
          "points": [[x, y], ...],         // 원본 해상도 픽셀, 아래→위 (.lines.txt와 같은 점)
          "matched_gt": 0,                 // 공식 평가와 같은 매칭으로 짝지은 GT 번호 (없으면 null)
          "iou": 0.81,
          "correct": true                  // iou > 0.5
        }
      ],
      "gt_lanes": [[[x, y], ...], ...]
    }
  ]
}
```
- 차선별 `correct` 합계는 공식 평가의 `F1@50.TP`와 같습니다.
- `conf_threshold`(0.4)보다 낮은 차선은 원본처럼 출력하지 않습니다. 낮은 점수 구간까지 분석하려면 config에 `test_parameters = dict(conf_threshold=0.1, nms_thres=50, nms_topk=4)`처럼 덮어써서 따로 실행하세요. 다만 이렇게 하면 F1이 베이스라인 조건과 달라집니다.

---

## 6. 로컬 검증 (GPU 불필요)

```bash
python -I test_hglane_local.py $CLRNET_ROOT [HG-Lane 경로]   # 기본값: $CLRNET_ROOT/data/CULane_6x5k
```
서버 전용 의존성(CUDA nms, mmcv 등)만 가짜 모듈로 대체하고, 실제 CLRNet 코드와 HG-Lane test 6,000장으로 다음을 확인합니다.
- `HGCLRHead`: 차선 좌표가 원본 `CLRHead`와 같음. `score`가 원본 필터링 점수와 같음. 후보마다 독립적으로 계산됨.
- `HGLane.evaluate`: GT를 예측으로 넣으면 F1 ≈ 1. 일부러 넣은 가짜 차선은 `correct=false`. 차선별 TP 합계가 공식 TP와 같음.
---

## 7. 주의사항

- **HG-Lane은 CULane 테스트 장면에서 만들어졌습니다.** HG-Lane normal은 CULane `test_split/test0_normal`에서 뽑은 원본 이미지이고, 나머지 5개 조건은 이를 diffusion으로 변환한 것입니다.
  - CULane **학습셋**과는 겹치지 않습니다. 라벨 98,555개와 대조해 일치 0건을 확인했습니다. 따라서 CULane 사전학습 가중치로 HG-Lane을 평가해도 학습 데이터가 새지 않습니다.
  - 반대로 **HG-Lane으로 파인튜닝한 모델을 CULane test로 평가하면 오염됩니다.**
- 같은 번호(`<조건>_N`)는 장면과 라벨이 같습니다. 분할을 새로 만들 때는 번호 단위로 나눠야 합니다.
- HG-Lane `test_split/*.txt`는 줄바꿈이 CRLF입니다. 평가 코드가 `rstrip()`으로 처리하므로 문제는 없습니다.
