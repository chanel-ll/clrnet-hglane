# CLRNet × HG-Lane 평가·파인튜닝 확장

[CLRNet](https://github.com/Turoad/CLRNet)(CVPR 2022)으로 [HG-Lane](https://github.com/zdc233/HG-Lane) 악천후 차선 벤치마크(`CULane_6x5k`)를 평가하고 파인튜닝하는 확장 코드입니다. (ai_programming Team Project)

> **CLRNet 저장소가 필요합니다.** 이 저장소는 단독 실행 코드가 아니라, **CLRNet 루트 안에 `clrnet_hglane/`으로 clone해서** 쓰는 확장입니다. CLRNet의 `CULane`·`CLRHead`·`culane_metric`·`main.py`와 빌드된 CUDA NMS를 그대로 사용합니다.

- **CLRNet 원본 파일은 하나도 수정하지 않습니다.** CLRNet 루트에 `clrnet_hglane/` 폴더 하나만 추가되고, 업데이트는 `git pull`로 합니다.
- 모델, 전처리, 추론, NMS, 공식 평가(`culane_metric`)는 CULane 평가와 **완전히 같습니다.**
- 원래 출력(예측 `.lines.txt`)에 더해, 날씨별 지표와 차선별 좌표·신뢰도를 담은 `hglane_results.json`을 만듭니다.

> 기준 버전: Turoad/CLRNet `7269e9d` (main)

---

## 1. 원본 CLRNet 대비 변경 사항

### 추가한 파일 (수정한 원본 파일: 없음)

| 파일 (`clrnet_hglane/` 안) | 역할 |
|---|---|
| `main_hglane.py` | 실행 진입점. CLRNet 루트를 import 경로에 추가하고, `hglane`을 registry에 등록하고, 원본 `main` 모듈의 `Runner`를 `HGRunner`로 바꾼 뒤 원본 `main()`을 그대로 실행 (인자 동일). CLRNet 루트가 아닌 곳에서 실행하면 안내 메시지를 내고 종료 |
| `hglane/__init__.py` | `torch`를 먼저 import(아래 참고)한 뒤 `HGLane`·`HGCLRHead`·`HGDetector`·`HGLanePair`·`HGConsistencyDetector`를 CLRNet registry에 등록 |
| `hglane/hglane_dataset.py` | `HGLane(CULane)`: 데이터셋 클래스 (train / val / test) |
| `hglane/hglane_head.py` | `HGCLRHead(CLRHead)`: 차선별 신뢰도 기록 |
| `hglane/hglane_net.py` | `HGDetector(Detector)`: **E2 부분 파인튜닝**. 백본 고정 (가중치 + BatchNorm 통계) |
| `hglane/hglane_pair.py` | **E4** 날씨 불변 일관성 학습. `HGLanePair(HGLane)`: `<조건>_N`과 같은 번호 `normal_N`을 같은 증강으로 함께 반환. `HGConsistencyDetector(Detector)`: 고정된 CULane 교사(normal 입력)와 학생(악천후 입력)의 FPN 특징 일관성 손실 추가 |
| `hglane/hglane_runner.py` | `HGRunner(Runner)`: 학습 중 val F1@50이 최고일 때 `ckpt/best.pth` 저장 |
| `hglane/clr_resnet18_hglane.py` | 평가 config. `_base_ = '../../configs/clrnet/clr_resnet18_culane.py'`를 상속하고 head·dataset 종류와 경로만 바꿈 |
| `hglane/clr_resnet18_hglane_ft.py` | 공통 파인튜닝 config (= **E1** 전체 파인튜닝). HG-Lane train으로 학습, val로 검증 |
| `hglane/clr_resnet18_hglane_e2.py` | **E2** config. 공통 파인튜닝 config에서 `net`만 `HGDetector`로 바꿈 |
| `hglane/clr_resnet18_hglane_e3.py` | **E3** config. CULane 가중치 없이 ImageNet 백본에서 시작 (lr 6e-4) |
| `hglane/clr_resnet18_hglane_e4.py` | **E4** config. `net`·`dataset.train`만 바꾸고 `consistency_weight` 추가 |
| `hglane/clr_resnet18_dashcam.py` | 실제 블랙박스 영상 추론 config. `data/rainy_drivescene_culane` (상위 프로젝트의 `scripts/convert_dashcam_to_culane.py`로 변환, GT 없음 → F1 무의미, JSON의 차선·score만 사용). E0~E4 어떤 체크포인트든 `--test --load_from` |
| `sbatch/*.sh` | Slurm 제출 스크립트 (HG-Lane test 평가, 학습 + test 평가, E2 학습) |
| `test_hglane_local.py` | GPU 없이 하는 로컬 검증 (선택) |

config 파일은 모두 **ASCII만 사용**합니다. CLRNet Config가 locale 기본 인코딩으로 파일을 읽기 때문입니다.

### 원본과 다르게 동작하는 부분

HG-Lane은 이미지(1640×590)와 라벨(`.lines.txt`) 포맷이 CULane과 같아서 `CULane` 클래스를 그대로 상속합니다. 원본을 그대로 쓰면 안 되는 이유는 데이터가 아니라 **CULane 전용으로 박혀 있는 코드 세 군데**입니다.

| # | 원본 CLRNet | 문제 | HG-Lane 확장 |
|---|---|---|---|
| ① | `culane.py`의 `CATEGORYS`에 CULane 카테고리 9개(crowd, hlight …)가 고정 | 추론이 끝난 뒤 `test_split/test1_crowd.txt`를 열려다 **FileNotFoundError** | HG-Lane 카테고리 6개 (normal, night, dusk, rain, snow, fog) |
| ② | `cache/culane_{split}.pkl`이 있으면 목록을 다시 읽지 않음 | CULane을 평가한 뒤 돌리면 **CULane 목록으로 조용히 평가됨** | 캐시 미사용 (6,000장이라 매번 읽어도 몇 초) |
| ③ | `get_lanes`가 차선 점수로 필터링만 하고 버림. `metadata['conf']`는 softmax 이전 logit 하나 | 신뢰도 기반 시각화를 할 수 없음 | `HGCLRHead`가 `metadata`에 `score`, `logit_diff` 추가 (좌표·필터링·NMS는 원본과 같음) |

- `score` = 차선 후보 하나의 (배경, 차선) 2-class softmax = `sigmoid(logit_diff)`. 원본 `get_lanes`가 `conf_threshold` 필터링에 쓰는 값과 같습니다. 후보마다 따로 계산되므로 다른 차선의 영향을 받지 않습니다.
- `score`는 **보정(calibration)되지 않은 값**입니다. 보정이 필요하면 `logit_diff`로 temperature scaling을 하거나, JSON의 `correct` 플래그로 점수 구간별 정확도를 계산하세요.

학습(파인튜닝)에서는 CULane config를 그대로 물려받으면 다음 문제가 추가로 생깁니다.

| # | 원본 CLRNet | 문제 | HG-Lane 확장 |
|---|---|---|---|
| ④ | base config의 `val`이 `split='test'` | 학습 중 검증이 **test셋**으로 돌아 모델 선택에 test 누수 | `val`을 HG-Lane val(`list/val.txt`, 3,000장)로 지정. 카테고리는 파일 경로(`rain/…`)로 판별해서 val에서도 날씨별 점수 계산 |
| ⑤ | `Runner.validate()`는 점수를 로그에만 남김 | **best 체크포인트가 저장되지 않음** (`save_ep`마다 저장하는 것뿐) | `HGRunner`가 val F1@50 최고일 때 `ckpt/best.pth` 저장 |
| ⑥ | `total_iter = (88880 // batch) × epochs` | CULane 학습셋 크기로 계산된 cosine 스케줄이 HG-Lane(21,000장)과 안 맞음 | `total_iter`와 `scheduler.T_max`를 21,000 기준으로 다시 계산 |
| ⑦ | `clrnet/ops/nms.py`가 torch 없이 `nms_impl.so`를 로드 | `hglane`을 먼저 import하면 `ImportError: libc10.so` | `hglane/__init__.py`에서 `torch`를 먼저 import (원본 `main.py`와 같은 순서) |

---

## 2. 폴더 구조

```
$CLRNET_ROOT/                         # git clone https://github.com/Turoad/CLRNet (빌드 완료)
├── main.py                           # 원본 (CULane)
├── clrnet/                           # 원본 패키지 (수정 없음)
├── configs/clrnet/clr_resnet18_culane.py   # 원본 (HG-Lane config가 상속)
├── culane_r18.pth                    # CULane 사전학습 가중치 (위치는 자유, --load_from으로 지정)
├── data/
│   ├── CULane -> <CULane 경로>        # (기존 CULane 평가용)
│   └── CULane_6x5k -> <HG-Lane 경로>  # ★ 심볼릭 링크 추가
├── clrnet_hglane/                    # ★ 이 저장소 (git clone)
│   ├── main_hglane.py
│   ├── test_hglane_local.py
│   ├── README.md
│   ├── sbatch/test_r18_hglane.sh, train_r18_hglane.sh, train_r18_hglane_e2.sh
│   └── hglane/
│       ├── __init__.py
│       ├── hglane_dataset.py, hglane_head.py, hglane_net.py, hglane_pair.py, hglane_runner.py
│       └── clr_resnet18_hglane.py, _ft.py, _e2.py, _e3.py, _e4.py
└── work_dirs/clr/r18_hglane*/<실행시각>/  # 실행 결과 (자동 생성)
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

### 3.2 이 저장소 clone (CLRNet 루트 안에)
```bash
cd $CLRNET_ROOT
git clone https://github.com/chanel-ll/clrnet-hglane.git clrnet_hglane
echo "clrnet_hglane/" >> .git/info/exclude   # (선택) CLRNet 저장소의 git status에서 숨김. 원본 .gitignore는 그대로 둠
```
- 위치는 반드시 `$CLRNET_ROOT` 바로 아래여야 합니다. config의 `_base_`가 `../../configs/...` 상대경로이고, `main_hglane.py`가 상위 폴더를 CLRNet 루트로 씁니다. 이 문서의 명령은 폴더 이름이 `clrnet_hglane`이라고 가정합니다.
- 업데이트: `cd $CLRNET_ROOT/clrnet_hglane && git pull`

### 3.3 가중치
[CLRNet releases (models)](https://github.com/Turoad/CLRNet/releases/tag/models)에서 CULane 사전학습 가중치를 받습니다. CULane 평가에 쓰던 가중치가 있으면 그대로 쓰면 됩니다.
```bash
cd $CLRNET_ROOT
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
경로를 직접 지정하려면 `clrnet_hglane/hglane/clr_resnet18_hglane.py`의 `dataset_path`를 수정하세요.
`data/`는 CLRNet `.gitignore`에 들어 있어서, 실행할 때 `work_dir/code/`로 복사되지 않습니다. 데이터를 CLRNet 루트에 직접 두면 `.lines.txt`가 매번 복사되니 `data/` 링크 방식을 권장합니다.

---

## 4. 실행

**반드시 `$CLRNET_ROOT`에서 실행합니다.** CLRNet이 `./.gitignore`, `./data`, `work_dirs`를 현재 폴더 기준으로 찾기 때문입니다.
```bash
cd $CLRNET_ROOT
python clrnet_hglane/main_hglane.py clrnet_hglane/hglane/clr_resnet18_hglane.py \
    --test --load_from culane_r18.pth --gpus 0
```
- 인자는 원본 `main.py`와 같습니다. `--validate`도 같은 HG-Lane test 목록으로 평가합니다. `--view`를 붙이면 시각화 이미지도 저장합니다.
- 비교용 CULane 평가는 원본 그대로 `python main.py configs/clrnet/clr_resnet18_culane.py --test --load_from culane_r18.pth --gpus 0`로 합니다.

로그 끝부분 예:
```
[test] normal  F1@50 xx.xx
[test] night   F1@50 xx.xx
...
metric: 0.xxxx          # 전체 6,000장 F1@50
```

### 4.1 학습 (E1 / E2 / E3 / E4)

| 실험 | config | 학습 대상 |
|---|---|---|
| E1 전체 파인튜닝 | `clrnet_hglane/hglane/clr_resnet18_hglane_ft.py` | 전체 가중치 |
| E2 부분 파인튜닝 | `clrnet_hglane/hglane/clr_resnet18_hglane_e2.py` | neck + head (백본 고정) |
| E3 처음부터 | `clrnet_hglane/hglane/clr_resnet18_hglane_e3.py` | 전체 (ImageNet 백본, neck·head 랜덤 초기화). `--finetune_from` 없이 실행 |
| E4 일관성 | `clrnet_hglane/hglane/clr_resnet18_hglane_e4.py` | 전체 + 일관성 손실 (교사 = 고정 CULane 모델) |

모든 실험은 데이터·epoch·batch·스케줄이 같습니다(`_ft.py`에서 상속). E3만 lr이 6e-4(CLRNet 원래 값, 처음부터 학습)입니다.
- 기본값: 15 epoch, batch 24, AdamW lr 1e-4, cosine, 매 epoch val 검증
- 이 값들은 팀에서 정할 항목입니다. `_ft.py`만 고치면 E1·E2 모두 반영됩니다.

```bash
cd $CLRNET_ROOT
python clrnet_hglane/main_hglane.py clrnet_hglane/hglane/clr_resnet18_hglane_e2.py \
    --finetune_from culane_r18.pth --gpus 0
# Slurm: sbatch clrnet_hglane/sbatch/train_r18_hglane_e2.sh
```

`sbatch/train_r18_hglane.sh`는 학습이 끝나면 같은 잡에서 `best.pth`로 test 평가까지 합니다 (`-J`가 로그 이름).
```bash
sbatch -J clr-r18-HGLane_e3 clrnet_hglane/sbatch/train_r18_hglane.sh clrnet_hglane/hglane/clr_resnet18_hglane_e3.py
sbatch -J clr-r18-HGLane_e4 clrnet_hglane/sbatch/train_r18_hglane.sh clrnet_hglane/hglane/clr_resnet18_hglane_e4.py --finetune_from culane_r18.pth
```
E4 동작:
- 학습 샘플 `<조건>_N`마다 `normal_N`을 읽어 **같은 증강**(imgaug `to_deterministic`)을 적용합니다. worker마다 처음 normal 샘플에서 두 결과가 완전히 같은지 assert로 확인합니다.
- 교사 = `--finetune_from`의 백본 + FPN (고정, eval 모드). 학생 = 학습 중인 모델 (같은 가중치에서 시작).
- `loss = 차선 손실 + consistency_weight × mean(1 − cos(학생 FPN 특징(<조건>_N), 교사 FPN 특징(normal_N)))`. FPN 3개 레벨, 위치별 채널 벡터 cosine. 로그에 `cons_loss`로 찍힙니다.
- normal 샘플은 교사와 같은 이미지라서 맑은 날 성능을 유지하는 역할도 합니다. 평가(`--test --load_from`)에는 학생만 쓰고, 체크포인트의 `teacher.*` 키는 무시됩니다.
- `--finetune_from`: CULane 가중치에서 키가 일치하는 것만 불러옵니다 (CLRNet 원래 기능).
- 결과: `work_dirs/clr/r18_hglane_e2/<시각>/ckpt/best.pth` (val F1@50 최고), `ckpt/14.pth` (마지막 epoch), `hglane_results_val.json` (마지막 val 결과)
- E2 시작 시 로그에 `HGDetector: backbone frozen, trainable N / M params`가 찍힙니다.

학습한 모델을 test로 평가할 때는 같은 config에 `--test --load_from <ckpt/best.pth>`를 줍니다.
```bash
python clrnet_hglane/main_hglane.py clrnet_hglane/hglane/clr_resnet18_hglane_e2.py \
    --test --load_from work_dirs/clr/r18_hglane_e2/<시각>/ckpt/best.pth --gpus 0
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
    "checkpoint": "culane_r18.pth",
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
cd $CLRNET_ROOT
python clrnet_hglane/test_hglane_local.py [CLRNet 경로] [HG-Lane 경로]
# 기본값: CLRNet = clrnet_hglane/의 상위 폴더, HG-Lane = <CLRNet>/data/CULane_6x5k
```
설치되지 않은 의존성(CUDA nms, mmcv 등)만 가짜 모듈로 대체하고, 실제 CLRNet 코드와 HG-Lane 데이터로 다음을 확인합니다. 서버처럼 모두 설치된 환경에서는 아무것도 대체하지 않습니다.
- `HGCLRHead`: 차선 좌표가 원본 `CLRHead`와 같음. `score`가 원본 필터링 점수와 같음. 후보마다 독립적으로 계산됨.
- `HGLane` test: GT를 예측으로 넣으면 F1 ≈ 1. 일부러 넣은 가짜 차선은 `correct=false`. 차선별 TP 합계가 공식 TP와 같음. 경로로 판별한 카테고리가 공식 `test_split`과 같음.
- `HGLane` val / train: 3,000장(날씨별 500, 번호 3500–3999) / 21,000장(세그 라벨 모두 존재, 번호 0–3499).
- `HGDetector`: 백본에 기울기 없음, BN 통계 변화 없음, AdamW 한 스텝 후에도 백본 가중치 그대로.
- `HGRunner`: 학습 중 val 점수가 오를 때만 `best.pth` 저장.
- config (`addict`가 설치된 환경): E1/E2 상속·병합 결과 (split, net, total_iter, lr).
---

## 7. 주의사항

- **HG-Lane은 CULane 테스트 장면에서 만들어졌습니다.** HG-Lane normal은 CULane `test_split/test0_normal`에서 뽑은 원본 이미지이고, 나머지 5개 조건은 이를 diffusion으로 변환한 것입니다.
  - CULane **학습셋**과는 겹치지 않습니다. 라벨 98,555개와 대조해 일치 0건을 확인했습니다. 따라서 CULane 사전학습 가중치로 HG-Lane을 평가해도 학습 데이터가 새지 않습니다.
  - 반대로 **HG-Lane으로 파인튜닝한 모델을 CULane test로 평가하면 오염됩니다.**
- 같은 번호(`<조건>_N`)는 장면과 라벨이 같습니다. 분할을 새로 만들 때는 번호 단위로 나눠야 합니다.
- HG-Lane `test_split/*.txt`는 줄바꿈이 CRLF입니다. 평가 코드가 `rstrip()`으로 처리하므로 문제는 없습니다.
