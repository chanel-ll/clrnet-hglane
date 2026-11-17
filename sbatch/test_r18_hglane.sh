#!/usr/bin/bash
# HG-Lane test 평가. 인자 없이 제출하면 E0 (CULane 사전학습 R18).
# 제출 (어느 폴더에서 해도 됨: -D 로 고정):
#   E0: sbatch clrnet_hglane/sbatch/test_r18_hglane.sh
#   E2: sbatch clrnet_hglane/sbatch/test_r18_hglane.sh \
#         clrnet_hglane/hglane/clr_resnet18_hglane_e2.py work_dirs/clr/r18_hglane_e2/<시각>/ckpt/best.pth
# 결과: config의 work_dirs/<새 시각>/hglane_results.json

#SBATCH -J clr-r18-HGLane_test
#SBATCH -D /nas2/data/chan12/clrnet
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 1-0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH -o sbatch/logs/clr-r18-HGLane_test-%A.out

set -u
CONFIG=${1:-clrnet_hglane/hglane/clr_resnet18_hglane.py}
CKPT=${2:-culane_r18.pth}
pwd; hostname; date
echo "config: $CONFIG"; echo "ckpt:   $CKPT"

# 시간 제한 5분 전 USR1 -> 재큐
trap 'echo "[USR1] $(date) requeue"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

source /nas2/data/chan12/anaconda3/etc/profile.d/conda.sh
conda activate clrnet
cd /nas2/data/chan12/clrnet
nvidia-smi

# 계산 노드에서 ceph 데이터(심볼릭 링크 대상)가 보이는지 확인
ls data/CULane_6x5k/list/test.txt || { echo "HG-Lane 데이터 없음: data/CULane_6x5k -> $(readlink data/CULane_6x5k)"; exit 1; }
ls "$CONFIG" "$CKPT" || exit 1

# 백그라운드 실행 + wait 해야 trap 이 바로 동작한다
python clrnet_hglane/main_hglane.py "$CONFIG" --test --load_from "$CKPT" --gpus 0 & wait $!
date
