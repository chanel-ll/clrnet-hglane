#!/usr/bin/bash
# HG-Lane 학습 → 끝나면 같은 잡에서 val best.pth로 test 평가까지.
# 제출 (-J가 로그 이름이 됨):
#   E3: sbatch -J clr-r18-HGLane_e3 clrnet_hglane/sbatch/train_r18_hglane.sh clrnet_hglane/hglane/clr_resnet18_hglane_e3.py
#   E4: sbatch -J clr-r18-HGLane_e4 clrnet_hglane/sbatch/train_r18_hglane.sh clrnet_hglane/hglane/clr_resnet18_hglane_e4.py --finetune_from culane_r18.pth
#   E1: sbatch -J clr-r18-HGLane_e1 clrnet_hglane/sbatch/train_r18_hglane.sh clrnet_hglane/hglane/clr_resnet18_hglane_ft.py --finetune_from culane_r18.pth
# 결과: <config work_dirs>/<학습 시각>/ckpt/best.pth, <config work_dirs>/<평가 시각>/hglane_results.json

#SBATCH -J clr-r18-HGLane_train
#SBATCH -D /nas2/data/chan12/clrnet
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 1-0
#SBATCH -o sbatch/logs/%x-%A.out
# 학습은 재큐되면 처음부터 다시 시작하므로 --requeue는 넣지 않음

set -u
CONFIG=$1; shift   # 나머지 인자는 학습에 그대로 전달 (예: --finetune_from culane_r18.pth)
pwd; hostname; date
echo "config: $CONFIG"; echo "args:   $*"

source /nas2/data/chan12/anaconda3/etc/profile.d/conda.sh
conda activate clrnet
cd /nas2/data/chan12/clrnet
nvidia-smi

ls data/CULane_6x5k/list/train_gt.txt data/CULane_6x5k/list/val.txt "$CONFIG" || exit 1
WORK=$(sed -n "s/^work_dirs *= *'\(.*\)'.*/\1/p" "$CONFIG" | tail -1)
[ -n "$WORK" ] || { echo "config에 work_dirs 없음"; exit 1; }

python clrnet_hglane/main_hglane.py "$CONFIG" "$@" --gpus 0 || { echo "학습 실패"; exit 1; }
date

BEST=$(ls -t "$WORK"/*/ckpt/best.pth | head -1)
echo "test ckpt: $BEST"
python clrnet_hglane/main_hglane.py "$CONFIG" --test --load_from "$BEST" --gpus 0
date
