#!/usr/bin/bash
# E2 부분 파인튜닝 (백본 고정, neck+head 학습): CULane 사전학습 R18 → HG-Lane train
# 제출: sbatch clrnet_hglane/sbatch/train_r18_hglane_e2.sh
# 결과: work_dirs/clr/r18_hglane_e2/<시각>/ckpt/best.pth (val F1@50 최고), ckpt/14.pth (마지막)

#SBATCH -J clr-r18-HGLane_e2
#SBATCH -D /nas2/data/chan12/clrnet
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 1-0
#SBATCH -o sbatch/logs/clr-r18-HGLane_e2-%A.out
# 학습은 재큐되면 처음부터 다시 시작하므로 --requeue는 넣지 않음

set -u
pwd; hostname; date

source /nas2/data/chan12/anaconda3/etc/profile.d/conda.sh
conda activate clrnet
cd /nas2/data/chan12/clrnet
nvidia-smi

ls data/CULane_6x5k/list/train_gt.txt data/CULane_6x5k/list/val.txt || { echo "HG-Lane 데이터 없음"; exit 1; }

python clrnet_hglane/main_hglane.py clrnet_hglane/hglane/clr_resnet18_hglane_e2.py \
    --finetune_from culane_r18.pth --gpus 0
date
