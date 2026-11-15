"""HG-Lane 실행 진입점. 원본 main.py와 인자 동일 (학습 / --validate / --test).

CLRNet 루트 안에 clrnet_hglane/로 clone한 상태에서, CLRNet 루트에서 실행:
  평가:  python clrnet_hglane/main_hglane.py clrnet_hglane/hglane/clr_resnet18_hglane.py \
             --test --load_from culane_r18.pth --gpus 0
  학습:  python clrnet_hglane/main_hglane.py clrnet_hglane/hglane/clr_resnet18_hglane_e2.py \
             --finetune_from culane_r18.pth --gpus 0

HGLane / HGCLRHead / HGDetector를 registry에 등록하고, Runner를 HGRunner(val 기준 best 저장)로
바꿔 끼운 뒤 원본 main()을 그대로 실행한다.
"""
import sys
from pathlib import Path

CLRNET_ROOT = Path(__file__).resolve().parent.parent
# CLRNet은 ./.gitignore, ./data, work_dirs 등을 현재 폴더 기준으로 찾음
if Path.cwd().resolve() != CLRNET_ROOT:
    sys.exit(f'CLRNet 루트에서 실행하세요: cd {CLRNET_ROOT}')
sys.path.insert(1, str(CLRNET_ROOT))  # 원본 main.py import용 (hglane은 이 파일 옆이라 이미 경로에 있음)

import hglane  # noqa: E402  (registry 등록)
import main as clrnet_main  # noqa: E402

clrnet_main.Runner = hglane.HGRunner  # 원본 main()이 쓰는 Runner만 교체 (원본 파일은 그대로)

if __name__ == '__main__':
    clrnet_main.main()
