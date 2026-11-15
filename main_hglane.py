"""HG-Lane 실행 진입점. 원본 main.py와 인자 동일.

HGLane / HGCLRHead를 registry에 등록한 뒤 원본 main()을 그대로 실행한다.
  python main_hglane.py hglane/clr_resnet18_hglane.py --test --load_from <culane_r18.pth> --gpus 0
"""
import hglane  # noqa: F401  (registry 등록)
from main import main

if __name__ == '__main__':
    main()
