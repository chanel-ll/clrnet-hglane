# torch를 clrnet보다 먼저 import해야 함: clrnet/ops/nms.py가 torch 없이 nms_impl.so를 로드하는데,
# 이 .so는 libc10.so(torch)에 의존 → torch가 먼저 로드돼 있지 않으면 ImportError. (원본 main.py도 torch를 먼저 import)
import torch  # noqa: F401

# import만으로 CLRNet registry에 HGLane(데이터셋), HGCLRHead(헤드), HGDetector(E2 넷) 등록
from . import hglane_dataset, hglane_head, hglane_net  # noqa: F401,E402
from .hglane_runner import HGRunner  # noqa: F401,E402
