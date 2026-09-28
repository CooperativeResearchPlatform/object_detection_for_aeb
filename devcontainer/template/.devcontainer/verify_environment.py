import os
import subprocess

import mmcv
import mmdet
import mmdet3d
import mmengine
import torch
from mmcv import ops


device_profile = os.environ.get("MMDET3D_DEVICE", "cpu")
assert callable(ops.nms), "Az MMCV natív operátorai nem érhetők el."

if device_profile == "gpu":
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert torch.cuda.is_available(), "A PyTorch nem éri el az NVIDIA GPU-t."
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}: {gpu_name}")
else:
    assert not torch.cuda.is_available(), "A CPU profil nem használhat CUDA runtime-ot."
    print(f"PyTorch {torch.__version__}: CPU profil")

print(f"MMEngine {mmengine.__version__}, MMCV {mmcv.__version__}, MMDetection {mmdet.__version__}")
print(f"MMDetection3D {mmdet3d.__version__}: {mmdet3d.__file__}")
print("A fejlesztői környezet használatra kész.")