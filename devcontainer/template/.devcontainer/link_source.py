import sysconfig
from pathlib import Path


source_path = Path("/workspace/mmdetection3d").resolve()
site_packages = Path(sysconfig.get_path("purelib"))
pth_path = site_packages / "mmdetection3d_source.pth"
pth_path.write_text(f"{source_path}\n", encoding="utf-8")

print(f"MMDetection3D forrás bekötve: {pth_path} -> {source_path}")