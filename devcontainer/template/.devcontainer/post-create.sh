#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mmdetection3d

if [[ ! -f setup.py || ! -d mmdet3d ]]; then
    echo "A megnyitott mappa nem MMDetection3D repository." >&2
    exit 2
fi

python .devcontainer/link_source.py
python .devcontainer/verify_environment.py