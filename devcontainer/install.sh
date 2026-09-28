#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_REPO="${1:-}"

if [[ -z "${TARGET_REPO}" ]]; then
    echo "Használat: $0 /útvonal/a/mmdetection3d-forkhoz" >&2
    exit 2
fi

TARGET_REPO="$(realpath "${TARGET_REPO}")"
if [[ ! -f "${TARGET_REPO}/setup.py" || ! -d "${TARGET_REPO}/mmdet3d" ]]; then
    echo "A cél nem MMDetection3D repository: ${TARGET_REPO}" >&2
    exit 2
fi

if [[ -e "${TARGET_REPO}/.devcontainer" ]]; then
    echo "Már létezik: ${TARGET_REPO}/.devcontainer" >&2
    echo "A meglévő konfigurációt nem írtuk felül." >&2
    exit 1
fi

cp -R "${SCRIPT_DIR}/template/.devcontainer" "${TARGET_REPO}/.devcontainer"
echo "Dev Container telepítve: ${TARGET_REPO}/.devcontainer"
echo "Nyisd meg a forkot VS Code-ban, majd válaszd a Dev Containers: Reopen in Container parancsot."
echo "GPU nélküli gépen válaszd a CPU - data preparation konfigurációt."