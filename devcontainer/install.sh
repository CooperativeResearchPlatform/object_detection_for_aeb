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
    echo "Már létezik, nem írtuk felül: ${TARGET_REPO}/.devcontainer" >&2
else
    cp -R "${SCRIPT_DIR}/template/.devcontainer" "${TARGET_REPO}/.devcontainer"
    echo "Dev Container telepítve: ${TARGET_REPO}/.devcontainer"
fi

mkdir -p "${TARGET_REPO}/.vscode"
if [[ -e "${TARGET_REPO}/.vscode/launch.json.example" ]]; then
    echo "Már létezik, nem írtuk felül: ${TARGET_REPO}/.vscode/launch.json.example" >&2
else
    cp "${SCRIPT_DIR}/template/.vscode/launch.json.example" \
        "${TARGET_REPO}/.vscode/launch.json.example"
    echo "VS Code launch példa telepítve: ${TARGET_REPO}/.vscode/launch.json.example"
fi

echo "Nyisd meg a forkot VS Code-ban, majd válaszd a Dev Containers: Reopen in Container parancsot."
echo "GPU nélküli gépen válaszd a CPU - data preparation konfigurációt."