FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX" \
    FORCE_CUDA=1 \
    VIRTUAL_ENV=/opt/mmdet3d-venv \
    PATH=/opt/mmdet3d-venv/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        openssh-client \
        sudo \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

# A verziók megegyeznek a kurzus CenterPoint notebookjának bináris stackjével.
# A NumPy 1.23 megőrzi a v1.4.0 által még használt np.long aliast.
RUN python -m pip install "pip<25" "setuptools==69.5.1" "packaging==24.1" wheel \
    && python -m pip install \
        "numpy==1.23.5" \
        "opencv-python==4.10.0.84" \
        "mmengine==0.10.7" \
    && python -m pip install \
        "mmcv==2.1.0" \
        --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    && python -m pip install \
        "mmdet==3.3.0" \
        "nuscenes-devkit==1.2.0" \
        "lyft-dataset-sdk==0.0.8" \
        "numba==0.57.1" \
        "scikit-image==0.21.0" \
        jupyterlab \
        ipykernel \
        open3d \
        plyfile \
        tensorboard \
        trimesh \
    && python -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && chown -R "${USERNAME}:${USERNAME}" "${VIRTUAL_ENV}" \
    && chmod -R a+rwX "${VIRTUAL_ENV}"

USER ${USERNAME}
WORKDIR /workspace/mmdetection3d