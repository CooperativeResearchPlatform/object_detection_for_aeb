# MMDetection3D fejlesztői konténer

Ez a sablon a hallgató saját MMDetection3D forkját nyitja meg egy rögzített CPU-s vagy GPU-s fejlesztői környezetben. A repository nincs az image-be másolva: a hoston lévő munkakönyvtár bind mounttal kerül a `/workspace/mmdetection3d` útvonalra. Emiatt a konténerben végzett szerkesztések, commitok és új fájlok közvetlenül a host repositoryban jelennek meg.

Két választható profil áll rendelkezésre:

- **CPU - data preparation:** nem igényel NVIDIA GPU-t vagy CUDA-t; alkalmas a `tools/create_data.py` futtatására, info-fájlok és ground-truth adatbázisok készítésére, notebookokra és általános kódfejlesztésre;
- **GPU - training and inference:** CUDA-s MMCV buildet használ tanításhoz és modell-inferenciához.

## Előfeltételek

- Linux vagy WSL2;
- Docker Engine vagy Docker Desktop;
- VS Code, valamint a **Dev Containers** bővítmény.

A GPU profilhoz ezen felül NVIDIA GPU, működő host driver és NVIDIA Container Toolkit szükséges. A CPU profil ezek nélkül is használható.

A GPU-s Docker-hozzáférést a hoston ezzel érdemes ellenőrizni:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## Integrálás egy forkba

Klónozd a saját forkodat, majd ebből a kurzus-repositoryból futtasd:

```bash
./devcontainer/install.sh /abszolút/útvonal/mmdetection3d
```

A telepítő kizárólag a `.devcontainer` könyvtárat másolja át, és meglévő konfigurációt nem ír felül. Ezután commitolható a forkban:

```bash
cd /abszolút/útvonal/mmdetection3d
git add .devcontainer
git commit -m "Add course development container"
```

Nyisd meg a fork gyökerét VS Code-ban, majd futtasd a **Dev Containers: Reopen in Container** parancsot. A VS Code felajánlja a CPU és GPU konfigurációt. GPU nélküli gépen válaszd az **MMDetection3D CPU - data preparation** profilt.

Az első build letölti az alapimage-et és telepíti a csomagokat. A konténer létrejötte után a `post-create.sh` egy `.pth` fájllal közvetlenül beköti a megnyitott forrást, majd ellenőrzi a PyTorch/MMCV környezetet. Ez elkerüli a régi `setup.py develop` telepítési és jogosultsági problémáit.

## Mindennapi használat

A Python interpreter útvonala `/opt/mmdet3d-venv/bin/python`. A VS Code ezt automatikusan kiválasztja terminálhoz, Python fájlokhoz és notebookokhoz. Gyors ellenőrzés:

```bash
python .devcontainer/verify_environment.py
python demo/pcd_demo.py --help
```

Példa nuScenes mini adat-előkészítésre CPU profillal:

```bash
python -m tools.create_data nuscenes \
	--root-path data/nuscenes \
	--out-dir data/nuscenes \
	--extra-tag nuscenes \
	--version v1.0-mini \
	--max-sweeps 5
```

Az adatkonverzió CPU-n lassabb lehet, de nem használ CUDA-t. A létrejövő fájlok a bind mount miatt a host `data/nuscenes` könyvtárában maradnak, így átadhatók a GPU-val dolgozó csapattagoknak.

A repository teljes tartalma a hostról van mountolva. A `data/`, `work_dirs/` és checkpoint fájlok ezért a konténer újraépítése után is megmaradnak, amennyiben a fork könyvtárán belül vannak.

Python-függőség módosítása után futtasd újra a **Dev Containers: Rebuild Container** parancsot. A forráskód módosításához nem kell újratelepítés vagy rebuild, mert a `.pth` bekötés közvetlenül a mountra mutat.

## Rögzített stack

- Python 3.10 (a PyTorch image része)
- PyTorch 2.1.0+cpu vagy PyTorch 2.1.0 + CUDA 12.1
- MMCV 2.1.0 CPU vagy CUDA build
- MMEngine 0.10.7
- MMDetection 3.3.0
- NumPy 1.23.5

A NumPy 1.23.5 szándékos: az MMDetection3D v1.4.0 néhány kódútvonala még a későbbi NumPy-verziókból eltávolított `np.long` aliast használja.

## Gyakori hibák

`could not select device driver ... [[gpu]]`: GPU profilt választottál, de nincs telepítve vagy konfigurálva az NVIDIA Container Toolkit. Válaszd a CPU profilt, ha csak adat-előkészítést végzel.

`A PyTorch nem éri el az NVIDIA GPU-t`: ellenőrizd a host drivert a fenti `docker run` paranccsal, majd építsd újra a konténert.

Eltérő MMDetection3D branch függőségi hibája: ez a sablon a kurzusban használt v1.4.0 API-hoz készült. Más főverzióhoz a `.devcontainer/Dockerfile.cpu` és `.devcontainer/Dockerfile.gpu` OpenMMLab verzióit együtt kell frissíteni.