# MMDetection3D használati környezetek

Ez a kurzus az MMDetection3D `v1.4.0` API-ját használja. A csapat ugyanazt a forkot, commitot, Python-verziót, modellkonfigurációt és adatverziót rögzítse minden futtatási környezetben. A három támogatott munkamód ugyanazokat a `tools/create_data.py`, `tools/train.py` és `tools/test.py` belépési pontokat használja.

## Melyik környezetet válasszam?

| Környezet | Ajánlott használat | Előny | Korlát |
|---|---|---|---|
| Google Colab | első CenterPoint inference, nuScenes mini gyakorlat | nem igényel helyi telepítést, T4 GPU-val fut | ideiglenes tárhely, újracsatlakozás, futásidőkorlát |
| VS Code Dev Container | csapatmunka és reprodukálható fejlesztés | rögzített CPU/GPU stack, azonos környezet mindenkinél | Docker szükséges; a GPU profilhoz NVIDIA Container Toolkit kell |
| Helyi telepítés | saját GPU-s gép, ROS-integráció, célhardver | közvetlen hardver- és ROS-hozzáférés | a CUDA, PyTorch, MMCV és fordítóeszközök kompatibilitását nektek kell fenntartani |

## Google Colab

A [CenterPoint nuScenes notebook](../notebooks/mmdetection_intro/mmdet3d_centerpoint.ipynb) végigvezeti a hallgatót a Python 3.10-es környezet, a rögzített CUDA/PyTorch/MMCV stack, a nuScenes mini adatok, az előre tanított modell, az inference és a kiértékelés lépésein. A közvetlen Colab-link a [notebook rövid leírásában](../notebooks/mmdetection_intro/README.md) található.

1. Válaszd a **Runtime / Change runtime type / T4 GPU** beállítást.
2. A notebook elején állítsd a `REPO_URL` értékét a csapat forkjára és a `REPO_REF` értékét rögzített branchre, tagre vagy commitra.
3. Futtasd sorrendben a cellákat. A Python- és bináris stack telepítésekor két automatikus runtime-újraindítás történik.
4. A létrehozott checkpointokat és eredményeket még a Colab runtime törlése előtt mentsd tartós tárhelyre.

Colabban először az érintetlen, előre tanított modell működését igazoljátok. Saját adatkonverter vagy hosszabb tanítás fejlesztésére a Dev Container vagy a helyi telepítés alkalmasabb.

## VS Code Dev Container

A részletes előfeltételeket, CPU/GPU profilokat és hibakeresést a [Dev Container útmutató](../devcontainer/README.md) tartalmazza. A sablon a kurzus repository gyökeréből telepíthető a saját forkba:

```bash
./devcontainer/install.sh /abszolút/útvonal/mmdetection3d
```

A telepítő a `.devcontainer` könyvtár mellett egy `.vscode/launch.json.example` fájlt is elhelyez. Ebben nuScenes mini adat-előkészítési, CenterPoint tanítási és előre tanított modellt kiértékelő profil található. Aktiválás előtt másold `.vscode/launch.json` névre, majd ellenőrizd a config-, adat- és checkpoint-útvonalakat.

A CPU profil adatkonverzióra és általános fejlesztésre használható. A GPU profilt válaszd tanításhoz, teszteléshez és inferenciához.

## Helyi telepítés Linuxon vagy WSL2-ben

Helyi telepítésnél is külön virtuális környezetet használj. Az alábbi példa Python 3.10, PyTorch 2.1.0, CUDA 12.1, MMCV 2.1.0, MMEngine 0.10.7 és MMDetection 3.3.0 verziókat rögzít, összhangban a kurzus környezeteivel.

### Előfeltételek

- Linux vagy WSL2;
- Git, Python 3.10 fejlesztői csomagok és fordítóeszközök;
- tanításhoz CUDA-képes NVIDIA GPU és működő driver;
- legalább 16 GB szabad tárhely a környezeten felül; az adatok és checkpointok ennél lényegesen többet igényelhetnek.

Először ellenőrizd a drivert:

```bash
nvidia-smi
```

### Környezet létrehozása

```bash
git clone <a-csapat-forkjának-url-je> mmdetection3d
cd mmdetection3d
git checkout <rögzített-branch-tag-vagy-commit>

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<25" "setuptools==69.5.1" wheel packaging
```

CUDA 12.1-es GPU-környezethez:

```bash
python -m pip install \
  torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

Csak CPU-s adat-előkészítéshez:

```bash
python -m pip install \
  torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

Ezután telepítsd az OpenMMLab és adatkezelési függőségeket, majd a forkot editable módban:

```bash
python -m pip install \
  numpy==1.23.5 mmengine==0.10.7 mmdet==3.3.0 \
  nuscenes-devkit==1.2.0 lyft-dataset-sdk==0.0.8 \
  numba plyfile trimesh tensorboard
```

CUDA 12.1-es GPU-környezethez:

```bash
python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

CPU-s adat-előkészítő környezethez:

```bash
python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
```

Végül kösd be editable módban a forkot:

```bash
python -m pip install -v -e . --no-build-isolation
```

Ha a megfelelő wheel nem érhető el, a Dev Container CPU profilja a támogatott kerülőút. Más CUDA-verzió esetén ne csak a PyTorch csomagot cseréld le: a hozzá illő MMCV wheel-indexet is módosítani kell.

### Ellenőrzés

```bash
python -c "import torch, mmcv, mmdet, mmdet3d, mmengine; print(torch.__version__, mmcv.__version__, mmdet3d.__version__)"
python -c "from mmcv import ops; print(ops.nms)"
python tools/create_data.py --help
python tools/train.py --help
python tools/test.py --help
```

GPU-s használatnál ennek is igaznak kell lennie:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## Közös futtatási szabályok

- A `data/`, `checkpoints/` és `work_dirs/` nagy állományait ne commitoljátok Gitbe.
- Minden kísérlethez rögzítsétek a fork commitját, a configot, a checkpoint forrását, az adatverziót és a futtatási parancsot.
- A VS Code launch-konfiguráció ugyanazt a parancsot indítsa, mint amit terminálban dokumentáltatok.
- Tanítás előtt futtassátok a dataset-böngészőt, és jelenítsétek meg együtt a pontfelhőt és a ground-truth dobozokat.
- A Colab-, konténer- és helyi eredmények csak azonos config, checkpoint, adat split és evaluator mellett hasonlíthatók össze.

A saját dataset első végrehajtható ellenőrzése a hallgatók által létrehozott configgal:

```bash
python tools/misc/browse_dataset.py \
  configs/_base_/datasets/custom-lidar-3d.py \
  --task lidar_det \
  --output-dir work_dirs/custom_dataset_browse \
  --not-show
```

A config útvonala helykitöltő, az alap repositoryban nem létezik. Az elkészült képeken ellenőrizzétek, hogy a dobozok az objektumpontokra esnek-e, a dobozméret és yaw helyes-e, illetve nincs-e tengelycsere, tükrözés, skálahiba vagy hibás z-referenciapont. A parancs VS Code profilként is szerepel a telepített `.vscode/launch.json.example` fájlban.
