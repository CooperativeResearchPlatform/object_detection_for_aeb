[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CooperativeResearchPlatform/object_detection_for_aeb/blob/main/notebooks/mmdetection_intro/mmdet3d_centerpoint.ipynb)

## Lokális fejlesztés

A saját MMDetection3D fork reprodukálható CPU-s vagy GPU-s VS Code környezetéhez használd a [Dev Container sablont](../../devcontainer/README.md). A CPU profil GPU nélkül is futtatja a `create_data.py` adat-előkészítést. A sablon egy paranccsal integrálható a forkba, a forrásfájlokat pedig a hostról mountolja, ezért minden módosítás közvetlenül a helyi repositoryban marad.

```bash
./devcontainer/install.sh /abszolút/útvonal/mmdetection3d
```