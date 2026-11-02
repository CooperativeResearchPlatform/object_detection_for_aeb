"""
A 02_convert.py altal keszitett csomagot feltolti a Segments.ai-ra 
`pointcloud-cuboid-sequence` sample-kent

    data/custom_dataset/<run>/metadata.json + points/*.bin
        -> upload_asset()  (minden .bin -> URL)
        -> add_sample()    (frames lista az URL-ekkel es ego pozokkal)

Az API-kulcsot KORNYEZETI VALTOZOBOL vagy fajlbol olvassa, soha nem a kodbol.
A kulcsot sosem szabad commitolni.

Elokeszites (PowerShell):
    pip install segments-ai
    $env:SEGMENTS_API_KEY = "a_sajat_kulcsod"

Hasznalat - uj dataset letrehozasa es pilot feltoltese:
    python 04_upload.py data/custom_dataset/kocka_20261007 ^
        --dataset felhasznalonev/aeb_lidar_pilot --create

Meglevo datasetbe:
    python 04_upload.py data/custom_dataset/kocka_20261007 ^
        --dataset csapatnev/kozos_dataset

A feltoltott .bin URL-eket .upload_cache.json-ba menti, igy megszakadas utan
ujrafuttatva nem tolti fel megegyszer ugyanazt.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from segments import SegmentsClient
except ImportError:
    sys.exit("Hianyzik a csomag. Futtasd: pip install segments-ai")


# Az annotacios szabalyzattal EGYEZO sorrendben. A sorrend kotott: az
# MMDetection3D config `classes` listajanak ugyanez lesz a sorrendje.
DEFAULT_CATEGORIES = ["car", "pedestrian"]

ASSET_TYPE = "binary-xyzi"
TASK_TYPE = "pointcloud-cuboid-sequence"


def get_api_key(key_file):
    if key_file:
        p = Path(key_file)
        if not p.exists():
            sys.exit(f"[!] Nincs ilyen kulcsfajl: {p}")
        return p.read_text(encoding="utf-8").strip()

    key = os.environ.get("SEGMENTS_API_KEY", "").strip()
    if not key:
        sys.exit(
            "[!] Nincs API-kulcs.\n"
            "    PowerShell:  $env:SEGMENTS_API_KEY = \"a_kulcsod\"\n"
            "    vagy:        python 04_upload.py <mappa> --key-file kulcs.txt\n"
            "    A kulcsot a Segments.ai -> Account settings -> API keys alatt talalod.\n"
            "    A kulcsfajlt vedd fel a .gitignore-ba."
        )
    return key


def load_metadata(run_dir):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"[!] Nincs metadata.json itt: {run_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if not meta.get("frames"):
        sys.exit("[!] A metadata.json nem tartalmaz frame-eket.")

    if meta.get("point_format", {}).get("asset_type") != ASSET_TYPE:
        sys.exit(f"[!] A metadata.json asset_type erteke nem {ASSET_TYPE}.")

    return meta


def ensure_dataset(client, dataset_identifier, categories, description, create):
    try:
        ds = client.get_dataset(dataset_identifier)
        print(f"Meglevo dataset: {dataset_identifier}  (task_type={ds.task_type})")
        if str(ds.task_type) != TASK_TYPE and TASK_TYPE not in str(ds.task_type):
            print(f"  [!] FIGYELEM: a dataset task_type-ja nem {TASK_TYPE}.")
            print("      Sequence cimkezes es interpolacio csak azzal mukodik.")
        return
    except Exception:
        pass

    if not create:
        sys.exit(
            f"[!] A '{dataset_identifier}' dataset nem erheto el.\n"
            "    Ha ujat akarsz letrehozni, add hozza a --create kapcsolot.\n"
            "    Ha a csapat kozos datasetjebe toltenel, ellenorizd a nevet\n"
            "    (mindig 'felhasznalo/dataset' alakban) es a jogosultsagot."
        )

    name = dataset_identifier.split("/", 1)[-1]
    task_attributes = {
        "format_version": "0.1",
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(categories)],
    }
    print(f"Uj dataset letrehozasa: {dataset_identifier}")
    print(f"  osztalyok: {', '.join(categories)}")
    client.add_dataset(
        name=name,
        description=description,
        task_type=TASK_TYPE,
        task_attributes=task_attributes,
    )


def upload_frames(client, run_dir, meta, cache_path):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    frames = []
    total = len(meta["frames"])

    for n, fr in enumerate(meta["frames"], 1):
        bin_rel = fr["pcd_path"]
        bin_path = run_dir / bin_rel
        if not bin_path.exists():
            sys.exit(f"[!] Hianyzo pontfajl: {bin_path}")

        cache_key = f"{bin_rel}:{fr['sha256']}"
        url = cache.get(cache_key)

        if url:
            print(f"  [{n}/{total}] {bin_rel} - mar feltoltve")
        else:
            with open(bin_path, "rb") as f:
                asset = client.upload_asset(f, bin_path.name)
            url = asset.url
            cache[cache_key] = url
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            print(f"  [{n}/{total}] {bin_rel} -> feltoltve")

        frames.append({
            "pcd": {"url": url, "type": ASSET_TYPE},
            "name": fr["name"],
            "timestamp": fr["timestamp"],
            "ego_pose": fr["ego_pose"],
        })

    return frames


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a 02_convert.py altal keszitett mappa")
    ap.add_argument("--dataset", required=True,
                    help="cel dataset 'felhasznalo/dataset' alakban")
    ap.add_argument("--create", action="store_true",
                    help="hozza letre a datasetet, ha meg nincs")
    ap.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                    help="osztalyok vesszovel, KOTOTT sorrendben")
    ap.add_argument("--sample-name", default=None,
                    help="a sample neve; alapbol a mappa neve")
    ap.add_argument("--key-file", default=None,
                    help="API-kulcsot tartalmazo fajl (ird be a .gitignore-ba)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"[!] Nem mappa: {run_dir}")

    meta = load_metadata(run_dir)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    sample_name = args.sample_name or run_dir.name

    print(f"Meres        : {meta.get('dataset')}")
    print(f"Frame-ek     : {meta['num_frames']}")
    print(f"LiDAR topic  : {meta['source']['lidar_topic']}")
    print(f"Koord. frame : {meta['point_format']['coordinate_frame']}")
    print(f"Cel dataset  : {args.dataset}")
    print(f"Sample nev   : {sample_name}")
    print()

    client = SegmentsClient(get_api_key(args.key_file))

    ensure_dataset(client, args.dataset, categories,
                   description=f"AEB LiDAR objektumdetekcio - {meta.get('dataset')}",
                   create=args.create)

    print("\nPontfelhok feltoltese ...")
    cache_path = run_dir / ".upload_cache.json"
    frames = upload_frames(client, run_dir, meta, cache_path)

    print("\nSample letrehozasa ...")
    attributes = {"frames": frames}
    sample_metadata = {
        "mcap_file": meta["source"]["mcap_file"],
        "lidar_topic": meta["source"]["lidar_topic"],
        "lidar_frame_id": meta["source"]["lidar_frame_id"],
        "coordinate_frame": meta["point_format"]["coordinate_frame"],
        "intensity_source_field": meta["point_format"]["intensity_source_field"],
        "frame_stride": meta["source"]["frame_stride"],
        "run": meta.get("dataset"),
    }

    sample = client.add_sample(
        dataset_identifier=args.dataset,
        name=sample_name,
        attributes=attributes,
        metadata=sample_metadata,
    )

    print()
    print("=" * 60)
    print(f"Kesz: {len(frames)} frame feltoltve mint '{sample_name}'")
    print("=" * 60)
    print(f"  sample uuid : {getattr(sample, 'uuid', '?')}")
    print(f"  dataset     : https://segments.ai/{args.dataset}")
    print()
    print("Kovetkezo lepes a felületen:")
    print("  1. Settings -> Labeling -> kapcsold be a 3D rotation-t")
    print("  2. Nyisd meg a sample-t, Start labeling")
    print("  3. Eloszor az ego-motion interpolaciot ellenorizd (allo objektum")
    print("     ne vandoroljon frame-rol frame-re)")


if __name__ == "__main__":
    main()