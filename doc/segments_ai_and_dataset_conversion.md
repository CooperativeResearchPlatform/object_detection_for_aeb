# Segments.ai címkézés és saját adathalmaz konvertálása

Ez az útmutató egy folyamatban kezeli a LiDAR-adatok MCAP-felvételből történő kinyerését, Segments.ai felületén végzett címkézését és egy saját export MMDetection3D-kompatibilis, nuScenes-szerű info-fájlokra alakítását. A leírt saját formátum és konverter nem kész kurzusfeature: a csapatoknak kell megtervezniük, implementálniuk, tesztelniük és a saját forkjukban dokumentálniuk.

## MCAP-ből Segments.ai-ban használható adatok

### Elvárt MCAP-bemenet

Egy mérési menethez legalább az alábbi, közös időalapot használó ROS2 topicok szükségesek:

- pontfelhő: `sensor_msgs/msg/PointCloud2`;
- ego mozgás: `nav_msgs/msg/Odometry`, vagy indokolt esetben `geometry_msgs/msg/PoseStamped`;
- statikus LiDAR -> ego extrinsic transzformáció: TF-ből vagy külön kalibrációs fájlból.

A felvétellel együtt rögzíteni kell a topicneveket, a `frame_id` értékeket, a koordinátatengelyeket, a mértékegységeket, az időbélyeg forrását és a LiDAR pontmezőinek nevét, típusát és sorrendjét. A konverzió előtt ellenőrizzétek, hogy az MCAP valóban tartalmazza a kiválasztott topicokat és a teljes mérés alatt rendelkezésre áll-e ego póz.

### Kinyert pontfelhők

A Segments.ai `pointcloud-cuboid-sequence` mintához minden címkézendő frame egy külön bináris pontfájl legyen. A kurzushoz javasolt szerződés:

- fájlnév: folytonos vagy stabil azonosító, például `0000001.bin`;
- adattípus: little-endian `float32`;
- rekord: pontosan négy érték, `(x, y, z, intensity)`;
- elrendezés: egymás után tárolt pontrekordok, fejléc nélkül;
- Segments.ai asset type: `binary-xyzi`;
- koordinátarendszer: egységes ego/jármű frame, méterben.

Ha a `PointCloud2` más mezőket is tartalmaz, például `reflectivity`, `ambient`, `ring` vagy pontonkénti idő, döntsétek el és dokumentáljátok, melyik mezőből készül az `intensity`. A Segments.ai feltöltéshez használt `binary-xyzi` fájlban csak négy `float32` feature legyen. Az eredeti többletmezőket külön megőrizhetitek a későbbi modellinputhoz, de ne nevezzetek öt- vagy hatdimenziós rekordot `binary-xyzi`-nek.

A pontokat a rögzített LiDAR-frame-ből a dokumentált ego-frame-be kell vinni a kalibrált homogén transzformációval. Hardcode-olt forgatás vagy eltolás csak akkor fogadható el, ha az adott szenzor kalibrációjából származik és a csapat dokumentálta. Az átalakítás után ellenőrizzetek néhány pontfelhőt vizuálisan: az $x/y/z$ tengelyek, a talajsík, az előre irány és a skála legyen helyes.

### Ego mozgás és időszinkron

Minden pontfelhőhöz a pontfelhő szenzoridőbélyegére érvényes ego pózt kell rendelni. Az odometria pozícióját és orientációját közös globális vagy menetlokális koordinátarendszerben tároljátok:

```json
{
    "timestamp": 12.345,
    "position": {"x": 1.2, "y": -0.4, "z": 0.0},
    "heading": {"qx": 0.0, "qy": 0.0, "qz": 0.05, "qw": 0.9987}
}
```

Az orientációt normalizált `(qx, qy, qz, qw)` kvaternióként adjátok át; ne cseréljétek minden frame-en identitáskvaternióra. Ha menetlokális origót használtok, az első póz teljes merevtest-transzformációjának inverzével képezzétek a relatív pózokat. A globális pozíció komponensenkénti kivonása és a yaw egyszerű kivonása forgó vagy 3D mozgásnál nem általánosan helyes.

Eltérő LiDAR- és odometria-frekvenciánál idő szerint interpoláljátok a pozíciót, az orientációt pedig SLERP-pel, vagy dokumentált maximális időeltéréssel válasszátok a legközelebbi pózt. Dobjátok el azt a pontfelhőt, amelyhez nincs megfelelően közeli ego póz. Az időbélyeg egysége legyen egységes és a `metadata.json` fájlban dokumentált.

### Feltöltési csomag

A Segments.ai importáló kód számára a következő minimális kimenetet készítsétek el:

```text
data/custom_dataset/
├── metadata.json
└── points/
        ├── 0000001.bin
        └── ...
```

A `metadata.json` minden frame-hez tartalmazza a stabil azonosítót, a pontfájl relatív útvonalát, checksumját, időbélyegét és teljes ego pózát. A feltöltő egy sequence sample `frames` listáját hozza létre, amelynek elemei ilyen szerkezetűek:

```json
{
    "pcd": {"url": "<feltöltés után kapott URL>", "type": "binary-xyzi"},
    "name": "frame_0000001",
    "timestamp": 12.345,
    "ego_pose": {
        "position": {"x": 1.2, "y": -0.4, "z": 0.0},
        "heading": {"qx": 0.0, "qy": 0.0, "qz": 0.05, "qw": 0.9987}
    }
}
```

A dataset task type értéke `pointcloud-cuboid-sequence`. Először rövid, 10-20 frame-es pilot sequence-et töltsetek fel, és ellenőrizzétek az ego-motion alapú frame-interpoláció irányát és nagyságát. API-kulcsot fájlból vagy környezeti változóból olvassatok, és soha ne commitoljátok.

## Segments.ai használata

A [Segments.ai](https://segments.ai/) böngészős címkézőplatform; részletes funkcióleírása a [hivatalos dokumentációban](https://docs.segments.ai/) található.

1. Regisztrálj az egyetemi e-mail-címeddel. Az akadémiai hozzáférés jóváhagyása néhány órát vagy napot is igénybe vehet.
2. A bal felső Segments.ai menüből válaszd ki a megosztott adatkészletet, majd ellenőrizd a kategóriákat és a címkézési típust.
3. A **Start labeling** gombbal indítsd el a munkát.
4. A kamera mozgatásához használd a `Ctrl` + bal egérgombot, forgatásához a `Shift` + bal egérgombot.
5. A megjelenítési beállításoknál állítsd a pontméretet, az osztály szerinti színezést és a z-tartományt úgy, hogy a célobjektumok jól elkülönüljenek.
6. Idősor címkézésekor az **enable interpolation** átviheti és ego-mozgással transzformálhatja a címkéket a következő frame-re. Az interpolált címke csak kiindulópont: minden frame-en kézzel ellenőrizni és igazítani kell.
7. Minden új címkéhez válaszd ki a megfelelő osztályt, majd rendszeresen ments a **Save** gombbal.

A projekt kezdete előtt írjatok annotációs szabályzatot. Ebben szerepeljen az osztálylista, a koordinátarendszer, a dobozközép és a yaw definíciója, a takart vagy kevés pontból álló objektumok kezelése, valamint az a szabály, hogy egy objektum mikor kap vagy nem kap címkét.

3D objektumdetekcióhoz orientált 3D cuboid címkéket exportáljatok. A polyline címke sáv- vagy pályahatárhoz megfelelő, de nem alakítható veszteség nélkül nuScenes `sample_annotation` 3D dobozzá. Polyline exporthoz külön feladat, adatbetöltő és metrika szükséges.

A Segments.ai exportban a cuboid orientációja jellemzően kvaternió. A címkekonverter ezt normalizálja, majd a dokumentált tengelykonvenció szerint számítsa ki belőle a z tengely körüli yaw szöget. Ne használjon minden dobozhoz `yaw = 0` helyettesítő értéket. A `dimensions.x/y/z` mezőket se rendeljétek automatikusan `length/width/height` mezőkhöz: egy pilot cuboidon igazoljátok a tengelyek jelentését, majd a leképezést tesztben rögzítsétek.

Hozzáférési tokent, személyes adatot vagy nyers szenzoradatot ne commitoljatok a forkba. A címkék verzióját és az export dátumát viszont rögzítsétek.

## A hallgatók által létrehozandó saját struktúra

Egy használható saját export például a következő elemekből áll:

```text
data/custom_dataset/
├── metadata.json
├── points/
│   ├── 0000000.bin
│   └── ...
├── labels/
│   ├── 0000000.txt
│   └── ...
├── unlabeled_pc/
└── ImageSets/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

A fenti struktúra követelmény és tervezési minta, nem letölthető példaadat. A `metadata.json` minden mintához tartalmazzon stabil azonosítót, időbélyeget, scene vagy mérési menet azonosítót, teljes ego pózt, a pontfelhő relatív útvonalát, a címkefájl relatív útvonalát és opcionálisan a korábbi sweep-eket. A modellhez használt pontfájl feature-száma egyezzen a config `load_dim`/`use_dim` értékeivel; ez eltérhet a Segments.ai feltöltéshez készített négydimenziós `binary-xyzi` példánytól.

3D dobozos egyszerű szöveges címke esetén egy sor ajánlott alakja:

```text
x y z length width height yaw class_name vx vy
```

A `vx vy` elhagyható, ekkor a konverter `[0.0, 0.0]` értéket adhat, de ezt ne tekintsétek mért sebességnek. A fájlformátumot és mértékegységeket az adatkészlet mellett dokumentálni kell.

A train/val/test listák scene-eket vagy teljes mérési meneteket osszanak fel. Szomszédos frame-ek véletlen szétosztása adatszivárgást okoz.

## Mit jelent itt a „nuScenes formátum”?

Két külön formátumot kell megkülönböztetni:

- a **nyers nuScenes devkit-formátum** több összekapcsolt JSON táblából (`scene`, `sample`, `sample_data`, `ego_pose`, `sample_annotation`, kalibráció stb.) és szenzorfájlokból áll;
- az **MMDetection3D v2 info-formátum** `metainfo` és `data_list` mezőket tartalmazó `.pkl` fájl, amelyből a dataset osztály közvetlenül dolgozik.

Saját tanításhoz általában az MMDetection3D v2 info-fájl előállítása a kisebb és ellenőrizhetőbb feladat. Ettől az adat még nem válik hivatalos nuScenes adathalmazzá, és a `NuScenesMetric` sem fog automatikusan működni: a hivatalos evaluator valódi nuScenes tokeneket, devkit-táblákat és rögzített splitet vár. Saját adathoz saját dataset osztályt és megfelelő evaluatort használjatok.

## Mezők leképezése

| Saját adat | MMDetection3D info-mező | Megjegyzés |
|---|---|---|
| mintaazonosító | `sample_idx` | spliten belül egyedi és folytonos egész lehet |
| checksum vagy UUID | `token` | globálisan stabil és egyedi legyen |
| szenzoridő | `timestamp` | egyetlen dokumentált egység, célszerűen mikroszekundum |
| ego pozíció és yaw | `ego2global` | 4x4 homogén transzformáció |
| LiDAR extrinsic | `lidar_points.lidar2ego` | ne legyen identitásmátrix, ha a szenzor nem az ego origóban van |
| `.bin` relatív útvonal | `lidar_points.lidar_path` | az adatgyökérhez képest relatív |
| pontjellemzők száma | `lidar_points.num_pts_feats` | egyezzen a bináris fájl alakjával és a pipeline-nal |
| korábbi frame-ek | `lidar_sweeps` | útvonal, időbélyeg és helyes transzformáció minden sweephez |
| osztálynév | `instances[].bbox_label_3d` | központi `class_name -> id` leképezésből |
| 3D cuboid | `instances[].bbox_3d` | `(x, y, z, l, w, h, yaw)` a választott LiDAR-konvencióban |
| objektumsebesség | `instances[].velocity` | `(vx, vy)`, azonos koordinátarendszerben |

A dobozok `z` referenciapontját, tengelyirányait és yaw előjelét ne találgatással állítsátok be. A Segments.ai export konvencióját explicit transzformáljátok a config által várt LiDAR-koordinátarendszerbe. Ha a címkék globális koordinátában vannak, alkalmazzátok a globális -> ego -> LiDAR transzformációt. Identitásmátrix csak akkor helyes, ha ezt a szenzor- és ego-frame definíció ténylegesen indokolja.

## Új `create_data.py` belépési pont

A következő nevek szemléltető pszeudokódot jelölnek: ezek a függvények és fájlok nem léteznek az alap MMDetection3D repositoryban. A konverzió érdemi logikáját a csapat hozza létre például `tools/dataset_converters/custom_converter.py` modulban, a `tools/create_data.py` fájlban pedig egy vékony belépési függvényt és egy új dataset ágat adjon hozzá:

```python
from tools.dataset_converters import custom_converter


def custom_to_nuscenes_data_prep(root_path, info_prefix, out_dir, max_sweeps=10):
    custom_converter.create_custom_infos(
        root_path=root_path,
        info_prefix=info_prefix,
        out_dir=out_dir,
        max_sweeps=max_sweeps,
    )
    create_groundtruth_database(
        "CustomLidarDataset",
        root_path,
        info_prefix,
        f"{info_prefix}_infos_train.pkl",
    )
```

Az argumentumkezelés alján adjatok hozzá egy ágat:

```python
elif args.dataset == "custom_nuscenes":
    custom_to_nuscenes_data_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        out_dir=args.out_dir,
        max_sweeps=args.max_sweeps,
    )
```

A hallgatók által implementált `create_custom_infos` feladatai:

1. Beolvassa és sémával ellenőrzi a `metadata.json` fájlt.
2. Scene-szinten beolvassa a train/val/test listákat, és ellenőrzi, hogy diszjunktak-e.
3. Egyetlen központi osztályleképezést készít, amely a config `classes` sorrendjével azonos.
4. Ellenőrzi minden hivatkozott pont- és címkefájl létezését, a ponttömb oszthatóságát a feature-számmal, valamint a dobozméretek pozitivitását.
5. Felépíti az ego- és szenzortranszformációkat, majd a dobozokat és sebességeket a kívánt LiDAR-frame-be viszi.
6. Minden splithez létrehozza a `metainfo` és `data_list` szerkezetet, majd `mmengine.dump` segítségével kiírja az info-fájlt.

Minimális kimeneti szerkezet:

```python
payload = {
    "metainfo": {
        "categories": class_name_to_id,
        "dataset": "custom_lidar",
        "version": "1.0",
        "info_version": "1.1",
    },
    "data_list": records,
}
mmengine.dump(payload, out_dir / f"{info_prefix}_infos_train.pkl")
```

Egy rekord lényegi része:

```python
record = {
    "sample_idx": sample_index,
    "token": sample_token,
    "timestamp": timestamp_us,
    "ego2global": ego2global,
    "images": {},
    "lidar_points": {
        "num_pts_feats": point_feature_count,
        "lidar_path": point_path,
        "lidar2ego": lidar2ego,
    },
    "lidar_sweeps": sweeps,
    "instances": instances,
    "cam_instances": {},
}
```

Egy dobozos instance:

```python
instance = {
    "bbox_3d": [x, y, z, length, width, height, yaw],
    "bbox_label": class_id,
    "bbox_label_3d": class_id,
    "bbox_3d_isvalid": True,
    "velocity": [vx, vy],
    "num_lidar_pts": points_in_box,
    "num_radar_pts": 0,
}
```

A `num_lidar_pts` értéket számoljátok ki a dobozba eső pontokból; ne használjatok minden objektumra rögzített helyettesítő értéket. A `lidar_sweeps` listába csak olyan korábbi frame kerüljön, amelynek időeltérése és LiDAR -> aktuális LiDAR transzformációja helyes.

Példa futtatás:

```bash
python tools/create_data.py custom_nuscenes \
    --root-path data/custom_dataset \
    --out-dir data/custom_dataset \
    --extra-tag custom_lidar \
  --max-sweeps 5
```

## Kötelező ellenőrzések tanítás előtt

- A train, val és test scene/token halmazok páronként diszjunktak.
- Minden token egyedi, minden útvonal feloldható, és a ponttömb feature-száma helyes.
- Az osztályazonosítók megegyeznek a config `classes` sorrendjével.
- Futtassátok a `tools/misc/browse_dataset.py` programot a saját dataset-configgal. A pontfelhő és a ground-truth dobozok együtt, helyes helyen és orientációval jelenjenek meg; ellenőrizzétek, hogy a dobozok tartalmaznak-e objektumpontokat, nem tükrözöttek vagy 90/180 fokkal elfordultak-e, és a méret-/tengelykonvenció helyes-e.
- Egy 10-20 mintás overfit próba képes a loss érdemi csökkentésére.
- Saját validációhoz saját evaluator készül; a hivatalos nuScenes NDS/mAP csak hivatalos nuScenes struktúrán és spliten értelmezhető.

Példa terminálparancs; a config útvonala szándékosan helykitöltő, ezt a csapatnak kell létrehoznia:

```bash
python tools/misc/browse_dataset.py \
    configs/_base_/datasets/custom-lidar-3d.py \
    --task lidar_det \
    --output-dir work_dirs/custom_dataset_browse \
    --not-show
```

Az MMDetection3D eszköz neve ebben a verzióban `browse_dataset.py`, nem `browse_data.py`. Ugyanez a futtatás megtalálható a kurzus `.vscode/launch.json.example` fájljában is.
