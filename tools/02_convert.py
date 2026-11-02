"""
MCAP -> binary-xyzi 

Egy .mcap mereshez eloallitja a Segments.ai `pointcloud-cuboid-sequence`
feltolteshez szukseges csomagot:

    <out>/<run_nev>/
    |-- metadata.json
    +-- points/
        |-- 0000000.bin
        +-- ...

  1. kiolvassa a valasztott LiDAR topic PointCloud2 uzeneteit (dinamikus
     mezo-parsing, a hardcode-olt offset hibalehetoseg nelkul)
  2. kiszuri az ervenytelen pontokat (NaN/inf es a (0,0,0) kitolto pontok)
  3. a /tf_static-bol olvasott kalibralt transzformacioval a LiDAR frame-bol
     a lexus3/base_link ego frame-be viszi a pontokat
  4. minden frame-hez a sajat szenzoridobelyegere interpolalt ego pozt rendel
     (pozicio linearisan, orientacio SLERP-pel), menetlokalis origoval
  5. kiirja a float32 (x, y, z, intensity) .bin fajlokat es a metadata.json-t
     sha256 checksummal

Futtatas (pilot, 20 frame):
    python 02_convert.py 20261007/kocka_20261007.mcap --max-frames 20 --ply

2 Hz-cel:
    python 02_convert.py 20261007/kocka_20261007.mcap --stride 5

Luminar helyett:
    python 02_convert.py <fajl> --lidar luminar
"""

import argparse
import hashlib
import json
import math
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("Hianyzik a numpy. Futtasd: pip install numpy")

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError:
    sys.exit("Hianyzik a csomag. Futtasd: pip install mcap mcap-ros2-support")


# ---------------------------------------------------------------- beallitasok

EGO_FRAME = "lexus3/base_link"
ODOM_TOPIC = "/novatel/oem7/odom"
TF_STATIC_TOPIC = "/tf_static"

LIDARS = {
    "ouster": {
        "topic": "/lexus3/os_center/points",
        # A leltar szerint a reflectivity kalibralt 1..255 tartomanyu, mig az
        # intensity mezo 5..5100 kozott szor szet. Cimkezeskor a reflectivity
        # ad hasznalhato szinezest, ezert ez lesz az intensity feature.
        "intensity_field": "reflectivity",
        "intensity_scale": 255.0,
    },
    "luminar": {
        "topic": "/luminar_pcl",
        # A Luminar intensity UINT8, de a mert tartomanya 0..2, tehat a
        # szinezes szinte lapos lesz. Nincs jobb mezo ebben a felhoben.
        "intensity_field": "intensity",
        "intensity_scale": 255.0,
    },
}

# sensor_msgs/PointField datatype -> numpy tipus
PF_NUMPY = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


# ------------------------------------------------------------ kvaternio mat.

def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Nulla hosszusagu kvaternio.")
    return q / n


def q_multiply(a, b):
    """Hamilton szorzat, (x, y, z, w) sorrendben."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def q_conjugate(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def q_to_matrix(q):
    x, y, z, w = q_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def q_slerp(q0, q1, t):
    """Gombi linearis interpolacio a rovidebb iv menten."""
    q0 = q_normalize(q0)
    q1 = q_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:          # rovidebb ut
        q1 = -q1
        dot = -dot
    if dot > 0.9995:       # majdnem egybeeso -> linearis eleg
        return q_normalize(q0 + t * (q1 - q0))
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta0 * t
    q2 = q_normalize(q1 - q0 * dot)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def q_yaw_deg(q):
    x, y, z, w = q_normalize(q)
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ------------------------------------------------------------------ beolvasas

def strip_slash(name):
    return name.lstrip("/")


def stamp_to_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def read_static_transform(mcap_path, lidar_frame):
    """A /tf_static-bol kikeresi az EGO_FRAME -> lidar_frame transzformaciot."""
    target = strip_slash(lidar_frame)
    found = {}
    for msg in read_ros2_messages(str(mcap_path), topics=[TF_STATIC_TOPIC]):
        for tr in msg.ros_msg.transforms:
            parent = strip_slash(tr.header.frame_id)
            child = strip_slash(tr.child_frame_id)
            found[(parent, child)] = tr.transform

    key = (strip_slash(EGO_FRAME), target)
    if key not in found:
        avail = "\n    ".join(f"{p} -> {c}" for p, c in sorted(found))
        raise SystemExit(
            f"[!] Nem talaltam a {EGO_FRAME} -> {lidar_frame} transzformaciot "
            f"a /tf_static-ban.\n  Elerheto parok:\n    {avail}"
        )

    tr = found[key]
    t = np.array([tr.translation.x, tr.translation.y, tr.translation.z])
    q = np.array([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])
    return t, q_normalize(q)


def read_odometry(mcap_path):
    """Visszaad (t[], pos[N,3], quat[N,4]) tombot a /novatel/oem7/odom-bol."""
    ts, pos, quat = [], [], []
    for msg in read_ros2_messages(str(mcap_path), topics=[ODOM_TOPIC]):
        o = msg.ros_msg
        ts.append(stamp_to_s(o.header.stamp))
        p = o.pose.pose.position
        q = o.pose.pose.orientation
        pos.append([p.x, p.y, p.z])
        quat.append([q.x, q.y, q.z, q.w])

    if not ts:
        raise SystemExit(
            "[!] Ebben a fajlban nincs /novatel/oem7/odom uzenet.\n"
            "    Ego poz nelkul a Segments.ai sequence interpolacio nem\n"
            "    hasznalhato, ezert ez a meres nem konvertalhato."
        )

    order = np.argsort(ts)
    return (np.asarray(ts)[order],
            np.asarray(pos)[order],
            np.asarray(quat)[order])


def interp_pose(ts, pos, quat, t_query, max_gap):
    """Ego poz a t_query idopontban. None, ha nincs eleg kozeli minta."""
    if t_query < ts[0] - max_gap or t_query > ts[-1] + max_gap:
        return None

    i = bisect_left(ts.tolist(), t_query)
    if i == 0:
        if abs(ts[0] - t_query) > max_gap:
            return None
        return pos[0].copy(), quat[0].copy()
    if i >= len(ts):
        if abs(ts[-1] - t_query) > max_gap:
            return None
        return pos[-1].copy(), quat[-1].copy()

    t0, t1 = ts[i - 1], ts[i]
    if (t_query - t0) > max_gap and (t1 - t_query) > max_gap:
        return None

    span = t1 - t0
    f = 0.0 if span <= 0 else (t_query - t0) / span
    p = pos[i - 1] + f * (pos[i] - pos[i - 1])
    q = q_slerp(quat[i - 1], quat[i], f)
    return p, q


# ----------------------------------------------------------- pontfelho parsol

def pointcloud_to_array(pc, intensity_field):
    """PointCloud2 -> (N,4) float32 [x, y, z, raw_intensity], szures nelkul."""
    by_name = {f.name: f for f in pc.fields}
    for need in ("x", "y", "z"):
        if need not in by_name:
            raise SystemExit(f"[!] Hianyzo mezo a pontfelhoben: {need}")
    if intensity_field not in by_name:
        have = ", ".join(sorted(by_name))
        raise SystemExit(
            f"[!] Nincs '{intensity_field}' mezo a pontfelhoben.\n"
            f"    Elerheto mezok: {have}"
        )

    names, formats, offsets = [], [], []
    for f in pc.fields:
        np_t = PF_NUMPY.get(f.datatype)
        if np_t is None or f.count != 1:
            continue
        names.append(f.name)
        formats.append(np.dtype(np_t).newbyteorder(">" if pc.is_bigendian else "<"))
        offsets.append(f.offset)

    dt = np.dtype({"names": names, "formats": formats,
                   "offsets": offsets, "itemsize": pc.point_step})
    n = pc.width * pc.height
    raw = np.frombuffer(bytes(pc.data), dtype=dt, count=n)

    out = np.empty((n, 4), dtype=np.float32)
    out[:, 0] = raw["x"]
    out[:, 1] = raw["y"]
    out[:, 2] = raw["z"]
    out[:, 3] = raw[intensity_field].astype(np.float32)
    return out


def clean_points(pts, min_range, max_range):
    """NaN/inf, (0,0,0) kitoltopontok es tartomanyon kivuli pontok kiszurese."""
    xyz = pts[:, :3]
    ok = np.isfinite(xyz).all(axis=1) & np.isfinite(pts[:, 3])
    r = np.linalg.norm(xyz, axis=1)
    ok &= r > min_range                       # (0,0,0) kitoltopontok is ide esnek
    if max_range > 0:
        ok &= r <= max_range
    return pts[ok]


def transform_to_ego(pts, t_le, q_le):
    """LiDAR frame -> ego (base_link) frame, kalibralt homogen transzformacio."""
    R = q_to_matrix(q_le)
    out = pts.copy()
    out[:, :3] = (pts[:, :3].astype(np.float64) @ R.T + t_le).astype(np.float32)
    return out


# ------------------------------------------------------------------- fo logika

def convert(args):
    mcap_path = Path(args.mcap)
    if not mcap_path.exists():
        sys.exit(f"[!] Nem letezik: {mcap_path}")

    cfg = LIDARS[args.lidar]
    topic = cfg["topic"]
    ifield = args.intensity_field or cfg["intensity_field"]
    iscale = args.intensity_scale or cfg["intensity_scale"]

    run_name = args.run_name or mcap_path.stem
    out_root = Path(args.out) / run_name
    points_dir = out_root / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    print(f"Meres        : {run_name}")
    print(f"LiDAR topic  : {topic}")
    print(f"Intensity    : {ifield} / {iscale}")
    print(f"Ego frame    : {EGO_FRAME}")
    print(f"Ritkitas     : minden {args.stride}. frame")
    print()

    print("Odometria beolvasasa ...")
    ts, pos, quat = read_odometry(mcap_path)
    print(f"  {len(ts)} poz, {ts[0]:.3f} .. {ts[-1]:.3f} ({ts[-1]-ts[0]:.1f} s)")

    # A LiDAR frame_id-t az elso uzenetbol vesszuk, majd abbol keressuk a TF-et.
    first_pc = None
    for msg in read_ros2_messages(str(mcap_path), topics=[topic]):
        first_pc = msg.ros_msg
        break
    if first_pc is None:
        sys.exit(f"[!] Nincs uzenet a(z) {topic} topicon ebben a fajlban.")

    lidar_frame = first_pc.header.frame_id
    t_le, q_le = read_static_transform(mcap_path, lidar_frame)
    print(f"\nExtrinsic {EGO_FRAME} -> {lidar_frame}:")
    print(f"  t = ({t_le[0]:.4f}, {t_le[1]:.4f}, {t_le[2]:.4f})")
    print(f"  q = ({q_le[0]:.5f}, {q_le[1]:.5f}, {q_le[2]:.5f}, {q_le[3]:.5f})"
          f"   yaw = {q_yaw_deg(q_le):.2f} fok")

    frames = []
    kept = 0
    skipped_no_pose = 0
    seen = 0
    ref_pos = ref_quat = None
    z_samples = []

    print("\nFrame-ek feldolgozasa ...")
    for msg in read_ros2_messages(str(mcap_path), topics=[topic]):
        if seen % args.stride != 0:
            seen += 1
            continue
        seen += 1

        pc = msg.ros_msg
        t_lidar = stamp_to_s(pc.header.stamp)

        pose = interp_pose(ts, pos, quat, t_lidar, args.max_pose_gap)
        if pose is None:
            skipped_no_pose += 1
            continue
        p_abs, q_abs = pose

        # Menetlokalis origo: az elso ervenyes poz teljes merevtest-
        # transzformaciojanak inverzevel kepezzuk a relativ pozokat.
        if ref_pos is None:
            ref_pos, ref_quat = p_abs.copy(), q_abs.copy()
        R_ref_T = q_to_matrix(ref_quat).T
        p_rel = R_ref_T @ (p_abs - ref_pos)
        q_rel = q_normalize(q_multiply(q_conjugate(ref_quat), q_abs))

        pts = pointcloud_to_array(pc, ifield)
        pts = clean_points(pts, args.min_range, args.max_range)
        if len(pts) < args.min_points:
            print(f"  [kihagy] {kept:07d}: csak {len(pts)} ervenyes pont")
            continue

        pts = transform_to_ego(pts, t_le, q_le)
        pts[:, 3] = np.clip(pts[:, 3] / iscale, 0.0, 1.0)

        name = f"{kept:07d}"
        bin_path = points_dir / f"{name}.bin"
        pts.astype("<f4").tofile(bin_path)

        digest = hashlib.sha256(bin_path.read_bytes()).hexdigest()
        z_samples.append((float(np.percentile(pts[:, 2], 1)),
                          float(np.percentile(pts[:, 2], 99))))

        frames.append({
            "name": f"frame_{name}",
            "sample_id": name,
            "pcd_path": f"points/{name}.bin",
            "sha256": digest,
            "num_points": int(len(pts)),
            "timestamp": round(t_lidar - ts[0], 6),
            "abs_timestamp_s": round(t_lidar, 6),
            "ego_pose": {
                "position": {"x": float(p_rel[0]),
                             "y": float(p_rel[1]),
                             "z": float(p_rel[2])},
                "heading": {"qx": float(q_rel[0]), "qy": float(q_rel[1]),
                            "qz": float(q_rel[2]), "qw": float(q_rel[3])},
            },
        })
        kept += 1

        if kept % 10 == 0:
            print(f"  {kept} frame kesz ...")
        if args.max_frames and kept >= args.max_frames:
            break

    if not frames:
        sys.exit("[!] Egyetlen hasznalhato frame sem keszult.")

    meta = {
        "dataset": run_name,
        "task_type": "pointcloud-cuboid-sequence",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "mcap_file": mcap_path.name,
            "lidar_topic": topic,
            "lidar_frame_id": lidar_frame,
            "odom_topic": ODOM_TOPIC,
            "frame_stride": args.stride,
        },
        "point_format": {
            "asset_type": "binary-xyzi",
            "dtype": "little-endian float32",
            "fields": ["x", "y", "z", "intensity"],
            "num_features": 4,
            "header": "none",
            "units": "meter",
            "coordinate_frame": EGO_FRAME,
            "intensity_source_field": ifield,
            "intensity_scale_divisor": iscale,
            "intensity_range": "0.0 - 1.0 (clipped)",
            "note": ("z=0 a base_link magassaga; a talaj kb. z=-0.37 m "
                     "(lexus3/ground_link offset)."),
        },
        "extrinsic_ego_to_lidar": {
            "parent_frame": EGO_FRAME,
            "child_frame": lidar_frame,
            "translation": {"x": float(t_le[0]), "y": float(t_le[1]),
                            "z": float(t_le[2])},
            "rotation": {"qx": float(q_le[0]), "qy": float(q_le[1]),
                         "qz": float(q_le[2]), "qw": float(q_le[3])},
            "source": "/tf_static",
        },
        "ego_pose_convention": {
            "origin": "run-local (elso ervenyes poz inverze)",
            "position_units": "meter",
            "orientation": "normalized quaternion (qx, qy, qz, qw)",
            "interpolation": "position: linear, orientation: SLERP",
            "max_pose_gap_s": args.max_pose_gap,
            "reference_abs_position_utm": {
                "x": float(ref_pos[0]), "y": float(ref_pos[1]),
                "z": float(ref_pos[2]),
            },
        },
        "timestamp_convention": {
            "timestamp": "masodperc, a meres elso odom mintajahoz kepest",
            "abs_timestamp_s": "abszolut UNIX ido masodpercben",
            "source": "PointCloud2 header.stamp (szenzoridobelyeg)",
        },
        "filtering": {
            "min_range_m": args.min_range,
            "max_range_m": args.max_range if args.max_range > 0 else None,
            "removed": "NaN/inf es (0,0,0) kitoltopontok",
        },
        "num_frames": len(frames),
        "frames": frames,
    }

    meta_path = out_root / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # ---- osszegzes es ellenorzo szamok
    npts = [f["num_points"] for f in frames]
    dists = [math.sqrt(f["ego_pose"]["position"]["x"] ** 2 +
                       f["ego_pose"]["position"]["y"] ** 2) for f in frames]
    z_lo = sum(s[0] for s in z_samples) / len(z_samples)
    z_hi = sum(s[1] for s in z_samples) / len(z_samples)

    print()
    print("=" * 60)
    print(f"Kesz: {len(frames)} frame -> {out_root}")
    print("=" * 60)
    print(f"  pont/frame      : min {min(npts)}  atlag {sum(npts)//len(npts)}  max {max(npts)}")
    print(f"  ego elmozdulas  : {max(dists):.2f} m a menet alatt")
    print(f"  z 1-99 pct      : {z_lo:.2f} .. {z_hi:.2f} m   (talaj kb. -0.37 koruli legyen)")
    if skipped_no_pose:
        print(f"  ego poz nelkul  : {skipped_no_pose} frame eldobva")

    if z_lo > 0.5 or z_lo < -3.0:
        print("\n  [!] A z also percentilis gyanus. Ellenorizd a transzformaciot,")
        print("      mielott feltoltod: a talajnak -0.4 m korul kell lennie.")

    if args.ply:
        write_ply(points_dir / "0000000.bin", out_root / "frame_0000000.ply")
        print(f"\n  Vizualis ellenorzeshez: {out_root / 'frame_0000000.ply'}")
        print("  Nyisd meg CloudCompare-ben vagy MeshLabben.")

    print(f"\n  metadata.json   : {meta_path}")


def write_ply(bin_path, ply_path):
    """Az elso frame ASCII PLY-be, hogy barmilyen nezegetovel ellenorizheto legyen."""
    pts = np.fromfile(bin_path, dtype="<f4").reshape(-1, 4)
    with open(ply_path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        for x, y, z, i in pts:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {i:.4f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mcap", help="a feldolgozando .mcap fajl")
    ap.add_argument("--lidar", choices=sorted(LIDARS), default="ouster",
                    help="melyik LiDAR topicot hasznaljuk (alap: ouster)")
    ap.add_argument("--out", default="data/custom_dataset",
                    help="kimeneti gyoker (alap: data/custom_dataset)")
    ap.add_argument("--run-name", default=None,
                    help="a meres neve; alapbol a fajlnev")
    ap.add_argument("--stride", type=int, default=5,
                    help="minden N. frame; 5 = 2 Hz a 10 Hz-es adatbol")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="legfeljebb ennyi frame (0 = mind). Pilothoz: 20")
    ap.add_argument("--min-range", type=float, default=0.5,
                    help="ennel kozelebbi pontok eldobasa, meterben")
    ap.add_argument("--max-range", type=float, default=0.0,
                    help="ennel tavolabbi pontok eldobasa (0 = nincs korlat)")
    ap.add_argument("--min-points", type=int, default=1000,
                    help="ennel kevesebb ervenyes pontu frame kihagyasa")
    ap.add_argument("--max-pose-gap", type=float, default=0.05,
                    help="max megengedett ido-elteres a legkozelebbi odom mintaig")
    ap.add_argument("--intensity-field", default=None,
                    help="felulirja az intensity forrasmezot")
    ap.add_argument("--intensity-scale", type=float, default=None,
                    help="felulirja az intensity oszto erteket")
    ap.add_argument("--ply", action="store_true",
                    help="az elso frame-et PLY-be is kiirja ellenorzeshez")
    convert(ap.parse_args())


if __name__ == "__main__":
    main()