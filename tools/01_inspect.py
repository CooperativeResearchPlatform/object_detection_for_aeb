"""
  - milyen topicok vannak, hany uzenettel, milyen tipussal
  - a LiDAR PointCloud2 mezoszerkezete (nev / offset / tipus)
  - frame_id, pontszam, uzenetgyakorisag
  - van-e /novatel/oem7/odom es lefedi-e a teljes menetet
  - /tf_static transzformaciok (LiDAR -> ego extrinsic)
  - intensity jellegu mezok ertektartomanya (skala meghatarozashoz)

Futtatas:
    python 01_inspect.py 20261001_no_odometry_msgs 20261007

"""

import sys
import math
import struct
from pathlib import Path

try:
    from mcap.reader import make_reader
except ImportError:
    sys.exit("Hianyzik a csomag. Futtasd: pip install mcap mcap-ros2-support")

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError:
    sys.exit("Hianyzik a csomag. Futtasd: pip install mcap mcap-ros2-support")


LIDAR_CANDIDATES = ["/luminar_pcl", "/lexus3/os_center/points"]
ODOM_TOPIC = "/novatel/oem7/odom"

# PointField datatype kodok -> (nev, struct kod, bajt)
PF_TYPES = {
    1: ("INT8", "b", 1),
    2: ("UINT8", "B", 1),
    3: ("INT16", "h", 2),
    4: ("UINT16", "H", 2),
    5: ("INT32", "i", 4),
    6: ("UINT32", "I", 4),
    7: ("FLOAT32", "f", 4),
    8: ("FLOAT64", "d", 8),
}

_out_lines = []


def say(text=""):
    print(text)
    _out_lines.append(text)


def ns_to_s(ns):
    return ns / 1e9


def summarize_file(path: Path):
    say("")
    say("=" * 78)
    say(f"FAJL: {path}")
    say("=" * 78)

    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        if summary is None:
            say("  [!] Nincs summary szekcio a fajlban (lehet serult vagy nem lezart).")
            return None

        schemas = {s.id: s.name for s in summary.schemas.values()}
        stats = summary.statistics
        counts = stats.channel_message_counts if stats else {}

        if stats:
            dur = ns_to_s(stats.message_end_time - stats.message_start_time)
            say(f"  Idotartam: {dur:.2f} s")
            say(f"  Osszes uzenet: {stats.message_count}")
        say("")
        say("  Topicok:")
        say(f"    {'topic':<42} {'tipus':<34} {'db':>8}")
        say(f"    {'-'*42} {'-'*34} {'-'*8}")

        present = {}
        for ch in summary.channels.values():
            n = counts.get(ch.id, 0)
            present[ch.topic] = (schemas.get(ch.schema_id, "?"), n)
            say(f"    {ch.topic:<42} {schemas.get(ch.schema_id,'?'):<34} {n:>8}")

    # --- LiDAR topicok reszletes vizsgalata ---
    for topic in LIDAR_CANDIDATES:
        if topic in present:
            inspect_pointcloud(path, topic, present[topic][1])

    # --- Odom ---
    inspect_odom(path, present)

    # --- TF static ---
    inspect_tf_static(path, present)

    return present


def inspect_pointcloud(path: Path, topic: str, count: int):
    say("")
    say(f"  --- PointCloud2 reszletek: {topic} ---")

    first = None
    stamps = []
    try:
        for i, msg in enumerate(read_ros2_messages(str(path), topics=[topic])):
            if first is None:
                first = msg.ros_msg
            hdr = msg.ros_msg.header.stamp
            stamps.append(hdr.sec + hdr.nanosec * 1e-9)
            if i >= 40:
                break
    except Exception as e:
        say(f"    [!] Dekodolasi hiba: {type(e).__name__}: {e}")
        return

    if first is None:
        say("    [!] Nem sikerult uzenetet olvasni.")
        return

    pc = first
    say(f"    frame_id        : {pc.header.frame_id}")
    say(f"    width x height  : {pc.width} x {pc.height}")
    say(f"    point_step      : {pc.point_step} bajt")
    say(f"    row_step        : {pc.row_step}")
    say(f"    is_bigendian    : {pc.is_bigendian}")
    say(f"    is_dense        : {getattr(pc, 'is_dense', 'n/a')}")
    say(f"    pontszam/frame  : {pc.width * pc.height}")
    say(f"    data meret      : {len(pc.data)} bajt")

    say("")
    say("    Mezok:")
    say(f"      {'nev':<16} {'offset':>7} {'tipus':<9} {'count':>6}")
    say(f"      {'-'*16} {'-'*7} {'-'*9} {'-'*6}")
    for fld in pc.fields:
        tname = PF_TYPES.get(fld.datatype, (f"?{fld.datatype}", None, 0))[0]
        say(f"      {fld.name:<16} {fld.offset:>7} {tname:<9} {fld.count:>6}")

    # frekvencia
    if len(stamps) > 2:
        diffs = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if diffs:
            dt = sum(diffs) / len(diffs)
            say("")
            say(f"    atlagos dt      : {dt*1000:.1f} ms  (~{1/dt:.1f} Hz)")
        say(f"    elso stamp      : {stamps[0]:.6f}")

    # mezo ertektartomanyok (az elso frame alapjan)
    dump_field_ranges(pc)


def dump_field_ranges(pc):
    """Minden mezo min/max erteke az elso frame-en - a skala es a tengelyek
    ellenorzesehez (pl. meter vagy mm, intensity 0-1 vagy 0-255)."""
    data = bytes(pc.data)
    npts = pc.width * pc.height
    if npts == 0 or pc.point_step == 0:
        return
    step = pc.point_step
    endian = ">" if pc.is_bigendian else "<"

    say("")
    say("    Mezo ertektartomanyok (elso frame, max 20000 pont):")
    limit = min(npts, 20000)

    for fld in pc.fields:
        info = PF_TYPES.get(fld.datatype)
        if info is None:
            continue
        _, code, size = info
        vmin, vmax = math.inf, -math.inf
        finite = 0
        for k in range(limit):
            off = k * step + fld.offset
            if off + size > len(data):
                break
            (v,) = struct.unpack_from(endian + code, data, off)
            if isinstance(v, float) and not math.isfinite(v):
                continue
            finite += 1
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        if finite:
            say(f"      {fld.name:<16} min={vmin:>12.4f}  max={vmax:>12.4f}  (ervenyes: {finite})")
        else:
            say(f"      {fld.name:<16} [nincs ervenyes ertek]")


def inspect_odom(path: Path, present: dict):
    say("")
    say(f"  --- Ego odometria: {ODOM_TOPIC} ---")
    if ODOM_TOPIC not in present:
        say("    [!] NINCS a fajlban.")
        say("        Ego poz nelkul a Segments.ai sequence interpolacio nem hasznalhato,")
        say("        es a dokumentacio szerint az ilyen pontfelhot el kell dobni.")
        return

    try:
        first = last = None
        n = 0
        stamps = []
        for msg in read_ros2_messages(str(path), topics=[ODOM_TOPIC]):
            o = msg.ros_msg
            if first is None:
                first = o
            last = o
            n += 1
            s = o.header.stamp
            stamps.append(s.sec + s.nanosec * 1e-9)
    except Exception as e:
        say(f"    [!] Dekodolasi hiba: {type(e).__name__}: {e}")
        return

    say(f"    uzenetszam      : {n}")
    say(f"    frame_id        : {first.header.frame_id}")
    say(f"    child_frame_id  : {getattr(first, 'child_frame_id', 'n/a')}")
    if len(stamps) > 2:
        diffs = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if diffs:
            dt = sum(diffs) / len(diffs)
            say(f"    atlagos dt      : {dt*1000:.1f} ms  (~{1/dt:.1f} Hz)")
        say(f"    idotartomany    : {stamps[0]:.3f} .. {stamps[-1]:.3f}")

    p = first.pose.pose.position
    q = first.pose.pose.orientation
    say(f"    elso position   : x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}")
    say(f"    elso heading    : qx={q.x:.5f} qy={q.y:.5f} qz={q.z:.5f} qw={q.w:.5f}")
    norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
    say(f"    kvaternio norma : {norm:.6f}  (1.0 korul kell legyen)")

    p2 = last.pose.pose.position
    say(f"    utolso position : x={p2.x:.3f} y={p2.y:.3f} z={p2.z:.3f}")
    dist = math.sqrt((p2.x - p.x) ** 2 + (p2.y - p.y) ** 2)
    say(f"    elmozdulas      : {dist:.2f} m  (ha ~0, a jarmu allt)")


def inspect_tf_static(path: Path, present: dict):
    say("")
    say("  --- /tf_static (LiDAR -> ego extrinsic) ---")
    if "/tf_static" not in present:
        say("    [!] Nincs /tf_static. Az extrinsic kulon kalibracios fajlbol kell.")
        return
    try:
        seen = set()
        for msg in read_ros2_messages(str(path), topics=["/tf_static"]):
            for tr in msg.ros_msg.transforms:
                key = (tr.header.frame_id, tr.child_frame_id)
                if key in seen:
                    continue
                seen.add(key)
                t = tr.transform.translation
                r = tr.transform.rotation
                say(f"    {tr.header.frame_id} -> {tr.child_frame_id}")
                say(f"      t = ({t.x:.4f}, {t.y:.4f}, {t.z:.4f})")
                say(f"      q = ({r.x:.5f}, {r.y:.5f}, {r.z:.5f}, {r.w:.5f})")
            if len(seen) > 60:
                break
        if not seen:
            say("    [!] Ures /tf_static.")
    except Exception as e:
        say(f"    [!] Dekodolasi hiba: {type(e).__name__}: {e}")


def main():
    folders = sys.argv[1:]
    if not folders:
        sys.exit("Hasznalat: python 01_inspect.py <mappa1> [<mappa2> ...]")

    files = []
    for folder in folders:
        p = Path(folder)
        if not p.exists():
            say(f"[!] Nem letezik: {folder}")
            continue
        files.extend(sorted(p.rglob("*.mcap")))

    if not files:
        sys.exit("Nem talaltam .mcap fajlt a megadott mappakban.")

    say(f"Talalt .mcap fajlok: {len(files)}")

    for path in files:
        try:
            summarize_file(path)
        except Exception as e:
            say(f"  [!] Hiba a fajl feldolgozasakor: {type(e).__name__}: {e}")

    out = Path("mcap_inventory.txt")
    out.write_text("\n".join(_out_lines), encoding="utf-8")
    print()
    print(f"A teljes riport ide keszult el: {out.resolve()}")


if __name__ == "__main__":
    main()
