#!/usr/bin/env python3
"""CenterPoint (pillar02, nuScenes) exportálása két ONNX modellbe a C++ perception node-hoz.

A notebook Colab környezetében futtasd (mmdet3d 1.4.0, mmcv 2.1.0, torch 2.1), plusz:
    pip install onnx onnxruntime

    cd /content/mmdetection3d
    python export_centerpoint_onnx.py \
        --config configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
        --checkpoint checkpoints/centerpoint_pillar02_nuscenes.pth \
        --sample-bin data/nuscenes/samples/LIDAR_TOP/<valamelyik>.pcd.bin \
        --out-dir /content/centerpoint_onnx

Kimenet (out-dir):
    centerpoint_pfe.onnx           PillarFeatureNet PFN rétege   [P,20,11]      -> [P,64]
    centerpoint_head.onnx          backbone+neck+head            [1,512,512,64] -> task{t}_{heatmap,reg,height,dim,rot,vel}
    sample_points.bin              a teszt pontfelhő (float32, Nx5, time_lag = 0)
    reference_detections.json      az mmdet3d saját kimenete ugyanerre a felhőre

A script két szinten ellenőriz, és hibával leáll, ha eltérést talál:
    1) a C++-ban implementált pillar-dekoráció + ONNX PFE == mmdet3d PillarFeatureNet
    2) ONNX head == PyTorch head ugyanarra a canvasra
A C++ elő-/utófeldolgozás végső ellenőrzése:
    centerpoint_cli centerpoint_pfe.onnx centerpoint_head.onnx sample_points.bin 5 0.3 > cpp.json
    python export_centerpoint_onnx.py --compare reference_detections.json cpp.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCORE_THR = 0.3
TASK_KEYS = ("heatmap", "reg", "height", "dim", "rot", "vel")


def compare(ref_path, cpp_path, max_center_dist=0.3):
    ref = json.loads(Path(ref_path).read_text())
    cpp = json.loads(Path(cpp_path).read_text())
    used, matched, worst = set(), 0, 0.0
    for r in sorted(ref, key=lambda d: -d["score"]):
        best = None
        for k, c in enumerate(cpp):
            if k in used or c["class_name"] != r["class_name"]:
                continue
            dist = float(np.hypot(c["center_xyz"][0] - r["center_xyz"][0], c["center_xyz"][1] - r["center_xyz"][1]))
            if best is None or dist < best[0]:
                best = (dist, k)
        if best is not None and best[0] <= max_center_dist:
            used.add(best[1])
            matched += 1
            worst = max(worst, best[0])
        else:
            print(f"  NINCS PÁR: {r['class_name']} score={r['score']:.3f} center={np.round(r['center_xyz'], 2).tolist()}")
    print(f"Referencia: {len(ref)}, C++: {len(cpp)}, párosítva: {matched}, max középpont-eltérés: {worst:.3f} m")
    # küszöb közeli score-oknál 1-2 eltérés elfogadható (float/voxel sorrend)
    return 0 if matched >= len(ref) - 2 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--sample-bin", help="nuScenes LIDAR_TOP .pcd.bin (float32, 5 oszlop)")
    ap.add_argument("--out-dir", default="centerpoint_onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--compare", nargs=2, metavar=("REF_JSON", "CPP_JSON"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))
    if not (args.config and args.checkpoint and args.sample_bin):
        ap.error("--config, --checkpoint és --sample-bin kötelező")

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from torch import nn
    from mmengine.config import Config
    from mmdet3d.apis import inference_detector, init_model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    # egyetlen sweep, közvetlenül átadott pontokkal (mint a notebookban)
    cfg.test_dataloader.dataset.pipeline = [
        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
        dict(type="Pack3DDetInputs", keys=["points"]),
    ]
    model = init_model(cfg, args.checkpoint, device=args.device).eval()
    vfe = model.pts_voxel_encoder
    assert type(vfe).__name__ == "PillarFeatureNet" and not vfe.legacy and not vfe._with_distance, \
        "Ez az export csak a PillarFeatureNet(legacy=False, with_distance=False) modellekhez készült"

    # ---------------- teszt pontfelhő ----------------
    raw = np.fromfile(args.sample_bin, dtype=np.float32).reshape(-1, 5)
    pts = raw.copy()
    pts[:, 4] = 0.0  # a nuScenes .bin 5. oszlopa ring index; a modell time_lag-et vár -> 0
    pts.tofile(out_dir / "sample_points.bin")
    points = torch.from_numpy(pts).to(args.device)

    with torch.no_grad():
        # mmdet3d saját voxelizációja és PFE-je (referencia)
        voxels, coors, num_points = model.data_preprocessor.voxel_layer(points)
        coors_b = F.pad(coors, (1, 0), mode="constant", value=0)
        ref_pfe = vfe(voxels, num_points, coors_b)

        # ugyanaz a dekoráció, amit a C++ csinál
        V, M, Fdim = voxels.shape
        mean = voxels[:, :, :3].sum(1, keepdim=True) / num_points.view(-1, 1, 1).float()
        dec = torch.zeros(V, M, Fdim + 6, device=voxels.device)
        dec[:, :, :Fdim] = voxels
        dec[:, :, Fdim:Fdim + 3] = voxels[:, :, :3] - mean
        dec[:, :, Fdim + 3] = voxels[:, :, 0] - (coors[:, 2].float().unsqueeze(1) * vfe.vx + vfe.x_offset)
        dec[:, :, Fdim + 4] = voxels[:, :, 1] - (coors[:, 1].float().unsqueeze(1) * vfe.vy + vfe.y_offset)
        dec[:, :, Fdim + 5] = voxels[:, :, 2] - (coors[:, 0].float().unsqueeze(1) * vfe.vz + vfe.z_offset)
        mask = (torch.arange(M, device=voxels.device).view(1, M) < num_points.view(-1, 1)).float().unsqueeze(-1)
        dec = dec * mask

    # ---------------- PFE export ----------------
    class PFNExport(nn.Module):
        def __init__(self, vfe):
            super().__init__()
            self.pfn_layers = vfe.pfn_layers

        def forward(self, features):
            x = features
            for pfn in self.pfn_layers:
                x = pfn(x)
            return x.squeeze(1)

    pfe_path = out_dir / "centerpoint_pfe.onnx"
    pfe_module = PFNExport(vfe).eval()
    torch.onnx.export(
        pfe_module, (dec,), str(pfe_path), opset_version=args.opset,
        input_names=["features"], output_names=["pillar_features"],
        dynamic_axes={"features": {0: "num_pillars"}, "pillar_features": {0: "num_pillars"}})

    sess = ort.InferenceSession(str(pfe_path), providers=["CPUExecutionProvider"])
    onnx_pfe = sess.run(None, {"features": dec.cpu().numpy()})[0]
    err = np.abs(onnx_pfe - ref_pfe.cpu().numpy()).max()
    print(f"[1] PFE: C++-dekoráció + ONNX vs mmdet3d PillarFeatureNet, max eltérés = {err:.2e}")
    assert err < 1e-3, "A PFE kimenet eltér: a dekoráció nem egyezik a modellel!"

    # ---------------- head export (NHWC bemenet) ----------------
    class HeadExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.backbone = model.pts_backbone
            self.neck = model.pts_neck
            self.head = model.pts_bbox_head

        def forward(self, canvas_nhwc):
            x = canvas_nhwc.permute(0, 3, 1, 2).contiguous()
            feats = self.neck(self.backbone(x))
            task_outs = self.head(feats)  # tuple[task] -> list[level] -> dict
            outs = []
            for task in task_outs:
                d = task[0]
                outs += [d["heatmap"].sigmoid(), d["reg"], d["height"], d["dim"], d["rot"], d["vel"]]
            return tuple(outs)

    num_tasks = len(model.pts_bbox_head.task_heads)
    out_names = [f"task{t}_{k}" for t in range(num_tasks) for k in TASK_KEYS]
    with torch.no_grad():
        canvas = model.pts_middle_encoder(ref_pfe, coors_b, 1)  # [1,64,ny,nx]
        canvas_nhwc = canvas.permute(0, 2, 3, 1).contiguous()
        head_module = HeadExport(model).eval()
        torch_outs = head_module(canvas_nhwc)

    head_path = out_dir / "centerpoint_head.onnx"
    torch.onnx.export(head_module, (canvas_nhwc,), str(head_path), opset_version=args.opset,
                      input_names=["spatial_features"], output_names=out_names)

    sess = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
    onnx_outs = sess.run(out_names, {"spatial_features": canvas_nhwc.cpu().numpy()})
    worst = max(float(np.abs(o - t.cpu().numpy()).max()) for o, t in zip(onnx_outs, torch_outs))
    print(f"[2] Head: ONNX vs PyTorch, max eltérés = {worst:.2e}")
    assert worst < 1e-2, "A head ONNX kimenete eltér a PyTorch-tól!"

    # ---------------- referencia detekciók ----------------
    result, _ = inference_detector(model, pts)
    inst = result.pred_instances_3d
    keep = inst.scores_3d >= SCORE_THR
    boxes = inst.bboxes_3d.tensor[keep].cpu().numpy()
    scores = inst.scores_3d[keep].cpu().numpy()
    labels = inst.labels_3d[keep].cpu().numpy().astype(int)
    # a label index a head task-jainak sorrendjét követi (ezt használja a C++ is)
    classes = [n for task in cfg.model.pts_bbox_head.tasks for n in task["class_names"]]
    ref = []
    for b, s, l in zip(boxes, scores, labels):
        center = b[:3].copy()
        center[2] += b[5] * 0.5  # mmdet3d: doboz alja -> C++: középpont
        ref.append({"class_name": classes[l], "score": float(s), "center_xyz": center.tolist(),
                    "size_xyz": b[3:6].tolist(), "yaw": float(b[6]),
                    "velocity_xy": b[7:9].tolist() if len(b) >= 9 else [0.0, 0.0]})
    (out_dir / "reference_detections.json").write_text(json.dumps(ref, indent=1))
    print(f"[3] {len(ref)} referencia detekció (score >= {SCORE_THR}) mentve.")
    print(f"Kész: {pfe_path}, {head_path}")


if __name__ == "__main__":
    main()
