# perception – CenterPoint (LiDAR) + akadály-/fékdöntés

## 1. Modell exportálása (Colab, a notebook környezetében)
    pip install onnx onnxruntime
    cd /content/mmdetection3d
    python export_centerpoint_onnx.py \
      --config configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
      --checkpoint checkpoints/centerpoint_pillar02_nuscenes.pth \
      --sample-bin data/nuscenes/samples/LIDAR_TOP/<fájl>.pcd.bin \
      --out-dir /content/centerpoint_onnx

A `centerpoint_pfe.onnx` és `centerpoint_head.onnx` fájlokat másold a `models/` mappába.

## 2. Fordítás
ONNX Runtime C++ release letöltése (GPU-hoz: onnxruntime-linux-x64-gpu-*.tgz), kicsomagolás, majd:

    colcon build --packages-select perception --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime

Megjegyzés: az ONNX Runtime verziója támogassa az export IR verzióját (torch 2.1 exporthoz az 1.16+ elég).

## 3. C++ vs. mmdet3d ellenőrzés

    ros2 run perception centerpoint_cli models/centerpoint_pfe.onnx models/centerpoint_head.onnx sample_points.bin 5 0.3 > cpp.json
    python3 export_centerpoint_onnx.py --compare reference_detections.json cpp.json

## 4. Futtatás

    ros2 launch perception perception.launch.py pointcloud_topic:=/lidar/points ego_topic:=/ego

Topicok:
- be: `pointcloud` (PointCloud2), `ego` (crp_msgs/Ego)
- ki: `scenario` (crp_msgs/Scenario), `brake_request` (std_msgs/Bool), `obstacle_distance` (std_msgs/Float32)

A paraméterek leírása: `config/perception.yaml`. Különösen ellenőrizendő: `input.yaw_rotation_deg`
(a modell nuScenes frame-je x=jobbra, y=előre), `input.z_offset_m`, `input.intensity_scale`,
`brake.front_offset_m`, `brake.half_width_m`, `brake.sensor_height_m`.
