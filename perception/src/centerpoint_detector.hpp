#ifndef PERCEPTION__CENTERPOINT_DETECTOR_HPP_
#define PERCEPTION__CENTERPOINT_DETECTOR_HPP_

// ROS-független CenterPoint (pillar) inferencia ONNX Runtime-mal.
// A paraméterek alapértelmezései pontosan a notebookban kiértékelt
// mmdetection3d configot követik:
//   centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
// Ha más configból exportálsz, ezeket is át kell állítani!

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace perception
{

struct CenterPointConfig
{
  std::string pfe_model_path;   // pts_voxel_encoder (PFN) ONNX
  std::string head_model_path;  // backbone + neck + head ONNX
  bool use_cuda = true;         // ha az ORT build nem tud CUDA-t, CPU-ra esik vissza
  int intra_op_threads = 4;

  // --- a tanítási configgal egyezniük kell ---
  std::array<float, 6> point_cloud_range{-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
  std::array<float, 3> voxel_size{0.2f, 0.2f, 8.0f};
  int max_points_per_voxel = 20;
  int max_voxels = 40000;       // max_voxels[1] (teszt)
  int num_point_features = 5;   // x, y, z, intensity, time_lag
  int pfe_out_channels = 64;
  int out_size_factor = 4;

  // --- dekódolás (CenterPointBBoxCoder + circle NMS) ---
  float score_threshold = 0.1f;
  int max_num_per_task = 500;
  int post_max_size = 83;
  std::array<float, 6> post_center_range{-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
  std::vector<int> task_num_classes{1, 2, 2, 1, 2, 2};
  std::vector<float> circle_nms_min_radius{4.0f, 12.0f, 10.0f, 1.0f, 0.85f, 0.175f};
};

// Egy detekció a MODELL koordináta-rendszerében (nuScenes LIDAR_TOP: x jobbra, y előre).
struct Detection3D
{
  float x = 0.f, y = 0.f, z = 0.f;  // doboz KÖZÉPPONTJA (nem az alja!)
  float length = 0.f, width = 0.f, height = 0.f;
  float yaw = 0.f;                  // rad, x tengelytől, óramutatóval ellentétesen
  float vx = 0.f, vy = 0.f;         // m/s (abszolút sebesség a szenzor tengelyeiben)
  float score = 0.f;
  int label = -1;                   // index a centerPointClassNames()-be
};

// Az osztályok sorrendje a head task-jainak sorrendje (mmdet3d label index).
const std::vector<std::string> & centerPointClassNames();

class CenterPointDetector
{
public:
  struct Timing
  {
    double voxelize_ms = 0.0;
    double pfe_ms = 0.0;
    double scatter_ms = 0.0;
    double head_ms = 0.0;
    double decode_ms = 0.0;
    int num_voxels = 0;
  };

  explicit CenterPointDetector(const CenterPointConfig & cfg);
  ~CenterPointDetector();

  CenterPointDetector(const CenterPointDetector &) = delete;
  CenterPointDetector & operator=(const CenterPointDetector &) = delete;

  // points: num_points x num_point_features, sorfolytonos float tömb, modell-frame-ben.
  std::vector<Detection3D> detect(const float * points, std::size_t num_points);

  const Timing & lastTiming() const;
  bool usingCuda() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace perception

#endif  // PERCEPTION__CENTERPOINT_DETECTOR_HPP_
