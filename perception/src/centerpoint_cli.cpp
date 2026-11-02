// Offline ellenőrző eszköz (ROS nélkül): beolvas egy nuScenes formátumú .bin
// pontfelhőt (float32, N x load_dim), lefuttatja a C++ CenterPoint pipeline-t és
// JSON-ként kiírja a detekciókat a MODELL frame-ben. Az export script által
// mentett reference_detections.json-nal összevetve ellenőrizhető, hogy a C++
// elő- és utófeldolgozás egyezik-e az mmdetection3d-vel.
//
//   centerpoint_cli <pfe.onnx> <head.onnx> <points.bin> [load_dim=5] [min_score=0.3] [--cpu]

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "perception/centerpoint_detector.hpp"

int main(int argc, char ** argv)
{
  if (argc < 4) {
    std::cerr << "Használat: " << argv[0]
              << " <pfe.onnx> <head.onnx> <points.bin> [load_dim=5] [min_score=0.3] [--cpu]\n";
    return 1;
  }
  const int load_dim = argc > 4 ? std::stoi(argv[4]) : 5;
  const float min_score = argc > 5 ? std::stof(argv[5]) : 0.3f;
  bool use_cuda = true;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--cpu") == 0) {
      use_cuda = false;
    }
  }

  std::ifstream f(argv[3], std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "Nem olvasható: " << argv[3] << "\n";
    return 1;
  }
  const auto bytes = static_cast<size_t>(f.tellg());
  f.seekg(0);
  std::vector<float> raw(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
  const size_t n = raw.size() / load_dim;

  // modell-bemenet: x, y, z, intensity, time_lag (= 5. oszlop, ha van, különben 0)
  std::vector<float> pts(n * 5, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 5 && k < load_dim; ++k) {
      pts[i * 5 + k] = raw[i * load_dim + k];
    }
  }

  perception::CenterPointConfig cfg;
  cfg.pfe_model_path = argv[1];
  cfg.head_model_path = argv[2];
  cfg.use_cuda = use_cuda;
  perception::CenterPointDetector det(cfg);

  auto dets = det.detect(pts.data(), n);
  dets = det.detect(pts.data(), n);  // második futás: bemelegített időmérés
  const auto & t = det.lastTiming();
  const auto & names = perception::centerPointClassNames();

  std::fprintf(stderr,
    "%zu pont, %d pillar | voxel %.1f ms, pfe %.1f ms, scatter %.1f ms, head %.1f ms, decode %.1f ms (%s)\n",
    n, t.num_voxels, t.voxelize_ms, t.pfe_ms, t.scatter_ms, t.head_ms, t.decode_ms,
    det.usingCuda() ? "CUDA" : "CPU");

  std::printf("[\n");
  bool first = true;
  for (const auto & d : dets) {
    if (d.score < min_score) {
      continue;
    }
    std::printf("%s  {\"class_name\": \"%s\", \"score\": %.4f, \"center_xyz\": [%.3f, %.3f, %.3f], "
                "\"size_xyz\": [%.3f, %.3f, %.3f], \"yaw\": %.4f, \"velocity_xy\": [%.3f, %.3f]}",
                first ? "" : ",\n", names[d.label].c_str(), d.score, d.x, d.y, d.z,
                d.length, d.width, d.height, d.yaw, d.vx, d.vy);
    first = false;
  }
  std::printf("\n]\n");
  return 0;
}
