#include "perception/centerpoint_detector.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace perception
{

namespace
{
using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// A head ONNX kimenetei taskonként ebben a sorrendben vannak
// (lásd scripts/export_centerpoint_onnx.py): task{t}_{key}
constexpr int kOutputsPerTask = 6;
const char * const kTaskOutputKeys[kOutputsPerTask] = {
  "heatmap", "reg", "height", "dim", "rot", "vel"};

struct Candidate
{
  float score;
  int cls;
  int idx;
};
}  // namespace

const std::vector<std::string> & centerPointClassNames()
{
  static const std::vector<std::string> names{
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};
  return names;
}

struct CenterPointDetector::Impl
{
  CenterPointConfig cfg;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "centerpoint"};
  std::unique_ptr<Ort::Session> pfe;
  std::unique_ptr<Ort::Session> head;
  Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  bool cuda = false;

  int nx = 0, ny = 0, nz = 0;
  int feat_w = 0, feat_h = 0;
  int decorated_features = 0;

  std::string pfe_input_name, pfe_output_name, head_input_name;
  std::vector<std::string> head_output_names;

  // munkapufferek (egyszer allokálva)
  std::vector<int> grid_to_voxel;     // nx*ny, -1 = üres
  std::vector<float> voxel_points;    // max_voxels * max_pts * F
  std::vector<int> voxel_num_points;  // max_voxels
  std::vector<int> voxel_coords;      // max_voxels * 3 (ix, iy, iz)
  std::vector<float> pfe_input;       // max_voxels * max_pts * (F+6)
  std::vector<float> canvas;          // ny * nx * C (NHWC)
  std::vector<int> used_cells;        // az előző frame-ben írt cellák (gyors nullázáshoz)
  std::vector<Candidate> cands;

  Timing timing;

  Ort::SessionOptions makeOptions()
  {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(cfg.intra_op_threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (cfg.use_cuda) {
      try {
        OrtCUDAProviderOptions cuda_opts{};
        so.AppendExecutionProvider_CUDA(cuda_opts);
        cuda = true;
      } catch (const Ort::Exception & e) {
        std::cerr << "[centerpoint] CUDA EP nem elérhető, CPU-n fut: " << e.what() << std::endl;
        cuda = false;
      }
    }
    return so;
  }

  static std::string inputName(Ort::Session & s, size_t i)
  {
    Ort::AllocatorWithDefaultOptions alloc;
    return s.GetInputNameAllocated(i, alloc).get();
  }
  static std::string outputName(Ort::Session & s, size_t i)
  {
    Ort::AllocatorWithDefaultOptions alloc;
    return s.GetOutputNameAllocated(i, alloc).get();
  }

  explicit Impl(const CenterPointConfig & c)
  : cfg(c)
  {
    const auto & r = cfg.point_cloud_range;
    const auto & v = cfg.voxel_size;
    nx = static_cast<int>(std::lround((r[3] - r[0]) / v[0]));
    ny = static_cast<int>(std::lround((r[4] - r[1]) / v[1]));
    nz = static_cast<int>(std::lround((r[5] - r[2]) / v[2]));
    if (nz != 1) {
      throw std::runtime_error("CenterPointDetector: pillar modellnél a z-irányú rácsméretnek 1-nek kell lennie");
    }
    if (nx % cfg.out_size_factor != 0 || ny % cfg.out_size_factor != 0) {
      throw std::runtime_error("CenterPointDetector: a rácsméret nem osztható az out_size_factor-ral");
    }
    feat_w = nx / cfg.out_size_factor;
    feat_h = ny / cfg.out_size_factor;
    decorated_features = cfg.num_point_features + 6;  // + cluster offset (3) + pillar-center offset (3)

    if (cfg.circle_nms_min_radius.size() != cfg.task_num_classes.size()) {
      throw std::runtime_error("CenterPointDetector: circle_nms_min_radius és task_num_classes hossza eltér");
    }

    Ort::SessionOptions so = makeOptions();
    pfe = std::make_unique<Ort::Session>(env, cfg.pfe_model_path.c_str(), so);
    head = std::make_unique<Ort::Session>(env, cfg.head_model_path.c_str(), so);

    pfe_input_name = inputName(*pfe, 0);
    pfe_output_name = outputName(*pfe, 0);
    head_input_name = inputName(*head, 0);

    // A head kimeneteit név szerint kérjük, így az ONNX-beli sorrend nem számít.
    std::vector<std::string> available;
    for (size_t i = 0; i < head->GetOutputCount(); ++i) {
      available.push_back(outputName(*head, i));
    }
    for (size_t t = 0; t < cfg.task_num_classes.size(); ++t) {
      for (int k = 0; k < kOutputsPerTask; ++k) {
        std::string name = "task" + std::to_string(t) + "_" + kTaskOutputKeys[k];
        if (std::find(available.begin(), available.end(), name) == available.end()) {
          throw std::runtime_error("CenterPointDetector: hiányzó head kimenet: " + name);
        }
        head_output_names.push_back(name);
      }
    }

    const size_t mv = static_cast<size_t>(cfg.max_voxels);
    const size_t mp = static_cast<size_t>(cfg.max_points_per_voxel);
    grid_to_voxel.assign(static_cast<size_t>(nx) * ny, -1);
    voxel_points.resize(mv * mp * cfg.num_point_features);
    voxel_num_points.resize(mv);
    voxel_coords.resize(mv * 3);
    pfe_input.resize(mv * mp * decorated_features);
    canvas.assign(static_cast<size_t>(cfg.pfe_out_channels) * nx * ny, 0.0f);
    used_cells.reserve(mv);
  }

  // mmcv hard voxelization megfelelője: a pontok érkezési sorrendjében,
  // floor((p - min) / voxel), max_points_per_voxel / max_voxels limittel.
  int voxelize(const float * pts, size_t n)
  {
    const auto & r = cfg.point_cloud_range;
    const auto & v = cfg.voxel_size;
    const int F = cfg.num_point_features;
    const int MP = cfg.max_points_per_voxel;
    int num_voxels = 0;

    for (size_t i = 0; i < n; ++i) {
      const float * p = pts + i * F;
      if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
        continue;
      }
      const int ix = static_cast<int>(std::floor((p[0] - r[0]) / v[0]));
      const int iy = static_cast<int>(std::floor((p[1] - r[1]) / v[1]));
      const int iz = static_cast<int>(std::floor((p[2] - r[2]) / v[2]));
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
        continue;
      }
      const int cell = iy * nx + ix;
      int vid = grid_to_voxel[cell];
      if (vid < 0) {
        if (num_voxels >= cfg.max_voxels) {
          continue;
        }
        vid = num_voxels++;
        grid_to_voxel[cell] = vid;
        voxel_num_points[vid] = 0;
        voxel_coords[vid * 3 + 0] = ix;
        voxel_coords[vid * 3 + 1] = iy;
        voxel_coords[vid * 3 + 2] = iz;
      }
      const int k = voxel_num_points[vid];
      if (k < MP) {
        std::copy(p, p + F, voxel_points.begin() + (static_cast<size_t>(vid) * MP + k) * F);
        voxel_num_points[vid] = k + 1;
      }
    }
    // a grid_to_voxel visszaállítása csak az érintett cellákon
    for (int vid = 0; vid < num_voxels; ++vid) {
      grid_to_voxel[voxel_coords[vid * 3 + 1] * nx + voxel_coords[vid * 3 + 0]] = -1;
    }
    return num_voxels;
  }

  // PillarFeatureNet dekoráció (legacy=False, with_distance=False):
  // [raw F] + [xyz - pillar átlag] + [xyz - pillar középpont], a padding sorok 0-k.
  void decorate(int num_voxels)
  {
    const auto & r = cfg.point_cloud_range;
    const auto & v = cfg.voxel_size;
    const int F = cfg.num_point_features;
    const int D = decorated_features;
    const int MP = cfg.max_points_per_voxel;
    const float x_off = v[0] / 2.f + r[0];
    const float y_off = v[1] / 2.f + r[1];
    const float z_off = v[2] / 2.f + r[2];

    std::fill(pfe_input.begin(), pfe_input.begin() + static_cast<size_t>(num_voxels) * MP * D, 0.0f);

    for (int vid = 0; vid < num_voxels; ++vid) {
      const int n = voxel_num_points[vid];
      const float * vp = voxel_points.data() + static_cast<size_t>(vid) * MP * F;
      float mx = 0.f, my = 0.f, mz = 0.f;
      for (int k = 0; k < n; ++k) {
        mx += vp[k * F + 0];
        my += vp[k * F + 1];
        mz += vp[k * F + 2];
      }
      mx /= n; my /= n; mz /= n;
      const float cx = voxel_coords[vid * 3 + 0] * v[0] + x_off;
      const float cy = voxel_coords[vid * 3 + 1] * v[1] + y_off;
      const float cz = voxel_coords[vid * 3 + 2] * v[2] + z_off;

      float * out = pfe_input.data() + static_cast<size_t>(vid) * MP * D;
      for (int k = 0; k < n; ++k) {
        const float * p = vp + k * F;
        float * o = out + k * D;
        for (int f = 0; f < F; ++f) {
          o[f] = p[f];
        }
        o[F + 0] = p[0] - mx;
        o[F + 1] = p[1] - my;
        o[F + 2] = p[2] - mz;
        o[F + 3] = p[0] - cx;
        o[F + 4] = p[1] - cy;
        o[F + 5] = p[2] - cz;
      }
    }
  }

  std::vector<Detection3D> run(const float * pts, size_t n)
  {
    timing = Timing{};
    auto t0 = Clock::now();
    const int num_voxels = voxelize(pts, n);
    decorate(num_voxels);
    auto t1 = Clock::now();
    timing.voxelize_ms = elapsedMs(t0, t1);
    timing.num_voxels = num_voxels;

    if (num_voxels == 0) {
      return {};
    }

    // ---- PFE ----
    const int MP = cfg.max_points_per_voxel;
    const int D = decorated_features;
    const int C = cfg.pfe_out_channels;
    std::array<int64_t, 3> pfe_shape{num_voxels, MP, D};
    Ort::Value pfe_in = Ort::Value::CreateTensor<float>(
      mem, pfe_input.data(), static_cast<size_t>(num_voxels) * MP * D, pfe_shape.data(), pfe_shape.size());
    const char * pfe_in_names[] = {pfe_input_name.c_str()};
    const char * pfe_out_names[] = {pfe_output_name.c_str()};
    auto pfe_out = pfe->Run(Ort::RunOptions{nullptr}, pfe_in_names, &pfe_in, 1, pfe_out_names, 1);
    const float * pillar_feats = pfe_out[0].GetTensorData<float>();
    {
      auto shape = pfe_out[0].GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 2 || shape[0] != num_voxels || shape[1] != C) {
        throw std::runtime_error("CenterPointDetector: váratlan PFE kimeneti alak");
      }
    }
    auto t2 = Clock::now();
    timing.pfe_ms = elapsedMs(t1, t2);

    // ---- PointPillarsScatter ----
    // A canvas NHWC elrendezésű ([1, ny, nx, C]); az NCHW-ra permutálás az ONNX
    // gráf első művelete. Így pillarenként egyetlen folytonos C-elemű másolás
    // kell, ami sokkal gyorsabb, mint 64 külön síkba szórva írni.
    for (int cell : used_cells) {
      std::fill_n(canvas.data() + static_cast<size_t>(cell) * C, C, 0.0f);
    }
    used_cells.clear();
    for (int vid = 0; vid < num_voxels; ++vid) {
      const int cell = voxel_coords[vid * 3 + 1] * nx + voxel_coords[vid * 3 + 0];
      used_cells.push_back(cell);
      std::copy_n(pillar_feats + static_cast<size_t>(vid) * C, C, canvas.data() + static_cast<size_t>(cell) * C);
    }
    auto t3 = Clock::now();
    timing.scatter_ms = elapsedMs(t2, t3);

    // ---- backbone + neck + head ----
    std::array<int64_t, 4> head_shape{1, ny, nx, C};  // NHWC
    Ort::Value head_in = Ort::Value::CreateTensor<float>(
      mem, canvas.data(), canvas.size(), head_shape.data(), head_shape.size());
    const char * head_in_names[] = {head_input_name.c_str()};
    std::vector<const char *> head_out_names;
    for (const auto & s : head_output_names) {
      head_out_names.push_back(s.c_str());
    }
    auto head_out = head->Run(
      Ort::RunOptions{nullptr}, head_in_names, &head_in, 1, head_out_names.data(), head_out_names.size());
    auto t4 = Clock::now();
    timing.head_ms = elapsedMs(t3, t4);

    // ---- dekódolás ----
    auto dets = decode(head_out);
    timing.decode_ms = elapsedMs(t4, Clock::now());
    return dets;
  }

  std::vector<Detection3D> decode(std::vector<Ort::Value> & outs)
  {
    std::vector<Detection3D> result;
    const int HW = feat_h * feat_w;
    const float step_x = cfg.out_size_factor * cfg.voxel_size[0];
    const float step_y = cfg.out_size_factor * cfg.voxel_size[1];
    const auto & pcr = cfg.post_center_range;
    int label_offset = 0;

    for (size_t t = 0; t < cfg.task_num_classes.size(); ++t) {
      const int num_cls = cfg.task_num_classes[t];
      const float * heat = outs[t * kOutputsPerTask + 0].GetTensorData<float>();
      const float * reg = outs[t * kOutputsPerTask + 1].GetTensorData<float>();
      const float * hei = outs[t * kOutputsPerTask + 2].GetTensorData<float>();
      const float * dim = outs[t * kOutputsPerTask + 3].GetTensorData<float>();
      const float * rot = outs[t * kOutputsPerTask + 4].GetTensorData<float>();
      const float * vel = outs[t * kOutputsPerTask + 5].GetTensorData<float>();

      auto hshape = outs[t * kOutputsPerTask].GetTensorTypeAndShapeInfo().GetShape();
      if (hshape.size() != 4 || hshape[1] != num_cls || hshape[2] != feat_h || hshape[3] != feat_w) {
        throw std::runtime_error("CenterPointDetector: váratlan heatmap alak a(z) " + std::to_string(t) + ". taskban");
      }

      // top-K (mmdet3d: osztályonkénti, majd globális top-K) == globális top-K a
      // küszöb feletti (osztály, cella) párokon; a heatmap már szigmoidos (export).
      cands.clear();
      for (int c = 0; c < num_cls; ++c) {
        const float * hc = heat + static_cast<size_t>(c) * HW;
        for (int i = 0; i < HW; ++i) {
          if (hc[i] > cfg.score_threshold) {
            cands.push_back({hc[i], c, i});
          }
        }
      }
      auto by_score = [](const Candidate & a, const Candidate & b) {return a.score > b.score;};
      if (static_cast<int>(cands.size()) > cfg.max_num_per_task) {
        std::partial_sort(cands.begin(), cands.begin() + cfg.max_num_per_task, cands.end(), by_score);
        cands.resize(cfg.max_num_per_task);
      } else {
        std::sort(cands.begin(), cands.end(), by_score);
      }

      std::vector<Detection3D> task_dets;
      task_dets.reserve(cands.size());
      for (const auto & cd : cands) {
        const int iy = cd.idx / feat_w;
        const int ix = cd.idx % feat_w;
        Detection3D d;
        d.x = (ix + reg[cd.idx]) * step_x + cfg.point_cloud_range[0];
        d.y = (iy + reg[HW + cd.idx]) * step_y + cfg.point_cloud_range[1];
        d.z = hei[cd.idx];  // a head a doboz középpontjának z-jét adja
        d.length = std::exp(dim[cd.idx]);
        d.width = std::exp(dim[HW + cd.idx]);
        d.height = std::exp(dim[2 * HW + cd.idx]);
        d.yaw = std::atan2(rot[cd.idx], rot[HW + cd.idx]);
        d.vx = vel[cd.idx];
        d.vy = vel[HW + cd.idx];
        d.score = cd.score;
        d.label = label_offset + cd.cls;
        if (d.x < pcr[0] || d.y < pcr[1] || d.z < pcr[2] || d.x > pcr[3] || d.y > pcr[4] || d.z > pcr[5]) {
          continue;
        }
        task_dets.push_back(d);
      }

      // circle NMS (mmdet3d.models.layers.circle_nms): négyzetes középpont-távolság <= küszöb
      const float thr = cfg.circle_nms_min_radius[t];
      std::vector<char> suppressed(task_dets.size(), 0);
      int kept = 0;
      for (size_t i = 0; i < task_dets.size() && kept < cfg.post_max_size; ++i) {
        if (suppressed[i]) {
          continue;
        }
        result.push_back(task_dets[i]);
        ++kept;
        for (size_t j = i + 1; j < task_dets.size(); ++j) {
          if (suppressed[j]) {
            continue;
          }
          const float dx = task_dets[i].x - task_dets[j].x;
          const float dy = task_dets[i].y - task_dets[j].y;
          if (dx * dx + dy * dy <= thr) {
            suppressed[j] = 1;
          }
        }
      }
      label_offset += num_cls;
    }
    return result;
  }
};

CenterPointDetector::CenterPointDetector(const CenterPointConfig & cfg)
: impl_(std::make_unique<Impl>(cfg))
{
}

CenterPointDetector::~CenterPointDetector() = default;

std::vector<Detection3D> CenterPointDetector::detect(const float * points, std::size_t num_points)
{
  return impl_->run(points, num_points);
}

const CenterPointDetector::Timing & CenterPointDetector::lastTiming() const
{
  return impl_->timing;
}

bool CenterPointDetector::usingCuda() const
{
  return impl_->cuda;
}

}  // namespace perception
