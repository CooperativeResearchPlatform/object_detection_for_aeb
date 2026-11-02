#include "perception/brake_decision.hpp"

#include <algorithm>
#include <cmath>

namespace perception
{

BrakeDecision::BrakeDecision(const BrakeDecisionConfig & cfg)
: cfg_(cfg)
{
}

void BrakeDecision::reset()
{
  on_count_ = 0;
  off_count_ = 0;
  state_ = false;
}

bool BrakeDecision::shouldBrake(const Candidate & c, float & ttc, float & decel) const
{
  ttc = std::numeric_limits<float>::infinity();
  decel = 0.0f;
  if (c.closing_speed > 0.1f) {
    ttc = c.distance / c.closing_speed;
    decel = c.closing_speed * c.closing_speed / (2.0f * std::max(c.distance, 0.1f));
  }
  return c.distance < cfg_.min_distance_m ||
         ttc < cfg_.ttc_threshold_s ||
         decel > cfg_.max_comfort_decel;
}

BrakeAssessment BrakeDecision::evaluate(
  const std::vector<Detection3D> & dets,
  const float * xyz, std::size_t n, std::size_t stride,
  float ego_speed_mps)
{
  BrakeAssessment out;
  const float x_min = cfg_.front_offset_m;
  const float x_max = cfg_.front_offset_m + cfg_.max_range_m;
  const float hw = cfg_.half_width_m;

  auto consider = [&](float distance, float closing, BrakeAssessment::Source src, int idx) {
      out.obstacle_in_path = true;
      float ttc = 0.f, decel = 0.f;
      const bool brake = shouldBrake({distance, closing}, ttc, decel);
      // a fékezést kiváltó, azon belül a legközelebbi jelölt a "legkritikusabb"
      const bool better =
        (brake && !out.brake_raw) ||
        (brake == out.brake_raw && distance < out.distance_m);
      if (better) {
        out.brake_raw = out.brake_raw || brake;
        out.distance_m = distance;
        out.ttc_s = ttc;
        out.required_decel = decel;
        out.source = src;
        out.detection_index = idx;
      }
    };

  // 1) Detektor alapú ellenőrzés: a doboz alaprajza metszi-e a folyosót
  for (std::size_t i = 0; i < dets.size(); ++i) {
    const auto & d = dets[i];
    if (d.score < cfg_.min_score) {
      continue;
    }
    const float c = std::cos(d.yaw), s = std::sin(d.yaw);
    const float hl = d.length * 0.5f, hwd = d.width * 0.5f;
    float bx_min = 1e9f, bx_max = -1e9f, by_min = 1e9f, by_max = -1e9f;
    for (int sx = -1; sx <= 1; sx += 2) {
      for (int sy = -1; sy <= 1; sy += 2) {
        const float px = d.x + c * sx * hl - s * sy * hwd;
        const float py = d.y + s * sx * hl + c * sy * hwd;
        bx_min = std::min(bx_min, px); bx_max = std::max(bx_max, px);
        by_min = std::min(by_min, py); by_max = std::max(by_max, py);
      }
    }
    const bool overlaps = bx_max >= x_min && bx_min <= x_max && by_max >= -hw && by_min <= hw;
    if (!overlaps) {
      continue;
    }
    const float distance = std::max(0.0f, bx_min - x_min);
    const float obj_vx = cfg_.use_object_velocity ? d.vx : 0.0f;
    consider(distance, ego_speed_mps - obj_vx, BrakeAssessment::Source::Detector, static_cast<int>(i));
  }

  // 2) Geometriai tartalék: talaj feletti pontok a folyosó közeli részén
  if (cfg_.geometric_check && xyz != nullptr && n > 0) {
    const float z_lo = -cfg_.sensor_height_m + cfg_.min_obstacle_height_m;
    const float z_hi = -cfg_.sensor_height_m + cfg_.max_obstacle_height_m;
    const float gx_max = cfg_.front_offset_m + cfg_.geometric_max_range_m;
    scratch_.clear();
    for (std::size_t i = 0; i < n; ++i) {
      const float * p = xyz + i * stride;
      if (p[0] >= x_min && p[0] <= gx_max && std::fabs(p[1]) <= hw && p[2] >= z_lo && p[2] <= z_hi) {
        scratch_.push_back(p[0]);
      }
    }
    const auto k = static_cast<std::size_t>(std::max(1, cfg_.min_obstacle_points));
    if (scratch_.size() >= k) {
      // a k-adik legközelebbi pont: néhány zajpont nem okoz fékezést
      std::nth_element(scratch_.begin(), scratch_.begin() + (k - 1), scratch_.end());
      const float distance = std::max(0.0f, scratch_[k - 1] - x_min);
      consider(distance, ego_speed_mps, BrakeAssessment::Source::Geometric, -1);
    }
  }

  // 3) Hiszterézis
  if (out.brake_raw) {
    ++on_count_;
    off_count_ = 0;
    if (on_count_ >= cfg_.frames_to_trigger) {
      state_ = true;
    }
  } else {
    ++off_count_;
    on_count_ = 0;
    if (off_count_ >= cfg_.frames_to_release) {
      state_ = false;
    }
  }
  out.brake = state_;
  return out;
}

}  // namespace perception
