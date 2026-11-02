#ifndef PERCEPTION__BRAKE_DECISION_HPP_
#define PERCEPTION__BRAKE_DECISION_HPP_

// Determinisztikus döntési réteg a detektor KIMENETE fölött.
// A neurális háló csak objektumokat lát; azt, hogy ezek a jármű pályájában
// vannak-e és kell-e fékezni, ez a modul dönti el, ellenőrizhető szabályokkal.
// Minden koordináta a SZENZOR frame-ben értendő, REP-103 szerint:
// x előre, y balra, z fel.

#include <cstddef>
#include <limits>
#include <vector>

#include "perception/centerpoint_detector.hpp"

namespace perception
{

struct BrakeDecisionConfig
{
  // Folyosó a jármű előtt
  float front_offset_m = 1.0f;     // szenzor origó -> első lökhárító (x irányban)
  float half_width_m = 1.2f;       // fél járműszélesség + oldalsó ráhagyás
  float max_range_m = 40.0f;       // ennél távolabbi objektumokkal nem foglalkozunk
  float min_score = 0.35f;         // döntéshez használt minimális detektor score

  // Fékezési szabályok (bármelyik teljesül -> fék)
  float min_distance_m = 3.0f;     // ennél közelebb mindig fék
  float ttc_threshold_s = 2.0f;    // time-to-collision küszöb
  float max_comfort_decel = 3.0f;  // m/s^2; ha ennél nagyobb lassulás kellene -> fék
  bool use_object_velocity = false;  // egy sweepből a sebesség megbízhatatlan, alapból statikusnak vesszük

  // Modelltől független geometriai tartalék (ismeretlen akadályokra is reagál)
  bool geometric_check = true;
  float sensor_height_m = 1.8f;     // a szenzor magassága a talaj felett
  float geometric_max_range_m = 20.0f;  // sík talaj feltevés csak közel érvényes
  float min_obstacle_height_m = 0.3f;   // talaj feletti magasság-sáv
  float max_obstacle_height_m = 2.5f;
  int min_obstacle_points = 15;

  // Hiszterézis a fantomfékezés és a "villogás" ellen
  int frames_to_trigger = 2;
  int frames_to_release = 5;
};

struct BrakeAssessment
{
  enum class Source { None, Detector, Geometric };

  bool obstacle_in_path = false;  // van valami a folyosóban (szűretlen)
  bool brake_raw = false;         // az aktuális frame alapján fékezni kéne
  bool brake = false;             // hiszterézis utáni kimenet
  float distance_m = std::numeric_limits<float>::infinity();
  float ttc_s = std::numeric_limits<float>::infinity();
  float required_decel = 0.0f;
  Source source = Source::None;
  int detection_index = -1;       // Detector forrásnál az objektum indexe
};

class BrakeDecision
{
public:
  explicit BrakeDecision(const BrakeDecisionConfig & cfg);

  // dets: szenzor frame-ben (x előre); xyz: n darab pont, stride float-onként.
  BrakeAssessment evaluate(
    const std::vector<Detection3D> & dets,
    const float * xyz, std::size_t n, std::size_t stride,
    float ego_speed_mps);

  void reset();

private:
  struct Candidate
  {
    float distance;
    float closing_speed;
  };
  bool shouldBrake(const Candidate & c, float & ttc, float & decel) const;

  BrakeDecisionConfig cfg_;
  int on_count_ = 0;
  int off_count_ = 0;
  bool state_ = false;
  std::vector<float> scratch_;
};

}  // namespace perception

#endif  // PERCEPTION__BRAKE_DECISION_HPP_
